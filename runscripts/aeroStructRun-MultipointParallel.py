"""
==============================================================================
Multipoint parallel Aerostructural runscript
==============================================================================
@File    :   aeroStructRun-MultipointParallel.py
@Date    :   2023/11/22
@Author  :   Alasdair Christison Gray
@Description : This script uses a hybrid MACH/MPhys approach, it uses MACH's multipoint package to create a separate
processor group for each flightpoint, and then creates a separate single point MPhys  model in each group. A separate
OpenMDAO model is created for the performance group which is called in the Multipoint objcon function.

The advantages of this approach are:
- The pyoptsparse history file contains a more completye set of data, not just the constraint, objective and DV values
- The values in the pyoptsparse history file are not scaled
- When everything is in one OpenMDAO model and there are multiple constraints/objective that depend on one of the
aerostructural outputs (e.g cruise lift), the coupled adjoint for this function is computed repeatedly for each
objective/constraint, this is completely redundant and slows down the optimisation by a factor of 2-3. The hybrid
approach avoids this due to the Multipoint packages approach of splitting the function evaluation into the expensive and
cheap (objcon) functions.
- Personally, I find the process of setting up processor groups with Multipoint simpler than setting up OpenMDAO
parallel multipoint groups and their parallel cderivative colouring.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import sys
from pprint import pprint as pp
import time

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from mpi4py import MPI
import openmdao.api as om
from mphys import Multipoint
from mphys.scenario_aerostructural import ScenarioAeroStructural
from adflow.mphys import ADflowBuilder
from tacs.mphys import TacsBuilder
from tacs.mphys.utils import add_tacs_constraints
from tacs import TACS
from funtofem.mphys import MeldBuilder
from pygeo.mphys import OM_DVGEOCOMP
import dill  # A better version of pickle
from baseclasses.utils import redirectIO
from multipoint import multiPointSparse
from pyoptsparse import Optimization, OPT


# ==============================================================================
# Extension modules
# ==============================================================================
import SETUP.setupTACS as setupTACS
from SETUP.setupDVGeo import setupDVGeo
from SETUP.setupADflow import getADflowOptions
from SETUP.setupIDWarp import getIDWarpOptions
from CommonArgs import parser
from OptimiserOptions import getOptOptions
from utils import (
    getOutputDir,
    getStructMeshPath,
    getAeroMeshPath,
    getFFDPath,
    setValsFromFiles,
    saveRunCommand,
    get_prom_name,
    addConstraintFromOpenMDAO,
)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import performanceCalc  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../AircraftSpecs"))
from STWFlightPoints import flightPointSets  # noqa: E402
from STWSpecs import aircraftSpecs  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../geometry"))
from wingGeometry import wingGeometry  # noqa: E402

# --- Get the start time ---
startTime = time.time()


# --- Get some info on the wing geometry ---
SPAN_INDEX = wingGeometry["spanIndex"]
CHOARD_INDEX = wingGeometry["chordIndex"]
VERTICAL_INDEX = wingGeometry["verticalIndex"]
INDEX_STRINGS = ["x", "y", "z"]
WING_SEMISPAN = wingGeometry["wing"]["semiSpan"]

# --- Figure out where to put the results ---
OUTPUT_PARENT_DIR = getOutputDir()

# set these for convenience
globalComm = MPI.COMM_WORLD
globalRank = globalComm.rank
isComplex = TACS.dtype == complex


# --- General options ---
parser.add_argument(
    "--task",
    type=str,
    default="check",
    choices=["check", "analysis", "derivCheck", "opt", "trim"],
    help="Task to run",
)
parser.add_argument("--flightPointSet", type=str, default="cruise", choices=list(flightPointSets.keys()))
parser.add_argument(
    "--procs",
    type=int,
    nargs="*",
    default=[],
    help="Number of processors to use for each flight point, if not specified then processors will be split evenly between flight points",
)

# --- Coupled solver options ---
parser.add_argument(
    "--noAitken", action="store_true", help="Don't use Aitken acceleration in the coupled aerostructural solver"
)

# --- Optimisation options ---
parser.add_argument("--rangeScale", type=float, default=1.0, help="Factor to scale the mission range by")
parser.add_argument("--maxWingLoading", type=float, default=600.0, help="Maximum allowable wing loading (kg/m^2)")
parser.add_argument(
    "--optType",
    type=str,
    choices=["fuelburn", "structMass"],
    default="fuelburn",
    help="Type of optimisation to perform, 'fuelburn' for fuelburn minimisation, 'structMass' for structural mass minimisation with only maneuver flight condition",
)

# --- Aero options ---
parser.add_argument("--aeroLevel", type=int, default=3, choices=[1, 2, 3])

# --- LDTransfer options ---
parser.add_argument(
    "--transferTo",
    type=str,
    default="skin+ribends",
    choices=["all", "skin", "skin+spar", "skin+ribends"],
    help="Which structural nodes to include in the LDTransfer",
)

args = parser.parse_args()

# If we're doing a derivative check, we should at least enable twist and structural DVs
if args.task == "derivCheck":
    args.addGeoDVs = True
    args.twist = True
    args.addStructDVs = True

# If we are doing a trim task then we should disable the structural and geometric design variables
if args.task == "trim":
    args.addStructDVs = False
    args.addGeoDVs = False

# If we're running the setup check, enable everything
if args.task == "check":
    args.addGeoDVs = True
    args.addStructDVs = True
    args.usePanelLengthDVs = True
    args.twist = True
    args.sweep = True
    args.shape = True
    args.span = True
    args.dihedral = True
    args.taper = True


# Define flight points to use
flightPoints = flightPointSets[args.flightPointSet]
# Set reference geometric values
for fp in flightPoints:
    fp.areaRef = aircraftSpecs["refArea"]
    fp.chordRef = aircraftSpecs["refChord"]

flightPointsDict = {fp.name: fp for fp in flightPoints}

# Can't run fuelburn minimisation without a cruise flight point
hasCruisePoint = any(["cruise" in fp.name.lower() for fp in flightPoints])
if args.task == "opt" and args.optType == "fuelburn" and not hasCruisePoint:
    raise ValueError("Cannot run fuelburn minimisation without a cruise flight point")

# Scale the mission range
aircraftSpecs["range"] *= args.rangeScale

# Account for additional drag on coarser mesh due to discretisation error.
# I estimated these values based on the average difference in cd between the L2 and L3 meshes and the L1 over an angle
# of attack range of 0-3 degrees
dragCorrection = {2: 0.0003, 3: 0.0013}
if args.aeroLevel > 1:
    aircraftSpecs["extraDragCoeff"] -= dragCorrection[args.aeroLevel]

# Create output directories
outputDir = os.path.join(OUTPUT_PARENT_DIR, args.output)

# Define location of input files
structMeshFile = getStructMeshPath(level=args.structLevel, order=args.structOrder)

aeroMeshFile = getAeroMeshPath(level=args.aeroLevel)

ffdFile = getFFDPath(level=args.ffdLevel)

structOnlyOpt = args.task == "opt" and args.optType == "structMass"

structMeshSpacing = {
    1: 0.035,
    2: 0.07,
    3: 0.14,
    4: 0.23,
}

aeroMeshChordSpacing = {
    1: 0.06005,
    2: 0.116,
    3: 0.2447,
}
aeroMeshSpanSpacing = {
    1: 0.157,
    2: 0.312,
    3: 0.60,
}

# This an approximation of the ratio of structural nodes to aero nodes in the coarsest part of the aero mesh, used to
# tell MELD how many structural nodes to connect each aero node to. Multiplying this estimate by 4 seems to
# provide a reasonable N value where the aero forces are not concentrated at the nearest structural nodes.
MELD_MESH_FACTOR = max(
    100,
    int(
        4
        * aeroMeshSpanSpacing[args.aeroLevel]
        * aeroMeshChordSpacing[args.aeroLevel]
        / structMeshSpacing[args.structLevel] ** 2
    ),
)
# The Super fine aero mesh is particularly prone to negative volumes due to structural deformations so we use a larger
# lower limit on N for it
if args.aeroLevel == 1:
    MELD_MESH_FACTOR = max(200, MELD_MESH_FACTOR)


# ==============================================================================
# Processor allocation
# ==============================================================================
#  Create multipoint communication object
MP = multiPointSparse(globalComm)

# We will use a single processor set which contains each flight point as a separate member
nMembers = len(flightPoints)
if len(args.procs) == 0:
    # If no processors are specified, split them evenly between the members
    procsPerMember = [globalComm.size // nMembers] * nMembers
    extraProcs = globalComm.size - sum(procsPerMember)
    for ii in range(1, extraProcs + 1):
        procsPerMember[-ii] += 1
else:
    if len(args.procs) != nMembers:
        raise ValueError(f"You specified {nMembers} flight points but only {len(args.procs)} processor counts")
    else:
        procsPerMember = args.procs
MP.addProcessorSet("all", nMembers=nMembers, memberSizes=procsPerMember)

# Create communicators
# ptComm : MPI.Intracomm
#     This is the communicator for the member of the procSet. Basically,
#     this is the communciator that the (parallel) analysis should be
#     created on
# setComm : MPI.Intracomm
#     This is the communicator that spans the entire processor set.
# setFlags : dict
#     This is a dictionary whose entry for \"setName\", as specified in
#     addProcessorSet() is True on a processor belonging to that set.
# groupFlags : list
#     This is list is used to distinguish between members within
#     a processor set. This list of of length nMembers and the
#     ith entry is true for the ith group.
# ptID : int
#     This is the index of the group that this processor belongs to
ptComm, setComm, setFlags, groupFlags, ptID = MP.createCommunicators()
ptRank = ptComm.rank

# For convenience, store which flightPoint we're working with on this proc
localFlightPoint = flightPoints[ptID]

# Create output directories
localOutputDir = os.path.join(outputDir, localFlightPoint.name)
localAeroOutputDir = os.path.join(localOutputDir, "aero")
localStructOutputDir = os.path.join(localOutputDir, "struct")
if ptRank == 0:
    os.makedirs(localAeroOutputDir, exist_ok=True)
    os.makedirs(localStructOutputDir, exist_ok=True)

# Print out the full list of command line arguments
saveRunCommand(parser, args, outputDir)

for ii in range(globalComm.size):
    if globalRank == ii:
        print(f"Processor {globalRank} is assigned to flight point {localFlightPoint.name}")
    globalComm.Barrier()

# --- Redirect I/O ---
if ptRank == 0:
    outFile = open(os.path.join(localOutputDir, "stdout.out"), "w")
    redirectIO(outFile)
    print("===============================================================================")
    print(localFlightPoint.name)
    print("===============================================================================")

# ==============================================================================
# TACS Setup
# ==============================================================================
problemOptions = {
    "outputDir": localStructOutputDir,
    "useMonitor": not args.nonlinear,
    "monitorFrequency": 1,
}
nonlinearOptions = {"writeNLIterSolutions": False, "printLevel": 1}
newtonOptions = None
continuationOptions = None
if args.nonlinear:
    problemOptions.update(nonlinearOptions)
    newtonOptions = {
        "UseEW": True,
        "MaxLinIters": 10,
        "skipFirstNLineSearch": 1,
        "ForceFirstIter": True,
        "maxIter": 20,
    }
    continuationOptions = {
        "RelTol": 1e-7,
        "UsePredictor": True,
        "NumPredictorStates": 4,
        "maxIter": 10,
    }


def setup_assembler(assembler):
    return setupTACS.setup_tacs_assembler(assembler, args)


def setup_tacs_problem(scenario_name, fea_assembler, problem):
    flightPoint = flightPointsDict[scenario_name]
    setupTACS.problem_setup(
        scenario_name,
        flightPoint,
        fea_assembler,
        problem,
        args=args,
        options=problemOptions,
        newtonOptions=newtonOptions,
        continuationOptions=continuationOptions,
    )


# Read in panel lengths, they're stored in a csv file along with the struct meshes
panelLengthFileName = os.path.join(os.path.dirname(structMeshFile), "PanelLengths.csv")
with open(panelLengthFileName, "r") as panelLengthFile:
    panelLengths = {line.split(",")[0]: float(line.split(",")[1]) for line in panelLengthFile}


def element_callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
    return setupTACS.element_callback(
        dvNum,
        compID,
        compDescript,
        elemDescripts,
        specialDVs,
        args,
        nonlinear=args.nonlinear,
        useComposite=args.useComposite,
        usePanelLengthDVs=args.usePanelLengthDVs,
        usePlyFractionDVs=args.usePlyFractionDVs,
        useStiffenerPitchDVs=args.useStiffPitchDVs,
        panelLengths=panelLengths,
        **kwargs,
    )


def constraint_callback(scenario_name, fea_assembler, constraints):
    if scenario_name == flightPoints[0].name:
        return setupTACS.setupConstraints(scenario_name, fea_assembler, constraints, args)
    return None


struct_builder = TacsBuilder(
    mesh_file=structMeshFile,
    assembler_setup=setup_assembler,
    element_callback=element_callback,
    constraint_setup=constraint_callback,
    problem_setup=setup_tacs_problem,
    coupled=True,
    write_solution=False,
    res_ref=1e3,
)

# ==============================================================================
# ADflow/getIDWarp Setup
# ==============================================================================
aero_options = getADflowOptions(aeroMeshFile, localAeroOutputDir, aerostructural=True)
if args.aeroLevel == 3:
    aero_options["anksecondordswitchtol"] *= 10
    aero_options["rkreset"] = True
if args.noFiles:
    aero_options["writeTecplotSurfaceSolution"] = False
    aero_options["writevolumesolution"] = False
    aero_options["writesurfacesolution"] = False
warp_options = getIDWarpOptions(aeroMeshFile)
aero_builder = ADflowBuilder(
    aero_options,
    mesh_options=warp_options,
    scenario="aerostructural",
    write_solution=False,
    res_ref=1e7,
    restart_failed_analysis=False,
)


# ==============================================================================
# Transfer Scheme Setup
# ==============================================================================
# --- First define which nodes in the aero and struct meshes will be involved in the transfer ---
if args.transferTo == "all":
    ldTransferBodies = None
elif args.transferTo in ["skin", "skin+ribends"]:  # The rib-spar intersection nodes will be added later
    ldTransferBodies = [{"aero": ["wall"], "struct": ["SKIN"]}]
elif args.transferTo == "skin+spar":
    ldTransferBodies = [{"aero": ["wall"], "struct": ["SKIN", "SPAR"]}]

isym = SPAN_INDEX  # spanwise-symmetry

ldxfer_builder = MeldBuilder(
    aero_builder,
    struct_builder,
    isym=isym,
    n=MELD_MESH_FACTOR,
    linearized=not args.nonlinear,
    body_tags=ldTransferBodies,
)


# Now we can create the MPhys model for each flight point
class AerostructuralFlightPoint(Multipoint):
    def setup(self):
        # ivc to keep the top level DVs
        dvs = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # ==============================================================================
        # MPHYS Setup
        # ==============================================================================
        scenarioName = localFlightPoint.name

        # --- initialize aero builder ---
        aero_builder.initialize(self.comm)
        aeroSolver = aero_builder.get_solver()

        # Add lift distribution and slice file output
        if not args.noFiles:
            aeroSolver.addLiftDistribution(100, INDEX_STRINGS[SPAN_INDEX])
            slicePositions = np.linspace(1e-5, WING_SEMISPAN * 0.99, 11)
            aeroSolver.addSlices(INDEX_STRINGS[SPAN_INDEX], slicePositions)

        # ==============================================================================
        # TACS Setup
        # ==============================================================================
        struct_builder.initialize(self.comm)

        structDVMap = setupTACS.buildStructDVDictMap(struct_builder.get_fea_assembler(), args)
        if globalRank == 0:
            with open(os.path.join(outputDir, "structDVMap.pkl"), "wb") as structDVMapFile:
                dill.dump(structDVMap, structDVMapFile)

        # --- Add TACS dvs ---
        init_dvs = struct_builder.get_initial_dvs()
        dvs.add_output("dv_struct", init_dvs)
        if args.addStructDVs:
            lb, ub = struct_builder.get_dv_bounds()
            structDVScaling = np.array(struct_builder.fea_assembler.scaleList)
            self.add_design_var("dv_struct", lower=lb, upper=ub, scaler=structDVScaling * args.structScalingFactor)

        # ==============================================================================
        # Mesh and geometry setup
        # ==============================================================================
        self.add_subsystem("mesh_aero", aero_builder.get_mesh_coordinate_subsystem())
        self.add_subsystem("mesh_struct", struct_builder.get_mesh_coordinate_subsystem())
        geometryComp = OM_DVGEOCOMP(file=ffdFile, type="ffd", options={"isComplex": isComplex})
        self.add_subsystem("geometry", geometryComp)

        # Setup mesh components for each discipline
        for discipline in ["aero", "struct"]:
            # Tell the geometry component that there will be a set of coordinates for the discipline
            geometryComp.nom_add_discipline_coords(discipline)
            # Connect the original mesh coordinates as an input to the geometry component
            self.connect(f"mesh_{discipline}.x_{discipline}0", f"geometry.x_{discipline}_in")

        # --- initialize MELD ---
        # Find the nodes at the intersections of the spars and ribs and include them in the LDTransfer
        if args.transferTo == "skin+ribends":
            FEASolver = struct_builder.get_fea_assembler()
            sparComps = FEASolver.selectCompIDs(include="SPAR")
            ribComps = FEASolver.selectCompIDs(include="RIB")
            sparNodes = FEASolver.getGlobalNodeIDsForComps(sparComps, nastranOrdering=True)
            ribNodes = FEASolver.getGlobalNodeIDsForComps(ribComps, nastranOrdering=True)
            sharedNodes = list(set(sparNodes).intersection(set(ribNodes)))
            ldxfer_builder.body_tags[0]["struct"] += sharedNodes
        ldxfer_builder.initialize(self.comm)

        # ==============================================================================
        # Create the scenario
        # ==============================================================================
        self.mphys_add_scenario(
            scenarioName,
            ScenarioAeroStructural(
                aero_builder=aero_builder,
                struct_builder=struct_builder,
                ldxfer_builder=ldxfer_builder,
            ),
        )

        for discipline in ["aero", "struct"]:
            self.mphys_connect_scenario_coordinate_source("geometry", scenarioName, discipline)
        self.connect("dv_struct", f"{scenarioName}.dv_struct")

    def configure(self):
        aeroMeshComp = self.mesh_aero
        geometryComp = self.geometry
        dvComp = self.dvs

        # Give ADflow the aero problem for this procset's flight point and add the angle of attack as a DV
        fp = localFlightPoint
        scenario = getattr(self, fp.name)
        fp.addDV("alpha", value=fp.alpha, name="aoa", units="deg")
        scenario.coupling.aero.mphys_set_ap(fp)
        scenario.aero_post.mphys_set_ap(fp)
        alphaDVName = f"{fp.name}_AOA"
        dvComp.add_output(alphaDVName, val=fp.alpha, units="deg")
        self.add_design_var(alphaDVName, lower=-20.0, upper=20.0, scaler=1.0)
        self.connect(alphaDVName, [f"{fp.name}.coupling.aero.aoa", f"{fp.name}.aero_post.aoa"])

        # We will compute the geometric constraints only on the first proc set, for this we need to get the triangulated
        # OML surface
        if ptID == 0:
            aeroTriSurf = aeroMeshComp.mphys_get_triangulated_surface()
            geometryComp.nom_setConstraintSurface(aeroTriSurf)

        # Setup the geometric DVs and constraints
        setupDVGeo(
            args,
            self,
            geometryComp,
            dvComp,
            dvScaleFactor=args.geoScalingFactor,
            geoCompName="geometry",
            addGeoDVs=args.addGeoDVs,
            addGeoConstraints=ptID == 0,
        )

        # Similarly, only add TACS constraints on the first proc set
        if ptID == 0 and args.addStructDVs:
            firstScenario = getattr(self, localFlightPoint.name)
            add_tacs_constraints(firstScenario)

        # ==============================================================================
        # Play with the coupled solver settings
        # ==============================================================================
        scenario = getattr(self, fp.name)
        scenario.coupling.nonlinear_solver = om.NonlinearBlockGS(
            maxiter=50,
            iprint=2,
            atol=1e-4 * args.tolFactor,
            rtol=1e-8 * args.tolFactor,
            use_aitken=not args.noAitken,
            aitken_initial_factor=0.5,
            aitken_max_factor=1.2,
            # reraise_child_analysiserror=True,
            # use_apply_nonlinear=True, This doesn't work
            restart_from_successful=True,
            err_on_non_converge=True,
        )
        scenario.coupling.linear_solver = om.PETScKrylov(
            atol=1e-4 * args.tolFactor, rtol=1e-8 * args.tolFactor, maxiter=50, iprint=2
        )
        scenario.coupling.linear_solver.precon = om.LinearBlockGS(maxiter=1, iprint=-2, use_aitken=False, rtol=1e-1)


# --- Now actually create the OpenMDAO model for each point ---
flightPointProb = om.Problem(reports=None, comm=ptComm)
flightPointProb.model = AerostructuralFlightPoint()

# --- Finally create the aircraft performance OpenMDAO model ---
performanceProb = om.Problem(reports=None, comm=globalComm)
performanceProb.model = performanceCalc.AircraftPerformanceGroup(aircraftSpecs=aircraftSpecs, flightPoints=flightPoints)

if args.task in ["trim", "opt", "check"]:
    # ==============================================================================
    # Setup constraints
    # ==============================================================================

    # --- Lift constraint for each flight point ---
    # Generally, massively scaling down the lift difference helps the optimiser make better progress, but in the trim
    # task, where all we care about is hitting the lift constraints, we won't scale them down quite as much so that we
    # really nail the right lift value.
    liftConScale = 1e-8 if args.task == "trim" else 1e-6
    performanceProb.model.add_constraint(
        f"{localFlightPoint.name}LiftDiff", equals=0.0, scaler=liftConScale, cache_linear_solution=True
    )

    if args.task in ["opt", "check"]:
        # --- TACS Failure constraints ---
        for group in localFlightPoint.failureGroups:
            failureConName = f"{localFlightPoint.name}.{group}_ksFailure"
            flightPointProb.model.add_constraint(failureConName, upper=1.0, scaler=1.0, cache_linear_solution=True)

        # --- Geometric constraints ---
        if not structOnlyOpt:
            # --- Wingbox volume constraint ---
            if args.span or args.taper or args.shape:
                performanceProb.model.add_constraint("fuelTankUsage", upper=1.0, cache_linear_solution=True)
            # --- Wing loading constraint ---
            if args.span or args.taper:
                # This constraint should only be applied if the optimiser has control over the wing planform
                performanceProb.model.add_constraint(
                    "wingLoading",
                    upper=args.maxWingLoading,
                    scaler=1.0 / args.maxWingLoading,
                    cache_linear_solution=True,
                )
    # ==============================================================================
    # Setup objective
    # ==============================================================================
    if ptID == 0:
        if args.task == "trim" or structOnlyOpt:
            # For the trim task we setup a dummy objective that doesn't depend on the trim variables so that the optimiser
            # just satisfies the trim constraints
            performanceProb.model.add_objective("airframeMass.wingMass", scaler=1e-3, cache_linear_solution=True)
        else:
            performanceProb.model.add_objective("TotalFuelBurn", scaler=1e-4, cache_linear_solution=True)


# ==============================================================================
# Setup the OpenMDAO models
# ==============================================================================
# Setup the aerostructural model for each flight point and get the names of their outputs
flightPointProb.setup(force_alloc_complex=isComplex, mode="rev")
tmp = ptComm.bcast(flightPointProb.model.list_outputs(out_stream=None), root=0)
flightPointProbOutputs = {}
for output in tmp:
    flightPointProbOutputs[output[1]["prom_name"]] = output[1]

# Setup the performance model and get the names of its inputs
performanceProb.setup(force_alloc_complex=True)
performanceProb.final_setup()  # TODO: Figure why I need to call this
performanceProbInputs = []
for _, inp in performanceProb.model.list_inputs(out_stream=None):
    promName = inp["prom_name"]
    if promName not in performanceProbInputs and "." not in promName:
        performanceProbInputs.append(promName)
performanceProbInputs = globalComm.bcast(performanceProbInputs, root=0)

# ==============================================================================
# Define grad and disp funcs
# ==============================================================================
# Grad funcs are the functions we need to compute derivatives of because they are objectives/constraints, or used in
# the computation of objectives/constraints. Disp funcs are functions that are not required for the computation of
# objectives/constraints but we want to track and store in the history file anyway
gradFuncs = []

# Define the map from the outputs of the flight point models to the inputs required for the performance calculation
# model
dvMap = {}
for inpName in performanceProbInputs:
    if inpName == "wingboxMass":
        dvMap[inpName] = f"{flightPoints[0].name}.mass"
    elif inpName == "wingboxVolume":
        dvMap[inpName] = "geometry.WingboxVolume"
    elif inpName == "wingArea":
        dvMap[inpName] = "geometry.WingArea"
    elif "Drag" in inpName or "Lift" in inpName:
        fpName = inpName[:-4]
        forceName = inpName[-4:]
        dvMap[inpName] = f"{fpName}.aero_post.{forceName.lower()}"

# The DVMap only get's defined on the root proc, so let's broadcast it to the rest (not sure if this is necessary)
dvMap = globalComm.bcast(dvMap, root=0)

if ptComm.rank == 0:
    pp(dvMap)

# ==============================================================================
# Potentially set initial DVs from a previous run
# ==============================================================================
if len(args.initDVs) != 0:
    setValsFromFiles(args.initDVs, flightPointProb)

om.n2(flightPointProb, show_browser=False, outfile=os.path.join(localOutputDir, "AeroStruct-N2-Pre-Run.html"))
om.n2(performanceProb, show_browser=False, outfile=os.path.join(outputDir, "Performance-N2-Pre-Run.html"))

# ==============================================================================
# Get design variables, constraints and objectives
# ==============================================================================
designVariables = {}
for dv in flightPointProb.model.get_design_vars().values():
    designVariables[dv["name"]] = dv
for dv in performanceProb.model.get_design_vars().values():
    designVariables[dv["name"]] = dv

# Create some groups of design variables to help us tell pyOptSparse which functions depend on which DVs
structDesignVariables = ["dv_struct"] if args.addStructDVs else []
aeroDesignVariables = [dvName for dvName in designVariables if "_AOA" in dvName]
geoDesignVariables = []
geoInputs = ptComm.bcast(flightPointProb.model.geometry.list_inputs(out_stream=None), root=0)
for dvName in designVariables:
    for geoInput in geoInputs:
        if geoInput[0] in dvName:
            geoDesignVariables.append(dvName)
            break

constraints = {}
for con in flightPointProb.model.get_constraints().values():
    conName = get_prom_name(flightPointProb.model, con["source"])
    constraints[conName] = con

    # Any constraints that are direct outputs of the flight point models and aren't linear should be added to gradFuncs
    if not constraints[conName]["linear"]:
        gradFuncs.append(conName)

for con in performanceProb.model.get_constraints().values():
    constraints[con["name"]] = con

objectives = {}
for obj in flightPointProb.model.get_objectives().values():
    objectives[obj["name"]] = obj

# Again if the objective is a direct output of the flight point models, add it to gradFuncs
for objName in objectives:
    if objName in flightPointProbOutputs:
        gradFuncs.append(objName)

for obj in performanceProb.model.get_objectives().values():
    objectives[obj["name"]] = obj

if ptComm.rank == 0:
    print("\n===============================================================================")
    print("Design variables:")
    for dv in designVariables:
        print(f"  - {dv}")

    print("\nConstraints:")
    for con in constraints:
        print(f"  - {con}")

    print("\nObjectives:")
    for obj in objectives:
        print(f"  - {obj}")
    print("===============================================================================\n")


# Up to now, we'll have caught any gradFuncs from the MPhys models that we need to compute the derivatives of
# because they're directly used as a constraint/objective, now we need to add the functions that we need to compute
# the gradients of because they're inputs to the performance model that computes further objectives/constraints.
for _, fpFuncName in dvMap.items():
    if fpFuncName in flightPointProbOutputs and fpFuncName not in gradFuncs:
        gradFuncs.append(fpFuncName)

# broadcast gradFuncs to all procs in this set
gradFuncs = ptComm.bcast(gradFuncs, root=0)

if ptComm.rank == 0:
    print("\n===============================================================================")
    print("Grad funcs:")
    for func in gradFuncs:
        print(f"  - {func}")
    print("===============================================================================\n")

# --- Disp funcs ---
dispFuncs = gradFuncs.copy()
for output in flightPointProbOutputs:
    # Component-wise masses (for first flight point only)
    if "mass" in output and ptID == 0:
        dispFuncs.append(output)
    # Wing area/volume
    elif "volume" in output.lower() or "area" in output.lower():
        dispFuncs.append(output)

# Remove duplicate entries from dispFuncs
dispFuncs = list(set(dispFuncs))

# broadcast dispFuncs to all procs in this set
dispFuncs = ptComm.bcast(dispFuncs, root=0)

if ptComm.rank == 0:
    print("\n===============================================================================")
    print("Disp funcs:")
    for func in dispFuncs:
        print(f"  - {func}")
    print("===============================================================================\n")


def runAeroStructAnalyses(x=None, evalFuncs=None, writeSolution=False):
    """Run aerostructural analyses for each flight point

    Parameters
    ----------
    x : dict[array], optional
        Design variable values, by default None
    evalFuncs : list[str], optional
        Functions to get the values of, by default None
    writeSolution : bool, optional
        Whether to write out the solution, by default False
    """
    if x is not None:
        for key, val in x.items():
            try:
                flightPointProb.set_val(key, val)
            except KeyError:
                pass
    fail = False
    try:
        flightPointProb.run_model()
    except om.AnalysisError:
        fail = True

    funcs = {"fail": fail}
    if evalFuncs is not None:
        for func in evalFuncs:
            funcs[func] = flightPointProb.get_val(func)

    if writeSolution and not args.noFiles:
        scenario = getattr(flightPointProb.model, localFlightPoint.name)
        scenario.struct_post.write_solution()
        scenario.aero_post.nom_write_solution()

    # Print out some interesting values
    if ptComm.rank == 0:
        for funcType in ["mass", "failure", "lift", "drag"]:
            print("\n==================================================")
            print(f"{funcType.upper()} FUNCTIONS:")
            for func in evalFuncs:
                if funcType in func.lower():
                    print(f"{func} = {funcs[func][0]:e}")
            print("==================================================\n")

    return funcs


def computeSens(x=None, funcs=None, gradFuncs=None, dispFuncs=None, writeSolution=False):
    """Compute sensitivities through the coupled aerostructural analyses

    Parameters
    ----------
    x : dict[array], optional
        Design variable values, by default None
    funcs : dict, optional
        Functions dictionary, not used but required for pyoptsparse function signature, by default None
    gradFuncs : dict[str], optional
        Functions to compute the derivatives of, by default None
    writeSolution : bool, optional
        Whether to write out the solution, by default False
    """
    if x is not None:
        for key, val in x.items():
            try:
                flightPointProb.set_val(key, val)
            except KeyError:
                pass
    if gradFuncs is None:
        gradFuncs = []

    if dispFuncs is None:
        dispFuncs = []

    funcSens = {}
    if len(gradFuncs) != 0:
        openMDAOTotals = flightPointProb.compute_totals(of=gradFuncs, return_format="dict")
        for of, sens in openMDAOTotals.items():
            ofName = get_prom_name(flightPointProb.model, of)
            funcSens[ofName] = {}
            for wrt, val in sens.items():
                wrtName = get_prom_name(flightPointProb.model, wrt)
                funcSens[ofName][wrtName] = val

    # HACK: We need to provide bogus empty derivatives for the functions that are in dispFuncs but not gradFuncs
    # otherwise multipoint will complain
    for func in dispFuncs:
        if func not in funcSens:
            funcSens[func] = {}

    if writeSolution and not args.noFiles:
        scenario = getattr(flightPointProb.model, localFlightPoint.name)
        scenario.struct_post.write_solution()
        scenario.aero_post.nom_write_solution()

    return funcSens


# This is the function that takes the function values from the aerostructural analyses and computes any remaining
# objective/constraints. In our case this involves running the performance model.
def objCon(funcs, printOK, passThroughFuncs):
    # Multiploint computes the derivatives through this objCOn function using complex step, printOK is False when objCon
    # is being complex-stepped
    performanceProb.set_complex_step_mode(not printOK)

    if ptComm.rank == 0 and printOK:
        print("\n==================================================")
        print("OBJCON Functions:")
        pp(funcs)
        print("==================================================\n")

    # Map from flight point outputs to performance model inputs
    for performanceVarName, funcName in dvMap.items():
        performanceProb.set_val(performanceVarName, funcs[funcName])
    performanceProb.run_model()

    outputs = globalComm.bcast(performanceProb.model.list_outputs(return_format="dict", print_arrays=False), root=0)
    for output in outputs.items():
        funcs[output[1]["prom_name"]] = output[1]["val"]

    if ptComm.rank == 0 and printOK:
        print("\n==================================================")
        print("OBJCON Functions:")
        pp(funcs)
        print("==================================================\n")

    return funcs


MP.setObjCon(objCon)


# ==============================================================================
# Create wrapped functions to be used by multipoint sparse
# ==============================================================================
def procSetObj(x=None):
    return runAeroStructAnalyses(x, evalFuncs=dispFuncs, writeSolution=False)


MP.addProcSetObjFunc("all", procSetObj)


# I set writeSolution to true on the sensitivity function so that we only write solution files on major iterations
# (this assumes you're using SNOPT's derivative-free line search)
def procSetSens(x=None, funcs=None):
    return computeSens(x, funcs, gradFuncs=gradFuncs, dispFuncs=dispFuncs, writeSolution=True)


MP.addProcSetSensFunc("all", procSetSens)


def runAnalysesRobustly(dvFiles, evalFuncs=None, writeSolution=False):
    """Sometimes OpenMDAO has trouble converging very flexible designs from scratch. This function will run an analysis,
    and if it fails it will reset the model and try again while gradually ramping the angle of attack until we get back
    to original values.

    Parameters
    ----------
    dvFiles : _type_
        _description_
    writeSolution : bool, optional
        _description_, by default False
    """
    funcs = runAeroStructAnalyses(evalFuncs=evalFuncs, writeSolution=writeSolution)
    if funcs["fail"]:
        # If we got an analysis error, try converging for a lower angle of attack first then run again
        name = localFlightPoint.name
        u_struct = np.copy(flightPointProb.get_val(f"{name}.solver.u_struct"))
        flightPointProb.set_val(f"{name}.solver.u_struct", np.zeros_like(u_struct))
        AOA = np.copy(flightPointProb.get_val(f"{name}_AOA"))

        flightPointProb.model.set_initial_values()
        if len(dvFiles) != 0:
            setValsFromFiles(dvFiles, flightPointProb)

        # HACK: This run will fail but on the next one ADflow will start from nice values
        runAeroStructAnalyses(evalFuncs=evalFuncs, writeSolution=writeSolution)

        for factor in np.linspace(0.1, 1.0, 5):
            flightPointProb.set_val(f"{name}_AOA", AOA * factor)
            funcs = runAeroStructAnalyses(evalFuncs=evalFuncs, writeSolution=writeSolution)
    return funcs


# if args.task != "check":
# ==============================================================================
# Run the model to initialize everything
# ==============================================================================
funcs = runAnalysesRobustly(args.initDVs, evalFuncs=dispFuncs, writeSolution=args.task == "analysis")
# Before proceeding, combine the function values from all flight points on the root proc and broadcast to the rest
gatheredFuncs = globalComm.gather(funcs, root=0)
funcs = {}
if globalRank == 0:
    for func in gatheredFuncs:
        funcs.update(func)
funcs = globalComm.bcast(funcs, root=0)
funcs = objCon(funcs, True, None)
if ptRank == 0:
    pp(funcs)

# If we have DVs that were supposed to be set after initialisation, we can set those now and re-run the model
if len(args.postInitDVs) != 0:
    print(f"Proc {globalRank}: Running again with postInitDVs", flush=True)
    setValsFromFiles(args.postInitDVs, flightPointProb)
    if args.task != "check":
        funcs = runAnalysesRobustly(args.postInitDVs, evalFuncs=dispFuncs, writeSolution=args.task == "analysis")
        gatheredFuncs = globalComm.gather(funcs, root=0)
        funcs = {}
        if globalRank == 0:
            for func in gatheredFuncs:
                funcs.update(func)
        funcs = globalComm.bcast(funcs, root=0)
        funcs = objCon(funcs, True, None)


if args.task in ["check", "opt", "trim"]:
    # ==============================================================================
    # Setup optimization problem
    # ==============================================================================
    optProb = Optimization("Aero-Structural Optimization", MP.obj)

    # ==============================================================================
    # Define design variables
    # ==============================================================================
    for dvName, dv in designVariables.items():
        try:
            value = flightPointProb.get_val(dvName)
        except KeyError:
            value = performanceProb.get_val(dvName)
        scale = 1.0 if dv["scaler"] is None else dv["scaler"]
        optProb.addVarGroup(
            dvName,
            nVars=dv["global_size"],
            value=value,
            lower=dv["lower"] / scale,
            upper=dv["upper"] / scale,
            scale=scale,
        )

    # ==============================================================================
    # Define constraints
    # ==============================================================================
    structConTypes = ["adjcon", "dvcon", "panellengthcon"]

    for conName, con in constraints.items():
        # --- Failure constraints (depend on struct dvs, geometry dvs, and the AoA DV for the relevant flightPoint) ---
        if "ksfailure" in conName.lower():
            wrt = structDesignVariables + geoDesignVariables + aeroDesignVariables
            addConstraintFromOpenMDAO(con, optProb, flightPointProb, wrt=wrt)

        # --- Structural constraints (depend on struct DVs, and maybe geometric DVs) ---
        elif "adjcon" in conName.lower() or "dvcon" in conName.lower():
            wrt = structDesignVariables.copy()
            addConstraintFromOpenMDAO(con, optProb, flightPointProb, wrt=wrt)

        elif "panellengthcon" in conName.lower():
            wrt = structDesignVariables + geoDesignVariables
            addConstraintFromOpenMDAO(con, optProb, flightPointProb, wrt=wrt)

        # --- Geometric constraints (depend only on geometry DVs) ---
        elif "geometry." in conName.lower():
            addConstraintFromOpenMDAO(con, optProb, flightPointProb, wrt=geoDesignVariables)

        # --- Lift constraints (depend on struct dvs, geometry dvs, and the AoA DV for the relevant flightPoint) ---
        elif "liftdiff" in conName.lower():
            wrt = structDesignVariables + geoDesignVariables + aeroDesignVariables
            if localFlightPoint.fuelFraction != 0.0 and "cruise" not in localFlightPoint.name.lower():
                wrt.append("dvs.cruise_AOA")
            addConstraintFromOpenMDAO(con, optProb, performanceProb, wrt=wrt)

        # --- Misc constraints (depend on all dvs) ---
        else:
            wrt = structDesignVariables + geoDesignVariables + aeroDesignVariables
            try:
                flightPointProb.get_val(conName)
                addConstraintFromOpenMDAO(con, optProb, flightPointProb, wrt=wrt)
            except KeyError:
                addConstraintFromOpenMDAO(con, optProb, performanceProb, wrt=wrt)

    # ==============================================================================
    # Define objective
    # ==============================================================================
    # Let's hope we don't have more than one objective defined
    for objName, obj in objectives.items():
        optProb.addObj(objName, scale=obj["scaler"])

    optProb.printSparsity(verticalPrint=True)

    MP.setOptProb(optProb)

    # ==============================================================================
    # Setup optimiser and driver
    # ==============================================================================
    optimiserMap = {
        "paroptsl1": "ParOpt",
        "paroptfilter": "ParOpt",
        "paroptmma": "ParOpt",
        "slsqp": "SLSQP",
        "nlpqlp": "NLPQLP",
        "snopt": "SNOPT",
        "ipopt": "IPOPT",
    }
    optHistFilename = os.path.join(outputDir, "AeroStructOpt.hst")
    optimiserOptions = getOptOptions(
        args.optimiser,
        outputDir,
        args.optIter,
        args.hessianUpdate,
        args.initPenalty,
        args.violLimit,
        args.stepLimit,
        args.feasibility,
        args.optimality,
    )

    restartDict = None
    if args.optimiser == "snopt":
        optimiserOptions["Return work arrays"] = True
        if args.task == "trim":
            optimiserOptions["Problem Type"] = "Feasible point"
            optimiserOptions["Major step limit"] = 10.0
        if args.timeLimit is not None:
            timeLimit = globalComm.bcast(args.timeLimit - (time.time() - startTime), root=0)
            optimiserOptions["Time limit"] = args.timeLimit
        if args.restartDict is not None:
            with open(args.restartDict, "rb") as restartFile:
                restartDict = dill.load(restartFile)
                optimiserOptions["Start"] = "Hot"


    optimiser = OPT(optimiserMap[args.optimiser], options=optimiserOptions)

    # ==============================================================================
    # Run the optimisation
    # ==============================================================================
    if args.task == "trim":
        # Do a basic Newton solve to trim
        maxTrimIter = 20
        alphas = {}
        for fpName in flightPointsDict:
            alphas[f"{fpName}_AOA"] = flightPointsDict[fpName].alpha

        if globalRank == 0:
            print("Trimming:")
            print("=========")
        for ii in range(maxTrimIter):
            funcs, _ = MP.obj(alphas)
            res = []
            if globalRank == 0:
                print(f"Trimming Iteration {ii}")
                print("=========================")
            for fpName in flightPointsDict:
                res.append(funcs[f"{fpName}LiftDiff"])
                if globalRank == 0:
                    print("=" * 80)
                    print(f"{fpName}: AoA = {alphas[f'{fpName}_AOA']}, LiftDiff: {res[-1]}")
                    print("=" * 80)
            if all(np.abs(res) < 1e-1):
                if globalRank == 0:
                    print("=" * 80)
                    print("Trim solve converged!")
                    print("=" * 80)
                break

            sens, _ = MP.sens(alphas, funcs)
            if globalRank == 0:
                print(f"{sens=}")
            for fpName in flightPointsDict:
                update = -funcs[f"{fpName}LiftDiff"] / sens[f"{fpName}LiftDiff"][f"{fpName}_AOA"]
                update = np.clip(update, -5.0, 5.0).flatten()
                alphas[f"{fpName}_AOA"] += update
    elif args.task == "opt":
        if restartDict is not None:
            sol = optimiser(optProb, MP.sens, storeHistory=optHistFilename, restartDict=restartDict)
        else:
            sol = optimiser(optProb, MP.sens, storeHistory=optHistFilename)
        if args.optimiser == "snopt":
            # SNOPT Returns it's working arrays in a restart dictionary that we should save for future hot starts
            restartDict = sol[-1]
            sol = sol[0]
            if globalRank == 0:
                with open(os.path.join(outputDir, "SNOPTRestart.pkl"), "wb") as f:
                    dill.dump(restartDict, f)

# --- Write out the DVs and outputs that aren't too long (e.g not the ADflow state vector) in unscaled form to a pickle file ---
outputs = flightPointProb.model.list_outputs(
    return_format="dict", print_arrays=False, excludes=["*adflow_vol_coords", "*adflow_states"]
)
outputData = {}
for output in outputs:
    try:
        data = flightPointProb.get_val(output)
        if not hasattr(data, "__len__") or len(data) < 10000:
            outputData[output] = data
    except TypeError:
        pass

# Accumulate the data from all flight points on the root proc
gatheredOutputs = globalComm.gather(outputData, root=0)
outputData = {}
if globalRank == 0:
    for output in gatheredOutputs:
        outputData.update(output)

# Add outputs from the performance problem
performanceOutputs = performanceProb.model.list_outputs(return_format="dict", print_arrays=False)

if globalRank == 0:
    outputData.update(performanceOutputs)

if MPI.COMM_WORLD.rank == 0:
    outFileName = os.path.join(outputDir, "Outputs.pkl")
    with open(outFileName, "wb") as f:
        dill.dump(outputData, f, protocol=-1)

om.n2(flightPointProb, show_browser=False, outfile=os.path.join(localOutputDir, "AeroStruct-N2-Post-Run.html"))
om.n2(performanceProb, show_browser=False, outfile=os.path.join(outputDir, "Performance-N2-Post-Run.html"))
