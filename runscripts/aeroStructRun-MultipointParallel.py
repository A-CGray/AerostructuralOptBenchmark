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
from scipy.optimize import lsq_linear
from mpi4py import MPI
import openmdao.api as om
from mphys import Multipoint, MPhysVariables
from mphys.scenarios import ScenarioAeroStructural, ScenarioStructural
from adflow.mphys import ADflowBuilder
from adflow import ADFLOW
from idwarp import USMesh
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
from SETUP.setupWimpress import setupWimpress
from CommonArgs import parser
from OptimiserOptions import getOptOptions
from utils import (
    getOutputDir,
    getStructMeshPath,
    getAeroMeshPath,
    getFFDPath,
    setValsFromFiles,
    saveRunCommand,
    getPromName,
    addConstraintFromOpenMDAO,
    writeOutputs,
    getTipDisplacement,
    getStructDVs,
    setupFuelMassGroup,
    mergeStructDVs,
    getTriangulatedSurface,
    AverageComp,
    getRelevantInputs,
)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import performanceCalc  # noqa: E402
from BFLCalculation.OMGroups import STWTakeoffAnalysisGroup  # noqa: E402
from AircraftSpecs.STWFlightPoints import flightPointSets  # noqa: E402
from AircraftSpecs.FlightPoint import FlightPoint  # noqa: E402
from AircraftSpecs.STWSpecs import aircraftSpecs  # noqa: E402
from geometry.wingGeometry import wingGeometry  # noqa: E402

# --- Get the start time, this is used later for correcting the time limit passed to the optimiser ---
startTime = time.time()

np.set_printoptions(linewidth=800)


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
    choices=[
        "writeJigShape",
        "check",
        "analysis",
        "derivCheck",
        "opt",
        "trim",
        "polar",
        "rawPolar",
    ],
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

# --- Polar options ---
parser.add_argument("--alphaMin", type=float, default=None, help="Angle of attack to start polar at")
parser.add_argument("--alphaMax", type=float, default=None, help="Angle of attack to end polar at")
parser.add_argument("--numAlpha", type=int, default=5, help="Number of points to use in polar")

# --- Coupled solver options ---
parser.add_argument(
    "--noAitken",
    action="store_true",
    help="Don't use Aitken acceleration in the coupled aerostructural solver",
)
parser.add_argument(
    "--aitkenInitFactor",
    type=float,
    default=0.5,
    help="Initial relaxation factor to use in NLBGS solver",
)

# --- Optimisation options ---
parser.add_argument("--rangeScale", type=float, default=1.0, help="Factor to scale the mission range by")
parser.add_argument(
    "--maxWingLoading",
    type=float,
    default=600.0,
    help="Maximum allowable wing loading (kg/m^2)",
)
parser.add_argument(
    "--includeBFL",
    action="store_true",
    help="Include the balanced field length constraint",
    default=False,
)
parser.add_argument(
    "--optType",
    type=str,
    choices=["fuelburn", "structMass", "pareto"],
    default="fuelburn",
    help="Type of optimisation to perform, 'fuelburn' for fuelburn minimisation, 'structMass' for structural mass minimisation with only maneuver flight condition. `pareto` for weighted combination of fuel burn and TOGM, with the weight controlled by the `paretoWeight` input argument",
)
parser.add_argument(
    "--paretoWeight",
    type=float,
    default=1.0,
    help="Weight factor on the fuelburn used in the weighted combination of fuel burn and TOGM used as the objective in the pareto front optimization",
)

# --- Aero options ---
parser.add_argument("--aeroLevel", type=int, default=3, choices=[1, 2, 3])
parser.add_argument(
    "--aeroTol",
    type=float,
    default=None,
    help="Relative tolerance for each NLBGS aero solve",
)
parser.add_argument(
    "--aeroMaxIter",
    type=int,
    default=None,
    help="Iteration limit for each NLBGS aero solve",
)
parser.add_argument("--sepSensorType", type=str, default="new", choices=["old", "new"])

# --- LDTransfer options ---
parser.add_argument(
    "--nMeld",
    type=int,
    default=None,
    help="Override for number of nodes to use in MELD",
)
parser.add_argument(
    "--transferType",
    type=str,
    default=None,
    choices=["linear", "nonlinear"],
    help="Which version (linear or nonlinear) of MELD to use, by default the version that matches the structural formulation will be chosen",
)
parser.add_argument(
    "--transferTo",
    type=str,
    default="skin+ribends",
    choices=["all", "skin", "skin+spar", "skin+ribends"],
    help="Which structural nodes to include in the LDTransfer",
)

args = parser.parse_args()

# If we're doing a derivative check, we should at least enable some geometric and structural DVs
if args.task == "derivCheck":
    args.addGeoDVs = True
    args.sweep = True
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
    args.useFuelMassDVs = True
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

# We can't include anything related to the fuelburn if we don't have a cruise point
hasCruisePoint = any(["cruise" in fp.name.lower() for fp in flightPoints])
if not hasCruisePoint:
    if args.task == "opt" and args.optType == "fuelburn":
        raise ValueError("Cannot run fuelburn minimisation without a cruise flight point")
    if args.includeBFL:
        raise ValueError(
            "Cannot include the balanced field length constraint without a cruise flight point to compute the takeoff gross mass"
        )

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


if args.nMeld is None:
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
else:
    MELD_MESH_FACTOR = args.nMeld


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
isAeroStruct = isinstance(localFlightPoint, FlightPoint)
isCruisePoint = hasCruisePoint and "cruise" in localFlightPoint.name.lower()

# Create output directories
localOutputDir = os.path.join(outputDir, localFlightPoint.name)
localAeroOutputDir = os.path.join(localOutputDir, "aero")
localStructOutputDir = os.path.join(localOutputDir, "struct")
if ptRank == 0:
    os.makedirs(localStructOutputDir, exist_ok=True)
    if isAeroStruct:
        os.makedirs(localAeroOutputDir, exist_ok=True)

# --- Create empty csv files that we will store timing data in ---
funcTimingFile = os.path.join(localOutputDir, f"{localFlightPoint.name}-FuncTiming.csv")
funcSensTimingFile = os.path.join(localOutputDir, f"{localFlightPoint.name}-FuncSensTiming.csv")
if ptRank == 0:
    for fileName in [funcTimingFile, funcSensTimingFile]:
        file = open(fileName, "w")
        file.close()

# Print out and save the full list of command line arguments
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
        "maxIter": 20,
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


structBuilder = TacsBuilder(
    mesh_file=structMeshFile,
    assembler_setup=setup_assembler,
    element_callback=element_callback,
    constraint_setup=constraint_callback,
    problem_setup=setup_tacs_problem,
    coupling_loads=[MPhysVariables.Structures.Loads.AERODYNAMIC] if isAeroStruct else None,
    write_solution=False,
    res_ref=1e3,
)

# ==============================================================================
# ADflow/getIDWarp Setup
# ==============================================================================
aeroOptions = getADflowOptions(aeroMeshFile, localAeroOutputDir, aerostructural=True)
if args.aeroLevel == 3:
    aeroOptions["anksecondordswitchtol"] *= 10
    aeroOptions["rkreset"] = True
if args.noFiles:
    aeroOptions["writeTecplotSurfaceSolution"] = False
    aeroOptions["writevolumesolution"] = False
    aeroOptions["writesurfacesolution"] = False
if args.aeroTol is not None:
    aeroOptions["L2ConvergenceRel"] = args.aeroTol
if args.aeroMaxIter is not None:
    aeroOptions["nCycles"] = args.aeroMaxIter

warpOptions = getIDWarpOptions(aeroMeshFile)
aeroBuilder = ADflowBuilder(
    aeroOptions,
    mesh_options=warpOptions,
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
elif args.transferTo in [
    "skin",
    "skin+ribends",
]:  # The rib-spar intersection nodes will be added later
    ldTransferBodies = [{"aero": ["wall"], "struct": ["SKIN"]}]
elif args.transferTo == "skin+spar":
    ldTransferBodies = [{"aero": ["wall"], "struct": ["SKIN", "SPAR"]}]

isym = SPAN_INDEX  # spanwise-symmetry
if args.transferType is None:
    useLinearizedMELD = not args.nonlinear
elif args.transferType.lower() == "linear":
    useLinearizedMELD = True
elif args.transferType.lower() == "nonlinear":
    useLinearizedMELD = False

ldxferBuilder = MeldBuilder(
    aeroBuilder,
    structBuilder,
    isym=isym,
    n=MELD_MESH_FACTOR,
    linearized=useLinearizedMELD,
    body_tags=ldTransferBodies,
)


# Now we can create the MPhys model for each flight point
class AnalysisPoint(Multipoint):
    def setup(self):
        self.fpName = localFlightPoint.name

        builders = {}
        disciplineVariables = {}

        # ivc to keep the top level DVs
        dvSys = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        # ==============================================================================
        # MPHYS Setup
        # ==============================================================================

        if isAeroStruct:
            # ==============================================================================
            # ADflow setup
            # ==============================================================================
            # --- initialize aero builder ---
            aeroBuilder.initialize(self.comm)
            self.aeroSolver = aeroBuilder.get_solver()
            builders["aero"] = aeroBuilder
            disciplineVariables["aero"] = MPhysVariables.Aerodynamics.Surface

            # Add lift distribution and slice file output
            if not args.noFiles:
                self.aeroSolver.addLiftDistribution(100, INDEX_STRINGS[SPAN_INDEX])
                slicePositions = np.linspace(1e-5, WING_SEMISPAN * 0.99, 51)
                self.aeroSolver.addSlices(INDEX_STRINGS[SPAN_INDEX], slicePositions)

        # ==============================================================================
        # TACS Setup
        # ==============================================================================
        structBuilder.initialize(self.comm)
        self.FEAAssembler = structBuilder.get_fea_assembler()
        disciplineVariables["struct"] = MPhysVariables.Structures
        builders["struct"] = structBuilder

        structDVMap = setupTACS.buildStructDVDictMap(structBuilder.get_fea_assembler(), args)
        if globalRank == 0:
            with open(os.path.join(outputDir, "structDVMap.pkl"), "wb") as structDVMapFile:
                dill.dump(structDVMap, structDVMapFile)

        if isAeroStruct:
            # ==============================================================================
            # MELD setup
            # ==============================================================================
            # Find the nodes at the intersections of the spars and ribs and include them in the LDTransfer
            if args.transferTo == "skin+ribends":
                FEASolver = structBuilder.get_fea_assembler()
                sparComps = FEASolver.selectCompIDs(include="SPAR")
                ribComps = FEASolver.selectCompIDs(include="RIB")
                sparNodes = FEASolver.getGlobalNodeIDsForComps(sparComps, nastranOrdering=True)
                ribNodes = FEASolver.getGlobalNodeIDsForComps(ribComps, nastranOrdering=True)
                sharedNodes = list(set(sparNodes).intersection(set(ribNodes)))
                ldxferBuilder.body_tags[0]["struct"] += sharedNodes
            ldxferBuilder.initialize(self.comm)

        # ==============================================================================
        # Mesh and geometry setup
        # ==============================================================================
        # Setup mesh components for each discipline
        for dName in disciplineVariables:
            self.add_subsystem(f"mesh_{dName}", builders[dName].get_mesh_coordinate_subsystem())

        geometryComp = OM_DVGEOCOMP(file=ffdFile, type="ffd", options={"isComplex": isComplex})
        self.add_subsystem("geometry", geometryComp)

        # Connect each discipline's mesh coordinates to the geometry component
        for dName, discipline in disciplineVariables.items():
            # Tell the geometry component that there will be a set of coordinates for the discipline
            geometryComp.nom_add_discipline_coords(discipline.Geometry)

            # Connect the original mesh coordinates as an input to the geometry component
            self.connect(
                f"mesh_{dName}.{discipline.Mesh.COORDINATES}",
                f"geometry.{discipline.Geometry.COORDINATES_INPUT}",
            )

        if isAeroStruct:
            # ==============================================================================
            # WimpressCalc Setup
            # ==============================================================================
            wimpressCalc, wimpressCalcComp, wimpressCoordsComp = setupWimpress()
            if globalRank == 0:
                wimpressCalc.writeTecplot(os.path.join(outputDir, "wimpressCalcInit.dat"))
            # Because the WimpressCalc component isn't following the MPhys convention we can just add the wimpress
            # coordinates directly as a pointset. The geometry component will then output the deformed coordinates without
            # needing a corresponding input. Unfortunately we can't do this in setup because the DVGeo instances associated
            # with the geometry component don't exist yet (they're created in its setup method), so we'll add the pointset
            # and connect things in configure instead.
            self.add_subsystem("PlanformValues", wimpressCalcComp)

        # ==============================================================================
        # Add TACS dvs
        # ==============================================================================
        initStructDVs, fuelMassDVInds, sizingDVInds = getStructDVs(structBuilder)
        self.numRibBays = len(fuelMassDVInds)
        dvSys.add_output("dv_struct", val=initStructDVs[sizingDVInds])

        if args.addStructDVs:
            lb, ub = structBuilder.get_dv_bounds()
            structDVScaling = np.array(structBuilder.fea_assembler.scaleList)
            self.add_design_var(
                "dv_struct",
                lower=lb[sizingDVInds],
                upper=ub[sizingDVInds],
                scaler=structDVScaling[sizingDVInds] * args.structScalingFactor,
            )

        # ==============================================================================
        # Fuel mass computation
        # ==============================================================================
        fuelDVName = f"{self.fpName}-fuelMass"
        dvSys.add_output(
            fuelDVName, val=10500.0, shape=1
        )  # Total fuel mass value, placeholder for now TODO: How should this be set if it's not a DV?
        if args.useFuelMassDVs:
            self.add_design_var(
                fuelDVName,
                lower=0.0,
                scaler=1e-3,
            )

        setupFuelMassGroup(self, fuelDVName, numRibBays=self.numRibBays)

        # Now add component that merges the fuel mass DVs and the structural DVs back into a full set of TACS DVs
        mergeStructDVs(self, fuelMassDVInds, sizingDVInds)

        # ==============================================================================
        # Create the scenario
        # ==============================================================================
        if isAeroStruct:
            self.mphys_add_scenario(
                self.fpName,
                ScenarioAeroStructural(
                    aero_builder=aeroBuilder,
                    struct_builder=structBuilder,
                    ldxfer_builder=ldxferBuilder,
                ),
            )
        else:
            self.mphys_add_scenario(self.fpName, ScenarioStructural(struct_builder=structBuilder))

        # Connect geometry to discipline coordinates in the aerostructural scenario
        for dName, discipline in disciplineVariables.items():
            src = f"geometry.{discipline.Geometry.COORDINATES_OUTPUT}"
            if dName == "aero":
                target = f"{self.fpName}.{discipline.COORDINATES_INITIAL}"
            else:
                target = f"{self.fpName}.{discipline.COORDINATES}"
            self.connect(src, target)

        self.connect("dv_struct_full", f"{self.fpName}.dv_struct")

        if isAeroStruct:
            # ==============================================================================
            # Setup performance calculations
            # ==============================================================================
            # Now we need to add the OpenMDAO groups for computing:
            # - Buffet constraints
            # - Landing gross mass (One point only)
            # - Fuel burn (Cruise point only)
            # - Take-off gross mass (Cruise point only)
            # - Balanced field length (Cruise point only)
            # - Wing loading (Cruise point only)
            if "buffet" in self.fpName.lower():
                buffetConName = f"{self.fpName}BuffetCon"
                buffetConstraintComp = om.AddSubtractComp(
                    output_name=buffetConName,
                    input_names=["SepArea", "wingArea"],
                    scaling_factors=[1, -0.04],
                )
                self.add_subsystem(
                    buffetConName,
                    buffetConstraintComp,
                    promotes_outputs=["*"],
                )
                self.connect("PlanformValues.wimpressArea", f"{buffetConName}.wingArea")
                sepSensorFuncName = "sepsensorksarea" if args.sepSensorType == "new" else "sepsensor"
                sepSensorFuncName = f"{self.fpName}.aero_post.{sepSensorFuncName}"
                self.connect(sepSensorFuncName, f"{buffetConName}.SepArea")

            if ptID == 0:
                massComp = performanceCalc.AirframeMassGroup(
                    aircraftSpecs=aircraftSpecs,
                )
                self.add_subsystem(
                    "airframeMass",
                    massComp,
                    promotes=[
                        "landingGrossMass",
                        ("wingboxMass", f"{self.fpName}.mass"),
                    ],
                )
            if isCruisePoint:
                fuelBurnGroup = performanceCalc.FuelBurnGroup(
                    aircraftSpecs=aircraftSpecs,
                    flightPoint=localFlightPoint,
                )
                self.add_subsystem(
                    "fuelBurn",
                    fuelBurnGroup,
                    promotes_inputs=["landingGrossMass"],
                    promotes_outputs=[
                        "totalFuelBurn",
                        "cruiseStartMass",
                        "takeoffMass",
                    ],
                )
                for force in [
                    "lift",
                    "drag",
                ]:
                    self.connect(
                        f"{self.fpName}.aero_post.{force.lower()}",
                        f"fuelBurn.cruise{force.capitalize()}",
                    )

                # --- Compute the mid-cruise mass ---
                cruiseMass = performanceCalc.MidSegmentMassComp()
                self.add_subsystem(
                    "midCruiseMass",
                    cruiseMass,
                    promotes_outputs=[("midSegmentMass", "midCruiseMass")],
                )
                self.connect("landingGrossMass", "midCruiseMass.finalMass")
                self.connect("cruiseStartMass", "midCruiseMass.initialMass")

                # --- Compute the wing loading ---
                wingLoadingComp = performanceCalc.WingLoadingComp()
                self.add_subsystem("wingLoading", wingLoadingComp, promotes_outputs=["*"])
                self.connect("takeoffMass", "wingLoading.MTOM")
                self.connect("PlanformValues.wimpressArea", "wingLoading.wingArea")

                # --- Compute the balanced field length ---
                if args.includeBFL:
                    # OpenConcept expects the wing area for the full aircraft, so we need to double it
                    self.add_subsystem(
                        "doubleWingArea",
                        om.ExecComp("doubleWingArea = 2 * wingArea", units="m**2"),
                        promotes_outputs=["*"],
                    )
                    self.connect("PlanformValues.wimpressArea", "doubleWingArea.wingArea")

                    # OpenConcept wants a single t/c value for the whole wing, so we'll take an average of the values computed by pyGeo
                    self.add_subsystem(
                        "avgToC",
                        AverageComp(),
                    )
                    self.connect("geometry.SectionToC", "avgToC.in")

                    takeoffGroup = STWTakeoffAnalysisGroup(
                        ivc_excludes=[
                            "ac|weights|MTOW",
                            "ac|geom|wing|S_ref",
                            "ac|geom|wing|AR",
                            "ac|geom|wing|c4sweep",
                            "ac|geom|wing|taper",
                            "ac|geom|wing|toverc",
                        ]
                    )
                    self.add_subsystem("takeoff", takeoffGroup)
                    self.connect("takeoffMass", "takeoff.ac|weights|MTOW")
                    self.connect("doubleWingArea", "takeoff.ac|geom|wing|S_ref")
                    self.connect("PlanformValues.aspectRatio", "takeoff.ac|geom|wing|AR")
                    self.connect("PlanformValues.QCSweep", "takeoff.ac|geom|wing|c4sweep")
                    self.connect("PlanformValues.trapTaper", "takeoff.ac|geom|wing|taper")
                    self.connect("avgToC.out", "takeoff.ac|geom|wing|toverc")

    def configure(self):
        geometryComp = self.geometry
        dvComp = self.dvs

        if isAeroStruct:
            # Give ADflow the aero problem for this procset's flight point and add the angle of attack as a DV
            fp = localFlightPoint
            scenario = getattr(self, self.fpName)
            fp.addDV("alpha", value=fp.alpha, name="aoa", units="deg")
            scenario.coupling.aero.mphys_set_ap(fp)
            scenario.aero_post.mphys_set_ap(fp)
            alphaDVName = f"{self.fpName}_AOA"
            dvComp.add_output(alphaDVName, val=fp.alpha, units="deg")
            self.add_design_var(alphaDVName, lower=-20.0, upper=20.0, scaler=1.0)
            self.connect(
                alphaDVName,
                [f"{self.fpName}.coupling.aero.aoa", f"{self.fpName}.aero_post.aoa"],
            )

        # Setup the geometric DVs and constraints, we only need to compute the geometric constraints (LE radius,
        # thickness, area etc) on one proc set
        geometryComp.nom_setConstraintSurface(getTriangulatedSurface())
        setupDVGeo(
            args,
            self,
            geometryComp,
            dvComp,
            dvScaleFactor=args.geoScalingFactor,
            geoCompName="geometry",
            addGeoDVs=args.addGeoDVs,
            addGeoConstraints=ptID == 0,
            computeRibBayVolumes=True,
            computeToC=isCruisePoint,
        )

        DVCon = self.geometry.nom_getDVCon()
        if globalRank == 0:
            DVCon.writeTecplot(os.path.join(outputDir, "DVConstraints.dat"))
            DVCon.writeSurfaceTecplot(os.path.join(outputDir, "DVConstraintsSurface.dat"))
            DVCon.writeSurfaceSTL(os.path.join(outputDir, "DVConstraintsSurface.stl"))

        # Connect rib bay volumes to the fuel distribution group
        for ii in range(self.numRibBays):
            self.connect(f"geometry.RibBay-Volume_{ii}", f"RibBay-Volume_{ii}")

        if isAeroStruct:
            # Add the wimpress coordinates as a pointset to the geometry component and connect them to the wimpress comp
            wimpressComp = self.PlanformValues
            wimpressCoordName = "x_wimpress"
            geometryComp.nom_addPointSet(
                wimpressComp.wimpressCalc.getCoords(packed=True),
                ptName=wimpressCoordName,
                distributed=False,
            )
            self.connect(f"geometry.{wimpressCoordName}", "PlanformValues.x_wimpress")

        # Only add TACS constraints on the first proc set
        if ptID == 0 and args.addStructDVs:
            firstScenario = getattr(self, self.fpName)
            add_tacs_constraints(firstScenario)

        if isAeroStruct:
            # ==============================================================================
            # Play with the coupled solver settings
            # ==============================================================================
            scenario = getattr(self, self.fpName)
            scenario.coupling.nonlinear_solver = om.NonlinearBlockGS(
                maxiter=100,
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
                atol=1e-4 * args.tolFactor,
                rtol=1e-8 * args.tolFactor,
                maxiter=50,
                iprint=2,
            )
            scenario.coupling.linear_solver.precon = om.LinearBlockGS(maxiter=1, iprint=-2, use_aitken=False, rtol=1e-1)

        # ==============================================================================
        # Setup dummy aero solver
        # ==============================================================================
        # In order to write out the jig shape of the OML and its twist distribution, I need to setup a dummy ADflow
        # instance that will recieve surface coordinates directly from the DVGeo without structural displacements. I
        # will then write solution files from this solver without every actually running it
        self.dummyAeroSolver = None
        if isAeroStruct:
            if not args.noFiles:
                if ptID == 0:
                    self.dummyAeroSolver = ADFLOW(options=aeroOptions, comm=self.comm)
                    self.dummyAeroSolver.setAeroProblem(localFlightPoint)
                    self.dummyAeroSolver.setDVGeo(geometryComp.nom_getDVGeo())
                    mesh = USMesh(options=self.aeroSolver.mesh.options, comm=self.comm)
                    self.dummyAeroSolver.setMesh(mesh)
                    self.dummyAeroSolver.addLiftDistribution(100, INDEX_STRINGS[SPAN_INDEX])
                    slicePositions = np.linspace(1e-5, WING_SEMISPAN * 0.99, 51)
                    self.dummyAeroSolver.addSlices(INDEX_STRINGS[SPAN_INDEX], slicePositions)
                    # In order to get solution files that don't contain NaNs that break tecplot, we need to actually run the
                    # solver, so we can just set the iteration limit to 0 so that the solver just does it's initialisation
                    # steps and doesn't actually waste any time solving.
                    self.dummyAeroSolver.setOption("nCycles", 0)


# --- Now actually create the OpenMDAO model for each point ---
flightPointProb = om.Problem(reports=None, comm=ptComm)
flightPointProb.model = AnalysisPoint()

# --- Finally create the aircraft performance OpenMDAO model ---
performanceProb = om.Problem(reports=None, comm=globalComm)
includeFuelConstraints = args.useFuelMassDVs and hasCruisePoint
performanceProb.model = performanceCalc.FuelAndMassConstraintGroup(
    aircraftSpecs=aircraftSpecs, flightPointSet=flightPoints, includeFuelConstraints=includeFuelConstraints
)

if args.task in ["trim", "opt", "check"]:
    # ==============================================================================
    # Setup objective
    # ==============================================================================
    if args.task == "trim" or structOnlyOpt:
        if ptID == 0:
            # For the trim task we setup a dummy objective that doesn't depend on the trim variables so that the optimiser
            # just satisfies the trim constraints
            performanceProb.model.add_objective("airframeMass.wingMass", scaler=1e-3, cache_linear_solution=True)
    elif args.optType == "fuelburn":
        if isCruisePoint:
            flightPointProb.model.add_objective("totalFuelBurn", scaler=1e-4, cache_linear_solution=True)

    # ==============================================================================
    # Setup constraints
    # ==============================================================================

    # --- Lift constraints ---
    # We have a lift=mass*g*loadFactor constraint for each aerostructural flight point. These should be enforced in both
    # the trim and opt tasks.

    # Generally, massively scaling down the lift difference helps the optimiser make better progress, but in the trim
    # task, where all we care about is hitting the lift constraints, we won't scale them down quite as much so that we
    # really nail the right lift value.
    liftConScale = 1e-6 if args.task == "trim" else 1e-8
    for fp in flightPoints:
        if isinstance(fp, FlightPoint):
            performanceProb.model.add_constraint(
                f"{fp.name}LiftDiff",
                equals=0.0,
                scaler=liftConScale,
                cache_linear_solution=True,
            )

    # --- Fuel mass consistency constraints ---
    # These constraints ensure that the design variable that controls the magnitude of the fuel mass loads is consistent
    # with this flight point's correct fuel mass, which depends on the computed mission fuel burn and the mass
    # configuration at this flight point. We include these in both opt and trim tasks if the we are using the fuel mass
    # DVs and there is a cruise point to compute the fuel burn.
    if includeFuelConstraints:
        for fp in flightPoints:
            performanceProb.model.add_constraint(
                f"{fp.name}FuelMassDiff",
                equals=0.0,
                scaler=liftConScale,
                cache_linear_solution=True,
            )

    if args.task in ["opt", "check"]:
        # --- TACS Failure constraints ---
        # These should be included in all opt tasks
        if localFlightPoint.failureGroups is not None:
            for group in localFlightPoint.failureGroups:
                failureConName = f"{localFlightPoint.name}.{group}_ksFailure"
                flightPointProb.model.add_constraint(failureConName, upper=1.0, scaler=1.0, cache_linear_solution=True)

        if not structOnlyOpt and args.addGeoDVs:
            # --- Fuel tank capacity constraint ---
            # We only apply this if we have a cruise point to compute the fuel burn and if we have geometric DVs that
            # will affect the fuel tank volume
            if includeFuelConstraints and (args.span or args.taper or args.shape):
                performanceProb.model.add_constraint("fuelTankUsage", upper=1.0, cache_linear_solution=True)
            # --- Wing loading constraint ---
            if ptID == 0 and (args.span or args.taper):
                # This constraint should only be applied if the optimiser has control over the wing planform
                flightPointProb.model.add_constraint(
                    "wingLoading",
                    upper=args.maxWingLoading,
                    scaler=1.0 / args.maxWingLoading,
                    cache_linear_solution=True,
                )
        # --- Buffet constraints ---
        if "buffet" in localFlightPoint.name.lower():
            flightPointProb.model.add_constraint(
                f"{localFlightPoint.name}BuffetCon",
                upper=0.0,
                scaler=1 / (0.04 * wingGeometry["wing"]["planformArea"]),
                cache_linear_solution=True,
            )
        # --- Balanced field length constraint ---
        # This should only be applied if the flight point is a cruise point and the balanced field
        if isCruisePoint and args.includeBFL:
            flightPointProb.model.add_constraint(
                "takeoff.rotate.range_final",
                upper=aircraftSpecs["maxBFL"],
                scaler=1.0 / aircraftSpecs["maxBFL"],
                cache_linear_solution=True,
            )


# ==============================================================================
# Setup the OpenMDAO models
# ==============================================================================
# Setup the model for this proc's flight point and get the names of its outputs
flightPointProb.setup(force_alloc_complex=isComplex, mode="rev")
# Need to call final_setup so that sizes of inputs and outputs are figured out correctly, otherwise calling
# `list_outputs` can fail (see https://github.com/OpenMDAO/OpenMDAO/issues/3560 for details)
flightPointProb.final_setup()
tmp = ptComm.bcast(flightPointProb.model.list_outputs(out_stream=None), root=0)
flightPointProbOutputs = {}
for output in tmp:
    flightPointProbOutputs[output[1]["prom_name"]] = output[1]

# Setup the performance model and get the names of its inputs
performanceProb.setup(force_alloc_complex=True)
performanceProbInputs = []
for _, inp in performanceProb.model.list_inputs(out_stream=None):
    promName = inp["prom_name"]
    if promName not in performanceProbInputs and "." not in promName:
        performanceProbInputs.append(promName)
performanceProbInputs = globalComm.bcast(list(set(performanceProbInputs)), root=0)
# For OpenMDAO's relevance checking to work, which we need to automatically figure out which design variables the
# outputs of the performance model depend on, we need to declare the inputs to the performance model that come from the
# flight point models as design variables in the performance model. For some godforsaken reason, the only way I can find
# to successfully add design variables after setup has been called (which I need so that I can call `list_inputs`) is to
# temporarily toggle the static_mode flag.
performanceProb.model._problem_meta["static_mode"] = not performanceProb.model._problem_meta["static_mode"]
for inpName in performanceProbInputs:
    performanceProb.model.add_design_var(inpName)
performanceProb.model._problem_meta["static_mode"] = not performanceProb.model._problem_meta["static_mode"]
performanceProb.final_setup()

# Define the map from the outputs of the flight point models to the inputs required for the performance calculation
# model
perf2FlightPointMap = {}
for inpName in performanceProbInputs:
    # These have the same name in both models
    if (
        inpName
        in [
            "wingboxVolume",
            "takeoffMass",
            "midCruiseMass",
            "cruiseStartMass",
            "landingGrossMass",
        ]
        or "fuelMass" in inpName
    ):
        perf2FlightPointMap[inpName] = inpName
    elif inpName == "fuelBurn":
        perf2FlightPointMap[inpName] = "totalFuelBurn"
    elif "Drag" in inpName or "Lift" in inpName:
        fpName = inpName[:-4]
        forceName = inpName[-4:]
        perf2FlightPointMap[inpName] = f"{fpName}.aero_post.{forceName.lower()}"

# The map only get's defined on the root proc, so let's broadcast it to the rest (not sure if this is necessary)
perf2FlightPointMap = globalComm.bcast(perf2FlightPointMap, root=0)

if ptComm.rank == 0:
    pp(perf2FlightPointMap)

# ==============================================================================
# Potentially set initial DVs from a previous run
# ==============================================================================
if len(args.initDVs) != 0:
    setValsFromFiles(args.initDVs, flightPointProb)

om.n2(
    flightPointProb,
    show_browser=False,
    outfile=os.path.join(localOutputDir, "AeroStruct-N2-Pre-Run.html"),
)
om.n2(
    performanceProb,
    show_browser=False,
    outfile=os.path.join(outputDir, "Performance-N2-Pre-Run.html"),
)


# ==============================================================================
# Define grad and disp funcs
# ==============================================================================
# Grad funcs are the functions in the flight point models that we need to compute derivatives of because they are
# objectives/constraints, or because they're used in the computation of objectives/constraints. Disp funcs are functions
# that are not required for the computation of objectives/constraints but we want to track and store in the history file
# anyway
gradFuncs = []

# --- Get design variables, constraints and objectives from OpenMDAO models ---
designVariables = {}
for dv in flightPointProb.model.get_design_vars().values():
    designVariables[dv["name"]] = dv

# Any constraints or objectives that are direct outputs of the flight point models and aren't linear should be added to
# gradFuncs
for con in flightPointProb.model.get_constraints().values():
    if not con["linear"]:
        conName = getPromName(flightPointProb.model, con["source"])
        gradFuncs.append(conName)

for obj in flightPointProb.model.get_objectives().values():
    gradFuncs.append(obj["name"])

# Up to now, we'll have caught any gradFuncs from the MPhys models that we need to compute the derivatives of
# because they're directly used as a constraint/objective, now we need to add the functions that we need to compute
# the gradients of because they're inputs to the performance model that computes further objectives/constraints.
for _, fpFuncName in perf2FlightPointMap.items():
    if fpFuncName in flightPointProbOutputs and fpFuncName not in gradFuncs:
        gradFuncs.append(fpFuncName)

# If we're doing a trim solve and not an optimization then we can remove the ksFailure and buffet constraints from the gradFuncs to avoid computing their adjoints
# TODO: Change how constraints are added to the OpenMDAO model so that I don't have to do this
if args.task == "trim":
    gradFuncs[:] = [x for x in gradFuncs if "ksfailure" not in x.lower()]
    gradFuncs[:] = [x for x in gradFuncs if "sepsensor" not in x.lower()]

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


# ==============================================================================
# Define functions to be used in the optimisation/analysis
# ==============================================================================
def writeAeroStructSolution():
    scenario = getattr(flightPointProb.model, localFlightPoint.name)
    scenario.struct_post.write_solution()
    if isAeroStruct:
        scenario.aero_post.nom_write_solution()
        if ptID == 0:
            dummyAeroSolver = flightPointProb.model.dummyAeroSolver
            dummyAeroSolver.setAeroProblem(localFlightPoint)
            dummyAeroSolver(localFlightPoint, writeSolution=False)
            dummyAeroSolver.writeSolution(baseName="jigshape", number=(scenario.aero_post.solution_counter - 1))


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
    funcStartTime = time.time()
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

    funcRunTime = time.time() - funcStartTime
    if ptRank == 0:
        with open(funcTimingFile, "a") as f:
            f.write(f"{funcRunTime:.16e}\n")

    if writeSolution and not args.noFiles:
        writeAeroStructSolution()

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
    funcStartTime = time.time()
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
            ofName = getPromName(flightPointProb.model, of)
            funcSens[ofName] = {}
            for wrt, val in sens.items():
                wrtName = getPromName(flightPointProb.model, wrt)
                funcSens[ofName][wrtName] = val

    funcRunTime = time.time() - funcStartTime
    if ptRank == 0:
        with open(funcSensTimingFile, "a") as f:
            f.write(f"{funcRunTime:.16e}\n")

    # HACK: We need to provide bogus empty derivatives for the functions that are in dispFuncs but not gradFuncs
    # otherwise multipoint will complain
    for func in dispFuncs:
        if func not in funcSens:
            funcSens[func] = {}

    if writeSolution and not args.noFiles:
        writeAeroStructSolution()

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
    for performanceVarName, funcName in perf2FlightPointMap.items():
        performanceProb.set_val(performanceVarName, funcs[funcName])
    performanceProb.run_model()

    outputs = globalComm.bcast(
        performanceProb.model.list_outputs(return_format="dict", print_arrays=False),
        root=0,
    )
    for output in outputs.items():
        funcs[output[1]["prom_name"]] = output[1]["val"]

    # Compute pareto font objective, weighted combination of fuel burn and TOGM
    if args.optType == "pareto":
        funcs["paretoObj"] = (
            args.paretoWeight * funcs["totalFuelBurn"] / 1e4
            + (1 - args.paretoWeight) * funcs["takeoffMass"] / aircraftSpecs["refMTOW"]
        )

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
    return runAeroStructAnalyses(x, evalFuncs=dispFuncs, writeSolution=args.task != "opt")


MP.addProcSetObjFunc("all", procSetObj)


# I set writeSolution to true on the sensitivity function so that we only write solution files on major iterations
# (this assumes you're using SNOPT's derivative-free line search)
def procSetSens(x=None, funcs=None):
    return computeSens(
        x,
        funcs,
        gradFuncs=gradFuncs,
        dispFuncs=dispFuncs,
        writeSolution=args.task == "opt",
    )


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


if args.task == "writeJigShape":
    x = {}
    for dvName in designVariables:
        try:
            value = flightPointProb.get_val(dvName)
        except KeyError:
            value = performanceProb.get_val(dvName)
        x[dvName] = value
    scenario = getattr(flightPointProb.model, localFlightPoint.name)
    if ptID == 0:
        dummyAeroSolver = flightPointProb.model.dummyAeroSolver
        dummyAeroSolver.DVGeo.setDesignVars(x)
    writeAeroStructSolution()
    exit(0)

if args.task != "check":
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
        funcs = runAnalysesRobustly(
            args.postInitDVs,
            evalFuncs=dispFuncs,
            writeSolution=args.task == "analysis",
        )
        gatheredFuncs = globalComm.gather(funcs, root=0)
        funcs = {}
        if globalRank == 0:
            for func in gatheredFuncs:
                funcs.update(func)
        funcs = globalComm.bcast(funcs, root=0)
        funcs = objCon(funcs, True, None)

if args.task == "derivCheck":
    # Define some groups of design variables
    structDesignVariables = ["dv_struct"] if args.addStructDVs else []
    aeroDesignVariables = [dvName for dvName in designVariables if "_AOA" in dvName]
    geoDesignVariables = []
    geoInputs = ptComm.bcast(flightPointProb.model.geometry.list_inputs(out_stream=None), root=0)
    for dvName in designVariables:
        for geoInput in geoInputs:
            if geoInput[0] in dvName:
                geoDesignVariables.append(dvName)
                break
    np.set_printoptions(precision=16, linewidth=200)
    wrt = geoDesignVariables + aeroDesignVariables  # + ["dv_struct"]
    fpName = localFlightPoint.name
    of = [
        f"{fpName}.aero_post.cl",
        f"{fpName}.aero_post.cd",
        f"{fpName}.compliance",
        f"{fpName}.l_skin_ksFailure",
    ]
    of = [f for f in of if f in flightPointProbOutputs]
    origDVs = {}
    for variable in wrt:
        origDVs[variable] = flightPointProb.get_val(variable)
    with open(os.path.join(localOutputDir, f"{fpName}-derivCheck-{ptRank:03d}.pkl"), "wb") as pickleFile:
        with open(
            os.path.join(localOutputDir, f"{fpName}-derivCheck-{ptRank:03d}.txt"),
            "w",
        ) as textFile:
            if ptComm.rank == 0:
                print(f"Testing derivatives of {of}, with respect to {wrt}")
            totalsCheckData = flightPointProb.check_totals(
                of=of,
                wrt=wrt,
                method="cs" if isComplex else "fd",
                form="central",
                step=1e-200 if isComplex else 1e-3,
                step_calc="abs",
                out_stream=textFile,
                compact_print=True,
                rel_err_tol=1e-8 if isComplex else 1e-2,
                abs_err_tol=1e6,
            )
            for variable in wrt:
                flightPointProb.set_val(variable, origDVs[variable])
            flightPointProb.run_model()
            if ptComm.rank == 0:
                print(f"Testing derivatives of {of}, with respect to dv_struct")
            totalsCheckData.update(
                flightPointProb.check_totals(
                    of=of,
                    wrt=["dv_struct"],
                    method="cs" if isComplex else "fd",
                    form="central",
                    step=1e-200 if isComplex else 1e-5,
                    step_calc="rel",
                    out_stream=textFile,
                    compact_print=True,
                    rel_err_tol=1e-8 if isComplex else 1e-2,
                    abs_err_tol=1e6,
                    directional=True,
                )
            )
        dill.dump(totalsCheckData, pickleFile, protocol=-1)

if args.task == "polar":
    alphaPert = 1.0
    machPert = 0.02
    numPoints = 9
    alphas = localFlightPoint.alpha + np.linspace(-alphaPert, alphaPert, numPoints)
    machs = localFlightPoint.mach + np.linspace(-machPert, machPert, numPoints)
    for alphaIndex, alpha in enumerate(alphas):
        for machIndex, mach in enumerate(machs):
            localFlightPoint.mach = mach
            # We have to set alpha through the dvs otherwise it will be overwritten by the default DV value
            x = {f"dvs.{localFlightPoint.name}_AOA": alpha}
            funcs = runAeroStructAnalyses(x=x, evalFuncs=dispFuncs, writeSolution=True)
            writeOutputs(
                flightPointProb,
                outputDir=localOutputDir,
                fileName=f"Mach-{machIndex}-Alpha-{alphaIndex}-Outputs",
            )
if args.task == "rawPolar":
    alphaMin = localFlightPoint.alpha - 1 if args.alphaMin is None else args.alphaMin
    alphaMax = localFlightPoint.alpha + 1 if args.alphaMax is None else args.alphaMax
    alphas = np.linspace(args.alphaMin, args.alphaMax, args.numAlpha)
    for alphaIndex, alpha in enumerate(alphas):
        # We have to set alpha through the dvs otherwise it will be overwritten by the default DV value
        x = {f"dvs.{localFlightPoint.name}_AOA": alpha}
        funcs = runAeroStructAnalyses(x=x, evalFuncs=dispFuncs, writeSolution=True)
        writeOutputs(
            flightPointProb,
            outputDir=localOutputDir,
            fileName=f"Alpha-{alphaIndex}-Outputs",
        )

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
            dv["value"] = flightPointProb.get_val(dvName)
        except KeyError:
            dv["value"] = performanceProb.get_val(dvName)
        if dv["scaler"] is None:
            dv["scaler"] = 1.0
        optProb.addVarGroup(
            dvName,
            nVars=dv["global_size"],
            value=dv["value"],
            lower=dv["lower"] / dv["scaler"],
            upper=dv["upper"] / dv["scaler"],
            scale=dv["scaler"],
        )

    # ==============================================================================
    # Define constraints
    # ==============================================================================
    # Add all constraints from the flight point model, addConstraintFromOpenMDAO will automatically figure out which DVs
    # each constraint depends on using the OpenMDAO model
    for con in flightPointProb.model.get_constraints().values():
        addConstraintFromOpenMDAO(con, optProb, flightPointProb, wrt="auto")

    # For the constraints coming from the performance model, things are more complicated because the design variables
    # are not in the performance model. We therefore need to do the following for each constraint in the performance
    # model:
    # 1) Figure out which inputs to the performance model it depends on
    # 2) Map those inputs back to outputs of the flight point models using perf2FlightPointMap
    # 3) Figure out which design variables those flight point outputs depend on using the flight point OpenMDAO model
    # This is complicated by the fact that some constraints in the performance model might depend on outputs from
    # multiple flight points
    for con in performanceProb.model.get_constraints().values():
        # Step 1:
        fullConName = con["source"]
        promConName = getPromName(performanceProb.model, fullConName)
        relevantInputs = getRelevantInputs(performanceProb, fullConName, dvOnly=True)
        relevantInputs = [getPromName(performanceProb.model, inp) for inp in relevantInputs]
        if ptRank == 0:
            print(f"\n\nPerformance constraint {promConName} depends on performance inputs:")
            for inp in relevantInputs:
                print(f"- {inp}", flush=True)
        # Step 2:
        relevantFlightPointOutputs = []
        for inpName in relevantInputs:
            if inpName in perf2FlightPointMap:
                relevantFlightPointOutputs.append(perf2FlightPointMap[inpName])
        relevantFlightPointOutputs = list(set(relevantFlightPointOutputs))
        if ptRank == 0:
            print("and thus on flight point outputs:")
            for output in relevantFlightPointOutputs:
                print(f"- {output}", flush=True)
        # Step 3:
        # Every proc has the full list of the outputs that are needed, so we can work through that and, if any of the
        # outputs are from this proc's flight point, we can use the flight point OpenMDAO model to figure out which DVs
        # they depend on

        # TODO: This doesn't work because OpenMDAO only seems to be able to compute relevance for outputs that are constraints or objectives, see if there's a way around this
        # relevantDVs = []
        # for output in relevantFlightPointOutputs:
        #     if output in flightPointProbOutputs:
        #         relevantDVs += getRelevantInputs(flightPointProb, getAbsName(flightPointProb.model, output), dvOnly=True)
        # relevantDVs = [getPromName(flightPointProb.model, inp) for inp in relevantDVs]
        # relevantDVs = list(set(relevantDVs))
        # # Gather the relevant DVs from all procs, combine them into a single list, then send back to all procs
        # allRelevantDVs = globalComm.gather(relevantDVs, root=0)
        # if globalRank == 0:
        #     relevantDVs = []
        #     for rdvs in allRelevantDVs:
        #         relevantDVs += rdvs
        #     relevantDVs = list(set(relevantDVs))
        # relevantDVs = globalComm.bcast(relevantDVs, root=0)
        # if ptRank == 0:
        #     print("and thus on design variables:")
        #     for dv in relevantDVs:
        #         print(f"- {dv}", flush=True)

        addConstraintFromOpenMDAO(con, optProb, performanceProb)  # , wrt=relevantDVs)

    # ==============================================================================
    # Define objective
    # ==============================================================================
    # Let's hope we don't have more than one objective defined
    if args.optType == "pareto":
        # In this case the objective doesn't come directly from either OpenMDAO model, so I just create a spoof
        # objective dict that matched what get_objectives returns
        objectives = {"paretoObj": {"name": "paretoObj", "scaler": 1.0}}
    else:
        objectives = flightPointProb.model.get_objectives()
        objectives.update(performanceProb.model.get_objectives())

    for obj in objectives.values():
        optProb.addObj(obj["name"], scale=obj["scaler"])

    # Print out some useful info about the optimization problem
    if ptComm.rank == 0:
        print("\n===============================================================================")
        print("Design variables:")
        for dv in optProb.variables:
            print(f"  - {dv}")

        print("\nConstraints:")
        for con in optProb.constraints:
            print(f"  - {con}")

        print("\nObjectives:")
        for obj in objectives:
            print(f"  - {obj}")
        print("===============================================================================\n")

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
            # Correct the time limit for the time that has elapsed already
            args.timeLimit = globalComm.bcast(args.timeLimit - (time.time() - startTime), root=0)
            optimiserOptions["Time limit"] = int(args.timeLimit)
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
        maxTrimIter = 6
        alphas = {}
        # Each proc should get the alpha for its flight point then do an allgather to get the right values on every proc
        alphas[f"{localFlightPoint.name}_AOA"] = localFlightPoint.alpha
        localAlphas = globalComm.allgather(alphas)
        for i in range(len(localAlphas)):
            alphas.update(localAlphas[i])
        # for fpName in flightPointsDict:
        #     alphas[f"{fpName}_AOA"] = flightPointsDict[fpName].alpha

        if ptRank == 0:
            print("Trimming:")
            print("=========")
        for ii in range(maxTrimIter):
            funcs, _ = MP.obj(alphas)
            res = []
            if ptRank == 0:
                print(f"Trimming Iteration {ii}")
                print("=========================")
            for fpName in flightPointsDict:
                res.append(funcs[f"{fpName}LiftDiff"])
                if ptRank == 0:
                    print("=" * 80)
                    print(f"{fpName}: AoA = {alphas[f'{fpName}_AOA']}, LiftDiff: {res[-1]}")
                    print("=" * 80)
            res = np.array(res).flatten()
            if all(np.abs(res) < 1e-1):
                if ptRank == 0:
                    print("=" * 80)
                    print("Trim solve converged!")
                    print("=" * 80)
                break

            sens, _ = MP.sens(alphas, funcs)
            if ptRank == 0:
                print(f"{sens=}")
            # Assemble the jacobian of all the lift differences w.r.t all the alphas
            jac = np.zeros((len(res), len(alphas)))
            for rowInd, fpName in enumerate(flightPointsDict):
                for colInd, fpName2 in enumerate(flightPointsDict):
                    if f"{fpName2}_AOA" in sens[f"{fpName}LiftDiff"]:
                        jac[rowInd, colInd] = sens[f"{fpName}LiftDiff"][f"{fpName2}_AOA"]
            if ptRank == 0:
                print(f"{jac=}")

            # Solve a least squares problem to solve Ax=b with bounds on x
            update = -lsq_linear(jac, res, bounds=(-1.0, 1.0), method="bvls", verbose=2).x
            if ptRank == 0:
                print(f"{update=}")

            for fpInd, fp in enumerate(flightPoints):
                alphas[f"{fp.name}_AOA"] += update[fpInd]

    elif args.task == "opt":
        if restartDict is not None:
            sol = optimiser(
                optProb,
                MP.sens,
                storeHistory=optHistFilename,
                restartDict=restartDict,
                timeLimit=args.timeLimit,
            )
        else:
            sol = optimiser(
                optProb,
                MP.sens,
                storeHistory=optHistFilename,
                timeLimit=args.timeLimit,
            )
        if args.optimiser == "snopt":
            # SNOPT Returns it's working arrays in a restart dictionary that we should save for future hot starts
            restartDict = sol[-1]
            sol = sol[0]
            if globalRank == 0:
                with open(os.path.join(outputDir, "SNOPTRestart.pkl"), "wb") as f:
                    dill.dump(restartDict, f)

# --- Write out the DVs and outputs that aren't too long (e.g not the ADflow state vector) in unscaled form to a pickle file ---
outputs = flightPointProb.model.list_outputs(
    return_format="dict",
    print_arrays=False,
    excludes=["*adflow_vol_coords", "*adflow_states"],
)
outputData = {}
for output in outputs:
    try:
        data = flightPointProb.get_val(output)
        if not hasattr(data, "__len__") or len(data) < 10000:
            outputData[output] = data
    except TypeError:
        pass

# Add wingbox tip displacement to the outputs
tipZDisp, tipTwist = getTipDisplacement(flightPointProb, localFlightPoint.name)
outputData[f"{localFlightPoint.name}-TipZDisp"] = tipZDisp
outputData[f"{localFlightPoint.name}-TipTwist"] = tipTwist

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

om.n2(
    flightPointProb,
    show_browser=False,
    outfile=os.path.join(localOutputDir, "AeroStruct-N2-Post-Run.html"),
)
om.n2(
    performanceProb,
    show_browser=False,
    outfile=os.path.join(outputDir, "Performance-N2-Post-Run.html"),
)
