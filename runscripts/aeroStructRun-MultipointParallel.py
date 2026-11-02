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
import fnmatch
import sys
from pprint import pprint as pp
import time
import copy
import argparse
from dataclasses import dataclass

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from mpi4py import MPI
import openmdao.api as om
from mphys import Multipoint, MPhysVariables
from mphys.scenarios import ScenarioAeroStructural, ScenarioStructural
from adflow.mphys import ADflowBuilder
from adflow import ADFLOW
from openaerostruct.mphys import AeroBuilder as OASBuilder
from idwarp import USMesh
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
from SETUP.setupWimpress import setupWimpress, PlanformConGroup, PLANFORM_CON_INPUT_NAMES
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
    getSource,
    addDesignVarsFromOpenMDAO,
    addConstraintFromOpenMDAO,
    writeOutputs,
    getTipDisplacement,
    getStructDVs,
    setupFuelMassGroup,
    mergeStructDVs,
    getTriangulatedSurface,
    AverageComp,
    getRelevantInputs,
    getDesignVarSources,
    allowResponsesAfterSetup,
    getAeroForceName,
    getPromotedInputNames,
    checkInputsHaveSources,
    computeProcsPerMember,
    gatherMergedDict,
    gatherUnion,
    setLocalVals,
    printBanner,
)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import performanceCalc  # noqa: E402
from BFLCalculation.OMGroups import STWTakeoffAnalysisGroup  # noqa: E402
from AircraftSpecs.STWFlightPoints import flightPointSets  # noqa: E402
from AircraftSpecs.FlightPoint import FlightPoint, LoadCase  # noqa: E402
from AircraftSpecs.STWSpecs import aircraftSpecs  # noqa: E402
from geometry.wingGeometry import wingGeometry  # noqa: E402

# --- Get the start time, this is used later for correcting the time limit passed to the optimiser ---
startTime = time.time()

np.set_printoptions(precision=16, linewidth=200)


# --- Get some info on the wing geometry ---
SPAN_INDEX = wingGeometry["spanIndex"]
INDEX_STRINGS = ["x", "y", "z"]
WING_SEMISPAN = wingGeometry["wing"]["semiSpan"]

# Flow conditions that OpenAeroStruct reads from inputs instead of the AeroProblem:
# (input name suffix, FlightPoint attribute, units)
VLM_FLOW_CONDITIONS = [
    ("rho", "rho", "kg/m**3"),
    (MPhysVariables.Aerodynamics.FlowConditions.MACH_NUMBER, "mach", None),
    ("v", "V", "m/s"),
    (MPhysVariables.Aerodynamics.FlowConditions.REYNOLDS_NUMBER, "re", "1/m"),
]

# --- Figure out where to put the results ---
OUTPUT_PARENT_DIR = getOutputDir()

# set this for convenience
isComplex = TACS.dtype == complex


# --- General options ---
parser.add_argument(
    "--task",
    type=str,
    default="check",
    choices=[
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
parser.add_argument("--vlmFlightPoints", nargs="*", default=[], help="List of flight points to use VLM for")
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
    default=aircraftSpecs["maxWingLoading"],
    help="Maximum allowable wing loading (kg/m^2)",
)
parser.add_argument(
    "--includeBFL",
    action="store_true",
    help="Include the balanced field length constraint",
    default=False,
)
parser.add_argument(
    "--useSimpleBFL",
    action="store_true",
    help="Use the simplified rotation phase in the OpenConcept BFL calculation",
    default=False,
)
parser.add_argument(
    "--maxBFL",
    type=float,
    default=aircraftSpecs["maxBFL"],
    help="Maximum allowable balanced field length (m)",
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
parser.add_argument("--vlmLevel", type=int, default=None, choices=[1, 2, 3])
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


# ==============================================================================
# Run configuration
# ==============================================================================
# Generally, massively scaling down the lift difference helps the optimiser make better progress, but in the trim
# task, where all we care about is hitting the lift constraints, we won't scale them down quite as much so that we
# really nail the right lift value.
LIFT_CON_SCALE = 1e-8
TRIM_LIFT_CON_SCALE = 1e-6


@dataclass(frozen=True)
class TaskSettings:
    """The command line arguments after the task overrides, and the settings that depend on the task."""

    # The command line arguments with the task's overrides applied
    args: argparse.Namespace
    # Whether this is a structural mass minimisation (`--optType structMass`)
    isStructOnlyOpt: bool
    # Whether to use the first flight point's structural mass as a dummy objective
    doUseStructMassObjective: bool
    # Whether to add the failure, fuel tank, planform, buffet and balanced field length constraints
    doAddDesignConstraints: bool
    # Whether to add the geometric constraints (on the first flight point only)
    doAddGeoConstraints: bool
    # Whether to add the TACS constraints (on the first flight point only)
    doAddTacsConstraints: bool
    # Whether to add the fuel tank volume constraint to the performance model
    doIncludeFuelVolumeConstraint: bool
    # Scaling factor for the lift constraints
    liftConScale: float
    # Whether the flight point functions evaluated are only the ones we need gradients of, rather than all disp funcs
    doEvaluateGradFuncsOnly: bool
    # Whether to run the model before the task itself, and again after setting any `--postInitDVs` (which are set
    # either way)
    doInitialRun: bool
    # Whether to write solution files from the initial runs
    doWriteInitialSolution: bool
    # Whether to write solution files every time the objective function is evaluated
    doWriteSolutionInObj: bool
    # Whether to write solution files every time the sensitivity function is evaluated
    doWriteSolutionInSens: bool


def resolveTaskSettings(args: argparse.Namespace) -> TaskSettings:
    """Apply the task's overrides to a copy of the command line arguments and work out the task-dependent settings.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments, which are not changed

    Returns
    -------
    TaskSettings
        The overridden arguments and the settings that depend on the task
    """
    taskArgs = copy.copy(args)

    # If we're doing a derivative check, we should at least enable some geometric and structural DVs
    if args.task == "derivCheck":
        taskArgs.addGeoDVs = True
        taskArgs.sweep = True
        taskArgs.addStructDVs = True
        # taskArgs.includeBFL = True
        taskArgs.useFuelMassDVs = True

    # If we are doing a trim task then we should disable the structural and geometric design variables
    if args.task == "trim":
        taskArgs.addStructDVs = False
        taskArgs.addGeoDVs = False
        taskArgs.addPlanformCons = False

    # If we're running the setup check, enable everything
    if args.task == "check":
        taskArgs.addGeoDVs = True
        taskArgs.addStructDVs = True
        taskArgs.usePanelLengthDVs = True
        taskArgs.useFuelMassDVs = True
        taskArgs.addPlanformCons = True
        taskArgs.twist = True
        taskArgs.sweep = True
        taskArgs.shape = True
        taskArgs.span = True
        taskArgs.xShift = True
        taskArgs.dihedral = True
        taskArgs.taper = True

    task = taskArgs.task
    isStructOnlyOpt = task == "opt" and taskArgs.optType == "structMass"
    return TaskSettings(
        args=taskArgs,
        isStructOnlyOpt=isStructOnlyOpt,
        doUseStructMassObjective=isStructOnlyOpt or task == "trim",
        doAddDesignConstraints=task != "trim",
        doAddGeoConstraints=task in ["opt", "check", "derivCheck"],
        doAddTacsConstraints=taskArgs.addStructDVs and task == "opt",
        doIncludeFuelVolumeConstraint=(
            taskArgs.useFuelMassDVs and task in ["opt", "check"] and (taskArgs.span or taskArgs.taper or taskArgs.shape)
        ),
        liftConScale=TRIM_LIFT_CON_SCALE if task == "trim" else LIFT_CON_SCALE,
        doEvaluateGradFuncsOnly=task == "derivCheck",
        doInitialRun=task != "check",
        doWriteInitialSolution=task == "analysis",
        doWriteSolutionInObj=task != "opt",
        doWriteSolutionInSens=task == "opt",
    )


# Account for additional drag on coarser mesh due to discretisation error.
# I estimated these values based on the average difference in cd between the L2 and L3 meshes and the L1 over an angle
# of attack range of 0-3 degrees
DRAG_CORRECTION = {2: 0.0003, 3: 0.0013}


@dataclass(frozen=True)
class RunContext:
    """Everything about this run that is fixed once the command line arguments are parsed and the procs are split."""

    # The task settings, including the command line arguments after the task overrides
    settings: TaskSettings
    # The multipoint object that runs each flight point on its own processor group
    multiPoint: multiPointSparse
    # The communicator spanning every proc
    globalComm: MPI.Intracomm
    # The communicator for this proc's flight point
    ptComm: MPI.Intracomm
    # The index of the flight point this proc belongs to
    ptID: int
    # All the flight points in this run
    flightPoints: list
    # The flight point this proc is working on
    localFlightPoint: FlightPoint | LoadCase
    # Whether this proc's flight point is aerostructural, rather than a structural-only load case
    isAeroStruct: bool
    # Whether this proc's flight point is a cruise point
    isCruisePoint: bool
    # Whether this proc's flight point uses VLM instead of ADflow
    usingVLM: bool
    # The aero mesh level for this proc's flight point: `--vlmLevel` on VLM procs if given, otherwise `--aeroLevel`
    aeroLevel: int
    structMeshFile: str
    aeroMeshFile: str
    ffdFile: str
    outputDir: str
    geoOutputDir: str
    localOutputDir: str
    localAeroOutputDir: str
    localStructOutputDir: str
    funcTimingFile: str
    funcSensTimingFile: str
    # A copy of the aircraft specs with the mission range scaled and the drag correction applied
    aircraftSpecs: dict

    @property
    def args(self) -> argparse.Namespace:
        """The command line arguments after the task overrides."""
        return self.settings.args

    @property
    def task(self) -> str:
        """The task being run."""
        return self.args.task

    @property
    def globalRank(self) -> int:
        """This proc's rank in the global communicator."""
        return self.globalComm.rank

    @property
    def ptRank(self) -> int:
        """This proc's rank in its flight point's communicator."""
        return self.ptComm.rank


def buildRunContext(args: argparse.Namespace, comm: MPI.Intracomm) -> RunContext:
    """Resolve the task settings, split the procs between the flight points and work out this proc's configuration.

    This creates no files or directories and prints nothing.

    Parameters
    ----------
    args : argparse.Namespace
        The command line arguments, which are not changed
    comm : MPI.Intracomm
        The communicator spanning every proc

    Returns
    -------
    RunContext
        This proc's run configuration
    """
    settings = resolveTaskSettings(args)
    # Everything below reads the arguments after the task overrides
    args = settings.args

    # Copy the aircraft specs so that the changes below don't affect anything else that imports them
    runAircraftSpecs = copy.deepcopy(aircraftSpecs)

    # Define flight points to use
    flightPoints = flightPointSets[args.flightPointSet]
    # Set reference geometric values
    for fp in flightPoints:
        fp.areaRef = runAircraftSpecs["refArea"]
        fp.chordRef = runAircraftSpecs["refChord"]

    flightPointsDict = {fp.name: fp for fp in flightPoints}

    # Raise an error if the user is trying to use a VLM for a flightpoint that doesn't support VLM
    for vlmFP in args.vlmFlightPoints:
        errorString = f"You specified flight point {vlmFP} to use VLM, but this flight point "
        if vlmFP not in flightPointsDict:
            raise ValueError(
                errorString
                + f"is not in the selected flight point set. Selected flight point set ({args.flightPointSet}) includes {list(flightPointsDict.keys())}"
            )
        if not flightPointsDict[vlmFP].canUseVLM:
            raise ValueError(errorString + "cannot be modeled using VLM")

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
    runAircraftSpecs["range"] *= args.rangeScale

    # Account for additional drag on coarser mesh due to discretisation error, based on the command line aero level
    if args.aeroLevel > 1:
        runAircraftSpecs["extraDragCoeff"] -= DRAG_CORRECTION[args.aeroLevel]

    # Define the output directory
    outputDir = os.path.join(OUTPUT_PARENT_DIR, args.output)

    # --- Processor allocation ---
    #  Create multipoint communication object
    multiPoint = multiPointSparse(comm)

    # We will use a single processor set which contains each flight point as a separate member
    nMembers = len(flightPoints)
    if len(args.procs) == 0:
        # If no processors are specified, split them evenly between the members
        procsPerMember = computeProcsPerMember(comm.size, nMembers)
    else:
        if len(args.procs) != nMembers:
            raise ValueError(f"You specified {nMembers} flight points but only {len(args.procs)} processor counts")
        else:
            procsPerMember = args.procs
    fpNamesWithoutProcs = [fp.name for fp, numProcs in zip(flightPoints, procsPerMember, strict=True) if numProcs < 1]
    if len(fpNamesWithoutProcs) > 0:
        raise ValueError(f"Every flight point needs at least one processor, but these have none: {fpNamesWithoutProcs}")
    multiPoint.addProcessorSet("all", nMembers=nMembers, memberSizes=procsPerMember)

    # Create communicators
    # ptComm : MPI.Intracomm
    #     This is the communicator for the member of the procSet. Basically,
    #     this is the communciator that the (parallel) analysis should be
    #     created on
    # ptID : int
    #     This is the index of the group that this processor belongs to
    ptComm, _, _, _, ptID = multiPoint.createCommunicators()

    # For convenience, store which flightPoint we're working with on this proc
    localFlightPoint = flightPoints[ptID]
    isAeroStruct = isinstance(localFlightPoint, FlightPoint)
    isCruisePoint = hasCruisePoint and "cruise" in localFlightPoint.name.lower()

    # Store a flag the tells us if we're using a VLM for this proc's flightpoint
    usingVLM = localFlightPoint.name in args.vlmFlightPoints

    # Define location of input files
    structMeshFile = getStructMeshPath(level=args.structLevel, order=args.structOrder)

    aeroLevel = args.vlmLevel if usingVLM and args.vlmLevel is not None else args.aeroLevel
    aeroMeshFile = getAeroMeshPath(level=aeroLevel, useVLM=usingVLM)

    ffdFile = getFFDPath(level=args.ffdLevel)

    # Define the output directories
    geoOutputDir = os.path.join(outputDir, "Geometry")
    localOutputDir = os.path.join(outputDir, localFlightPoint.name)
    localAeroOutputDir = os.path.join(localOutputDir, "aero")
    localStructOutputDir = os.path.join(localOutputDir, "struct")

    # --- Csv files that we will store timing data in ---
    funcTimingFile = os.path.join(localOutputDir, f"{localFlightPoint.name}-FuncTiming.csv")
    funcSensTimingFile = os.path.join(localOutputDir, f"{localFlightPoint.name}-FuncSensTiming.csv")

    return RunContext(
        settings=settings,
        multiPoint=multiPoint,
        globalComm=comm,
        ptComm=ptComm,
        ptID=ptID,
        flightPoints=flightPoints,
        localFlightPoint=localFlightPoint,
        isAeroStruct=isAeroStruct,
        isCruisePoint=isCruisePoint,
        usingVLM=usingVLM,
        aeroLevel=aeroLevel,
        structMeshFile=structMeshFile,
        aeroMeshFile=aeroMeshFile,
        ffdFile=ffdFile,
        outputDir=outputDir,
        geoOutputDir=geoOutputDir,
        localOutputDir=localOutputDir,
        localAeroOutputDir=localAeroOutputDir,
        localStructOutputDir=localStructOutputDir,
        funcTimingFile=funcTimingFile,
        funcSensTimingFile=funcSensTimingFile,
        aircraftSpecs=runAircraftSpecs,
    )


ctx = buildRunContext(parser.parse_args(), MPI.COMM_WORLD)

if ctx.args.profile:
    import cProfile

    pr = cProfile.Profile()
    pr.enable()

# ==============================================================================
# MELD settings
# ==============================================================================
structMeshSpacing = {
    1: 0.035,
    2: 0.07,
    3: 0.14,
    4: 0.23,
}

if ctx.usingVLM:
    aeroMeshChordSpacing = {
        1: 0.44,
        2: 0.88,
        3: 1.76,
    }
    aeroMeshSpanSpacing = {
        1: 0.38,
        2: 0.76,
        3: 1.52,
    }
else:
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


if ctx.args.nMeld is None:
    # This an approximation of the ratio of structural nodes to aero nodes in the coarsest part of the aero mesh, used to
    # tell MELD how many structural nodes to connect each aero node to. Multiplying this estimate by 4 seems to
    # provide a reasonable N value where the aero forces are not concentrated at the nearest structural nodes.
    MELD_MESH_FACTOR = max(
        100,
        int(
            4
            * aeroMeshSpanSpacing[ctx.aeroLevel]
            * aeroMeshChordSpacing[ctx.aeroLevel]
            / structMeshSpacing[ctx.args.structLevel] ** 2
        ),
    )
    # The Super fine aero mesh is particularly prone to negative volumes due to structural deformations so we use a larger
    # lower limit on N for it
    if ctx.aeroLevel == 1:
        MELD_MESH_FACTOR = max(200, MELD_MESH_FACTOR)
else:
    MELD_MESH_FACTOR = ctx.args.nMeld

# Multiply by 3 for VLM because we'll be transferring to top and bottom of wingbox
if ctx.usingVLM:
    MELD_MESH_FACTOR *= 3

# ==============================================================================
# Setup output directories and files
# ==============================================================================

# Create output directories
if ctx.globalRank == 0:
    os.makedirs(ctx.geoOutputDir, exist_ok=True)

if ctx.ptRank == 0:
    os.makedirs(ctx.localStructOutputDir, exist_ok=True)
    if ctx.isAeroStruct:
        os.makedirs(ctx.localAeroOutputDir, exist_ok=True)

# --- Create empty csv files that we will store timing data in ---
if ctx.ptRank == 0:
    for fileName in [ctx.funcTimingFile, ctx.funcSensTimingFile]:
        file = open(fileName, "w")
        file.close()

# Print out and save the full list of command line arguments
saveRunCommand(parser, ctx.args, ctx.outputDir)

for ii in range(ctx.globalComm.size):
    if ctx.globalRank == ii:
        print(f"Processor {ctx.globalRank} is assigned to flight point {ctx.localFlightPoint.name}", flush=True)
    ctx.globalComm.Barrier()

# --- Redirect I/O ---
if ctx.ptRank == 0:
    outFile = open(os.path.join(ctx.localOutputDir, "stdout.out"), "w")
    redirectIO(outFile)
    print("===============================================================================", flush=True)
    print(ctx.localFlightPoint.name, flush=True)
    print("===============================================================================", flush=True)

# ==============================================================================
# TACS Setup
# ==============================================================================
problemOptions = {
    "outputDir": ctx.localStructOutputDir,
    "useMonitor": not ctx.args.nonlinear,
    "monitorFrequency": 1,
}
structBuilder = setupTACS.createTacsBuilder(
    ctx.args,
    ctx.structMeshFile,
    ctx.flightPoints,
    problemOptions,
    coupling_loads=[MPhysVariables.Structures.Loads.AERODYNAMIC] if ctx.isAeroStruct else None,
    write_solution=False,
    res_ref=1e3,
)

# ==============================================================================
# Aero Setup
# ==============================================================================
if ctx.usingVLM:
    VLMMesh = np.load(ctx.aeroMeshFile)
    surf_dict = {
        # Wing definition
        "name": "wall",  # name of the surface
        "symmetry": True,  # if true, model one half of wing
        "S_ref_type": "wetted",  # how we compute the wing area, can be 'wetted' or 'projected'
        "CL0": 0.0,  # CL of the surface at alpha=0
        "CD0": 0.0,  # CD of the surface at alpha=0
        "with_viscous": False,  # if true, compute viscous drag,
        "k_lam": 0.05,  # percentage of chord with laminar flow
        "t_over_c": 0.12,  # thickness over chord ratio (NACA0012)
        "c_max_t": 0.303,  # chordwise location of maximum (NACA0012)
        "with_wave": False,
        "mesh": VLMMesh,
    }
    aeroBuilder = OASBuilder(
        [surf_dict], options={"compressible": True, "output_dir": ctx.localAeroOutputDir, "write_solution": True}
    )
    aeroOptions = None
else:
    aeroOptions = getADflowOptions(ctx.aeroMeshFile, ctx.localAeroOutputDir, aerostructural=True)
    if ctx.aeroLevel == 3:
        aeroOptions["anksecondordswitchtol"] *= 10
        aeroOptions["rkreset"] = True
    if ctx.args.noFiles:
        aeroOptions["writeTecplotSurfaceSolution"] = False
        aeroOptions["writevolumesolution"] = False
        aeroOptions["writesurfacesolution"] = False
    if ctx.args.aeroTol is not None:
        aeroOptions["L2ConvergenceRel"] = ctx.args.aeroTol
    if ctx.args.aeroMaxIter is not None:
        aeroOptions["nCycles"] = ctx.args.aeroMaxIter

    warpOptions = getIDWarpOptions(ctx.aeroMeshFile)
    aeroBuilder = ADflowBuilder(
        aeroOptions,
        mesh_options=warpOptions,
        scenario="aerostructural",
        write_solution=False,
        res_ref=1e7,
        restart_failed_analysis=False,
        linear_precon_only=False,
    )


# ==============================================================================
# Transfer Scheme Setup
# ==============================================================================
# --- First define which nodes in the aero and struct meshes will be involved in the transfer ---
if ctx.args.transferTo == "all":
    ldTransferBodies = None
elif ctx.args.transferTo in [
    "skin",
    "skin+ribends",
]:  # The rib-spar intersection nodes will be added later
    ldTransferBodies = [{"aero": ["wall"], "struct": ["SKIN"]}]
elif ctx.args.transferTo == "skin+spar":
    ldTransferBodies = [{"aero": ["wall"], "struct": ["SKIN", "SPAR"]}]

isym = SPAN_INDEX  # spanwise-symmetry
if ctx.args.transferType is None:
    useLinearizedMELD = not ctx.args.nonlinear
elif ctx.args.transferType.lower() == "linear":
    useLinearizedMELD = True
elif ctx.args.transferType.lower() == "nonlinear":
    useLinearizedMELD = False

ldxferBuilder = MeldBuilder(
    aeroBuilder,
    structBuilder,
    isym=isym,
    n=MELD_MESH_FACTOR,
    linearized=useLinearizedMELD,
    body_tags=ldTransferBodies,
)


# --- Spanwise lift distribution and slice file output for the ADflow solvers ---
LIFT_DISTRIBUTION_NUM_POINTS = 100
NUM_SLICES = 51
SLICE_ROOT_POSITION = 1e-5
SLICE_TIP_SPAN_FRACTION = 0.99


def addAeroOutputs(solver) -> None:
    """Add the spanwise lift distribution and the spanwise slices to an ADflow solver's output files

    Parameters
    ----------
    solver : adflow.ADFLOW
        ADflow solver to add the outputs to
    """
    solver.addLiftDistribution(LIFT_DISTRIBUTION_NUM_POINTS, INDEX_STRINGS[SPAN_INDEX])
    slicePositions = np.linspace(SLICE_ROOT_POSITION, WING_SEMISPAN * SLICE_TIP_SPAN_FRACTION, NUM_SLICES)
    solver.addSlices(INDEX_STRINGS[SPAN_INDEX], slicePositions)


# Now we can create the MPhys model for each flight point
class AnalysisPoint(Multipoint):
    def initialize(self):
        self.options.declare("ctx", types=RunContext, desc="Run configuration")
        self.options.declare("aeroBuilder", desc="MPhys builder for this flight point's aero solver (ADflow or VLM)")
        self.options.declare("structBuilder", desc="MPhys builder for the TACS structural model")
        self.options.declare("ldxferBuilder", desc="MPhys builder for the MELD load and displacement transfer")
        self.options.declare(
            "aeroOptions",
            default=None,
            types=(dict, type(None)),
            desc="ADflow options, used to create the dummy aero solver, None when using VLM",
        )

    def setup(self):
        ctx = self.options["ctx"]
        self.fpName = ctx.localFlightPoint.name

        builders = {}
        disciplineVariables = {}

        # ivc to keep the top level DVs
        dvSys = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])

        if ctx.isAeroStruct:
            self._addAeroDiscipline(builders, disciplineVariables)
        self._addStructDiscipline(builders, disciplineVariables)
        if ctx.isAeroStruct:
            self._addLoadTransfer()
        self._addGeometry(builders, disciplineVariables)
        if ctx.isAeroStruct:
            self._addPlanformCalc()
        self._addStructAndFuelDVs(dvSys)
        self._addScenario(disciplineVariables)
        if ctx.isAeroStruct:
            self._addAeroForceOutputs()
            self._addPerformanceGroups()

    def configure(self):
        ctx = self.options["ctx"]
        if ctx.isAeroStruct:
            self._configureFlowConditions()
        self._configureGeometry()

        # Only add TACS constraints on the first proc set and if we're actually doing an optimisation
        if ctx.ptID == 0 and ctx.settings.doAddTacsConstraints:
            add_tacs_constraints(getattr(self, self.fpName))

        if ctx.isAeroStruct:
            self._configureCoupledSolvers()
        self._addDummyAeroSolver()

    def _addAeroDiscipline(self, builders: dict, disciplineVariables: dict) -> None:
        """Set up the aero discipline"""
        ctx = self.options["ctx"]
        aeroBuilder = self.options["aeroBuilder"]

        # --- initialize aero builder ---
        aeroBuilder.initialize(self.comm)
        self.aeroSolver = None if ctx.usingVLM else aeroBuilder.get_solver()
        builders["aero"] = aeroBuilder
        disciplineVariables["aero"] = MPhysVariables.Aerodynamics.Surface

        # Add lift distribution and slice file output
        if not ctx.args.noFiles and self.aeroSolver is not None:
            addAeroOutputs(self.aeroSolver)

    def _addStructDiscipline(self, builders: dict, disciplineVariables: dict) -> None:
        """Set up the TACS structural discipline"""
        ctx = self.options["ctx"]
        structBuilder = self.options["structBuilder"]

        structBuilder.initialize(self.comm)
        self.FEAAssembler = structBuilder.get_fea_assembler()
        disciplineVariables["struct"] = MPhysVariables.Structures
        builders["struct"] = structBuilder

        setupTACS.writeStructDVMap(self.FEAAssembler, ctx.args, ctx.outputDir, ctx.globalComm)

    def _addLoadTransfer(self) -> None:
        """Set up the MELD load and displacement transfer"""
        ctx = self.options["ctx"]
        structBuilder = self.options["structBuilder"]
        ldxferBuilder = self.options["ldxferBuilder"]

        # Find the nodes at the intersections of the spars and ribs and include them in the LDTransfer
        if ctx.args.transferTo == "skin+ribends":
            FEASolver = structBuilder.get_fea_assembler()
            sparComps = FEASolver.selectCompIDs(include="SPAR")
            ribComps = FEASolver.selectCompIDs(include="RIB")
            sparNodes = FEASolver.getGlobalNodeIDsForComps(sparComps, nastranOrdering=True)
            ribNodes = FEASolver.getGlobalNodeIDsForComps(ribComps, nastranOrdering=True)
            sharedNodes = list(set(sparNodes).intersection(set(ribNodes)))
            ldxferBuilder.body_tags[0]["struct"] += sharedNodes
        ldxferBuilder.initialize(self.comm)

    def _addGeometry(self, builders: dict, disciplineVariables: dict) -> None:
        """Add the mesh and geometry components"""
        ctx = self.options["ctx"]

        # Setup mesh components for each discipline
        for dName in disciplineVariables:
            self.add_subsystem(f"mesh_{dName}", builders[dName].get_mesh_coordinate_subsystem())

        geometryComp = OM_DVGEOCOMP(file=ctx.ffdFile, type="ffd", options={"isComplex": isComplex})
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

    def _addPlanformCalc(self) -> None:
        """Add the WimpressCalc planform calculation and the planform constraints"""
        ctx = self.options["ctx"]

        wimpressCalc, wimpressCalcComp, _ = setupWimpress(addPlanformCons=ctx.args.addPlanformCons)
        if ctx.globalRank == 0:
            wimpressCalc.writeTecplot(os.path.join(ctx.geoOutputDir, "wimpressCalcInit.dat"))
        # Because the WimpressCalc component isn't following the MPhys convention we can just add the wimpress
        # coordinates directly as a pointset. The geometry component will then output the deformed coordinates without
        # needing a corresponding input. Unfortunately we can't do this in setup because the DVGeo instances associated
        # with the geometry component don't exist yet (they're created in its setup method), so we'll add the pointset
        # and connect things in configure instead.
        self.add_subsystem("PlanformValues", wimpressCalcComp)

        if ctx.args.addPlanformCons:
            self.add_subsystem("PlanformCons", PlanformConGroup(wingGeometry=wingGeometry))
            for varName in PLANFORM_CON_INPUT_NAMES:
                self.connect(f"PlanformValues.{varName}", f"PlanformCons.{varName}")

    def _addStructAndFuelDVs(self, dvSys: om.IndepVarComp) -> None:
        """Add the TACS DVs and the fuel mass computation"""
        ctx = self.options["ctx"]
        structBuilder = self.options["structBuilder"]

        initStructDVs, fuelMassDVInds, sizingDVInds = getStructDVs(structBuilder)
        self.numRibBays = len(fuelMassDVInds)
        dvSys.add_output("dv_struct", val=initStructDVs[sizingDVInds])

        if ctx.args.addStructDVs:
            lb, ub = structBuilder.get_dv_bounds()
            structDVScaling = np.array(structBuilder.fea_assembler.scaleList)
            self.add_design_var(
                "dv_struct",
                lower=lb[sizingDVInds],
                upper=ub[sizingDVInds],
                scaler=structDVScaling[sizingDVInds] * ctx.args.structScalingFactor,
            )

        # --- Fuel mass computation ---
        fuelDVName = f"{self.fpName}-fuelMass"
        dvSys.add_output(
            fuelDVName, val=0.0, shape=1
        )  # Total fuel mass value, placeholder for now TODO: How should this be set if it's not a DV?
        if ctx.args.useFuelMassDVs:
            self.add_design_var(
                fuelDVName,
                lower=0.0,
                scaler=1e-4,
            )

        setupFuelMassGroup(self, fuelDVName, numRibBays=self.numRibBays)

        # Now add component that merges the fuel mass DVs and the structural DVs back into a full set of TACS DVs
        mergeStructDVs(self, fuelMassDVInds, sizingDVInds)

    def _addScenario(self, disciplineVariables: dict) -> None:
        """Create the scenario and connect the geometry and structural DVs to it"""
        ctx = self.options["ctx"]
        aeroBuilder = self.options["aeroBuilder"]
        structBuilder = self.options["structBuilder"]
        ldxferBuilder = self.options["ldxferBuilder"]

        if ctx.isAeroStruct:
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

    def _addAeroForceOutputs(self) -> None:
        """Fix lift and drag outputs"""
        ctx = self.options["ctx"]

        # ADflow and OpenAerostruct don't compute lift and drag in the same way, so we need to fix the outputs
        # ADflow computes the lift/drag of the single wing, the outputs are not promoted and are called "lift" and "drag"
        # OpenAerostruct computes the lift and drag for 2 wings, the outputs are promoted and are called "L" and "D"

        aeroForces = ["Lift"]
        if ctx.isCruisePoint:
            aeroForces.append("Drag")

        # To fix this, we'll make an addSubtractComp that will create an output called "lift" and "drag" that is the lift/drag of the single wing
        if ctx.usingVLM:
            scaleFactor = 0.5
            inputNames = [
                f"{self.fpName}.{f[0].upper()}" for f in aeroForces
            ]  # [f"{self.fpName}.L", f"{self.fpName}.D"]
        else:
            scaleFactor = 1.0
            inputNames = [
                f"{self.fpName}.aero_post.{f.lower()}" for f in aeroForces
            ]  # [f"{self.fpName}.aero_post.lift", f"{self.fpName}.aero_post.drag"]

        for outputName, inputName in zip(aeroForces, inputNames, strict=True):
            self.add_subsystem(
                f"{outputName}CorrectionComp",
                om.ExecComp(
                    f"{getAeroForceName(self.fpName, outputName)} = {scaleFactor} * F",
                    units="N",
                ),
                promotes_outputs=["*"],
            )
            self.connect(inputName, f"{outputName}CorrectionComp.F")

    def _addPerformanceGroups(self) -> None:
        """Set up the performance calculations"""
        ctx = self.options["ctx"]

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
            sepSensorFuncName = "sepsensorksarea" if ctx.args.sepSensorType == "new" else "sepsensor"
            sepSensorFuncName = f"{self.fpName}.aero_post.{sepSensorFuncName}"
            self.connect(sepSensorFuncName, f"{buffetConName}.SepArea")

        if ctx.ptID == 0:
            massComp = performanceCalc.AirframeMassGroup(
                aircraftSpecs=ctx.aircraftSpecs,
            )
            self.add_subsystem(
                "airframeMass",
                massComp,
                promotes=[
                    "landingGrossMass",
                    ("wingboxMass", f"{self.fpName}.mass"),
                ],
            )
        if ctx.isCruisePoint:
            fuelBurnGroup = performanceCalc.FuelBurnGroup(
                aircraftSpecs=ctx.aircraftSpecs,
                flightPoint=ctx.localFlightPoint,
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
                    getAeroForceName(self.fpName, force),
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

            # --- Compute the fuel burn/TOGM pareto objective if necessary ---
            if ctx.args.optType == "pareto":
                paretoComp = om.AddSubtractComp(
                    output_name="paretoObj",
                    input_names=["totalFuelBurn", "takeoffMass"],
                    scaling_factors=[
                        ctx.args.paretoWeight / 1e4,
                        (1 - ctx.args.paretoWeight) / ctx.aircraftSpecs["refMTOW"],
                    ],
                )
                self.add_subsystem("paretoComp", paretoComp, promotes=["*"])

            # --- Compute the balanced field length ---
            if ctx.args.includeBFL:
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
                    AverageComp(inputName="sectionToC", outputName="avgToC"),
                    promotes_outputs=["*"],
                )
                self.connect("geometry.SectionToC", "avgToC.sectionToC")

                takeoffGroup = STWTakeoffAnalysisGroup(
                    ivc_excludes=[
                        "ac|weights|MTOW",
                        "ac|geom|wing|S_ref",
                        "ac|geom|wing|AR",
                        "ac|geom|wing|c4sweep",
                        "ac|geom|wing|taper",
                        "ac|geom|wing|toverc",
                    ],
                    transition_method="simplified" if ctx.args.useSimpleBFL else "ode",
                )
                self.add_subsystem("takeoff", takeoffGroup)
                self.connect("takeoffMass", "takeoff.ac|weights|MTOW")
                self.connect("doubleWingArea", "takeoff.ac|geom|wing|S_ref")
                self.connect("PlanformValues.aspectRatio", "takeoff.ac|geom|wing|AR")
                self.connect("PlanformValues.QCSweep", "takeoff.ac|geom|wing|c4sweep")
                self.connect("PlanformValues.trapTaper", "takeoff.ac|geom|wing|taper")
                self.connect("avgToC", "takeoff.ac|geom|wing|toverc")

    def _configureFlowConditions(self) -> None:
        """Add the angle of attack DV and give the aero solver its flow conditions"""
        ctx = self.options["ctx"]
        dvComp = self.dvs

        fp = ctx.localFlightPoint
        scenario = getattr(self, self.fpName)
        # Add this point's angle of angle of attack as a DV
        alphaDVName = f"{fp.name}_AOA"
        dvComp.add_output(alphaDVName, val=fp.alpha, units="deg")
        self.add_design_var(alphaDVName, lower=-20.0, upper=20.0, scaler=0.1)
        if ctx.usingVLM:
            # OpenAerostruct gets flight conditions from inputs instead of the aero problem
            self.connect(alphaDVName, f"{fp.name}.{MPhysVariables.Aerodynamics.FlowConditions.ANGLE_OF_ATTACK}")
            for suffix, fpAttr, units in VLM_FLOW_CONDITIONS:
                dvComp.add_output(f"{fp.name}-{suffix}", val=getattr(fp, fpAttr), units=units)
                self.connect(f"{fp.name}-{suffix}", f"{fp.name}.{suffix}")
        else:
            # Give ADflow the aero problem for this procset's flight point and add the angle of attack as a DV
            fp.addDV("alpha", value=fp.alpha, name="aoa", units="deg")
            scenario.coupling.aero.mphys_set_ap(fp)
            scenario.aero_post.mphys_set_ap(fp)
            self.connect(
                alphaDVName,
                [f"{self.fpName}.coupling.aero.aoa", f"{self.fpName}.aero_post.aoa"],
            )

    def _configureGeometry(self) -> None:
        """Set up the geometric DVs and constraints, and connect the geometry to the fuel mass and planform groups"""
        ctx = self.options["ctx"]
        geometryComp = self.geometry
        dvComp = self.dvs

        # Setup the geometric DVs and constraints, we only need to compute the geometric constraints (LE radius,
        # thickness, area etc) on one proc set
        geometryComp.nom_setConstraintSurface(getTriangulatedSurface())
        setupDVGeo(
            ctx.args,
            self,
            geometryComp,
            dvComp,
            dvScaleFactor=ctx.args.geoScalingFactor,
            geoCompName="geometry",
            addGeoDVs=ctx.args.addGeoDVs,
            addGeoConstraints=ctx.ptID == 0 and ctx.settings.doAddGeoConstraints,
            computeRibBayVolumes=True,
            computeToC=ctx.isCruisePoint,
        )

        DVCon = self.geometry.nom_getDVCon()
        if ctx.globalRank == 0:
            DVCon.writeTecplot(os.path.join(ctx.geoOutputDir, "DVConstraints.dat"))
            DVCon.writeSurfaceTecplot(os.path.join(ctx.geoOutputDir, "DVConstraintsSurface.dat"))
            DVCon.writeSurfaceSTL(os.path.join(ctx.geoOutputDir, "DVConstraintsSurface.stl"))

        # Connect rib bay volumes to the fuel distribution group
        for ii in range(self.numRibBays):
            self.connect(f"geometry.RibBay-Volume_{ii}", f"RibBay-Volume_{ii}")

        if ctx.isAeroStruct:
            # Add the wimpress coordinates as a pointset to the geometry component and connect them to the wimpress comp
            wimpressComp = self.PlanformValues
            wimpressCoordName = "x_wimpress"
            geometryComp.nom_addPointSet(
                wimpressComp.wimpressCalc.getCoords(packed=True),
                ptName=wimpressCoordName,
                distributed=False,
            )
            self.connect(f"geometry.{wimpressCoordName}", "PlanformValues.x_wimpress")

    def _configureCoupledSolvers(self) -> None:
        """Set up the coupled nonlinear and linear solvers"""
        ctx = self.options["ctx"]

        scenario = getattr(self, self.fpName)
        scenario.coupling.nonlinear_solver = om.NonlinearBlockGS(
            maxiter=100,
            iprint=2,
            atol=1e-4 * ctx.args.tolFactor,
            rtol=1e-8 * ctx.args.tolFactor,
            use_aitken=not ctx.args.noAitken,
            aitken_initial_factor=ctx.args.aitkenInitFactor,
            aitken_max_factor=1.2,
            # reraise_child_analysiserror=True,
            # use_apply_nonlinear=True, This doesn't work
            restart_from_successful=True,
            err_on_non_converge=True,
        )
        scenario.coupling.linear_solver = om.PETScKrylov(
            atol=1e-4 * ctx.args.tolFactor,
            rtol=1e-8 * ctx.args.tolFactor,
            maxiter=50,
            iprint=2,
            rhs_checking={
                "max_cache_entries": 10,
                "check_zero": True,
                "collect_stats": True,
                "auto": True,
                "verbose": ctx.ptRank == 0,
                "atol": 1e-14,
                "rtol": 1e-14,
            },
        )
        scenario.coupling.linear_solver.precon = om.LinearBlockGS(maxiter=1, iprint=-2, use_aitken=False, rtol=1e-1)

    def _addDummyAeroSolver(self) -> None:
        """Set up the dummy aero solver"""
        ctx = self.options["ctx"]
        aeroOptions = self.options["aeroOptions"]
        geometryComp = self.geometry

        # In order to write out the jig shape of the OML and its twist distribution, I need to setup a dummy ADflow
        # instance that will recieve surface coordinates directly from the DVGeo without structural displacements. I
        # will then write solution files from this solver without every actually running it
        self.dummyAeroSolver = None
        if ctx.isAeroStruct:
            if not ctx.args.noFiles and self.aeroSolver is not None:
                if ctx.ptID == 0:
                    self.dummyAeroSolver = ADFLOW(options=aeroOptions, comm=self.comm)
                    self.dummyAeroSolver.setAeroProblem(ctx.localFlightPoint)
                    self.dummyAeroSolver.setDVGeo(geometryComp.nom_getDVGeo())
                    mesh = USMesh(options=self.aeroSolver.mesh.options, comm=self.comm)
                    self.dummyAeroSolver.setMesh(mesh)
                    addAeroOutputs(self.dummyAeroSolver)
                    # In order to get solution files that don't contain NaNs that break tecplot, we need to actually run the
                    # solver, so we can just set the iteration limit to 0 so that the solver just does it's initialisation
                    # steps and doesn't actually waste any time solving.
                    self.dummyAeroSolver.setOption("nCycles", 0)


# --- Now actually create the OpenMDAO model for each point ---
flightPointProb = om.Problem(reports=None, comm=ctx.ptComm, work_dir=ctx.localOutputDir)
flightPointProb.model = AnalysisPoint(
    ctx=ctx,
    aeroBuilder=aeroBuilder,
    structBuilder=structBuilder,
    ldxferBuilder=ldxferBuilder,
    aeroOptions=aeroOptions,
)

# --- Finally create the aircraft performance OpenMDAO model ---
performanceProb = om.Problem(reports=None, comm=ctx.globalComm, work_dir=ctx.outputDir)
includeFuelMassConstraints = ctx.args.useFuelMassDVs
performanceProb.model = performanceCalc.FuelAndMassConstraintGroup(
    aircraftSpecs=ctx.aircraftSpecs,
    flightPointSet=ctx.flightPoints,
    includeFuelMassConstraints=includeFuelMassConstraints,
    includeFuelVolumeConstraint=ctx.settings.doIncludeFuelVolumeConstraint,
)

# ==============================================================================
# Setup objective
# ==============================================================================
if ctx.settings.doUseStructMassObjective:
    if ctx.ptID == 0:
        # For the trim task we setup a dummy objective that doesn't depend on the trim variables so that the optimiser
        # just satisfies the trim constraints
        flightPointProb.model.add_objective(
            f"{ctx.localFlightPoint.name}.mass", scaler=1e-3, cache_linear_solution=True
        )
elif ctx.args.optType == "fuelburn":
    if ctx.isCruisePoint:
        flightPointProb.model.add_objective("totalFuelBurn", scaler=1e-4, cache_linear_solution=True)
elif ctx.args.optType == "pareto":
    if ctx.isCruisePoint:
        flightPointProb.model.add_objective("paretoObj", scaler=1.0, cache_linear_solution=True)
else:
    raise ValueError("No objective function defined")

# ==============================================================================
# Setup constraints
# ==============================================================================

# --- Lift constraints ---
# We have a lift=mass*g*loadFactor constraint for each aerostructural flight point. These should be enforced in both
# the trim and opt tasks.

for fp in ctx.flightPoints:
    if isinstance(fp, FlightPoint):
        performanceProb.model.add_constraint(
            f"{fp.name}LiftDiff",
            equals=0.0,
            scaler=ctx.settings.liftConScale,
            cache_linear_solution=True,
        )

# --- Fuel mass consistency constraints ---
# These constraints ensure that the design variable that controls the magnitude of the fuel mass loads is consistent
# with this flight point's correct fuel mass, which depends on the computed mission fuel burn and the mass
# configuration at this flight point. We include these in both opt and trim tasks if the we are using the fuel mass
# DVs and there is a cruise point to compute the fuel burn.
if includeFuelMassConstraints:
    for fp in ctx.flightPoints:
        performanceProb.model.add_constraint(
            f"{fp.name}FuelMassDiff",
            equals=0.0,
            scaler=1e-4,
            cache_linear_solution=True,
        )

if ctx.settings.doAddDesignConstraints:
    # --- TACS Failure constraints ---
    # These should be included in all opt tasks
    if ctx.localFlightPoint.failureGroups is not None:
        for group in ctx.localFlightPoint.failureGroups:
            failureConName = f"{ctx.localFlightPoint.name}.{group}_ksFailure"
            flightPointProb.model.add_constraint(failureConName, upper=1.0, scaler=1.0, cache_linear_solution=True)

    if not ctx.settings.isStructOnlyOpt and ctx.args.addGeoDVs:
        # --- Fuel tank capacity constraint ---
        # We only apply this if we have a cruise point to compute the fuel burn and if we have geometric DVs that
        # will affect the fuel tank volume
        if ctx.settings.doIncludeFuelVolumeConstraint:
            performanceProb.model.add_constraint("fuelTankUsage", upper=1.0, cache_linear_solution=True)
        # --- Planform constraints ---
        if ctx.isCruisePoint:
            if ctx.args.span or ctx.args.taper:
                # This constraint should only be applied if the optimiser has control over the wing area
                flightPointProb.model.add_constraint(
                    "wingLoading",
                    upper=ctx.args.maxWingLoading,
                    scaler=1.0 / ctx.args.maxWingLoading,
                    cache_linear_solution=True,
                )

            if ctx.args.addPlanformCons and (ctx.args.span or ctx.args.taper or ctx.args.sweep):
                # This constraint should only be applied if the optimiser has control over the wing planform
                # The centre wingbox TE spar location constraint is already handled by DVCon, here we just need to constrain that, at the landing gear spanwise coordinate:
                # - TE chordwise coordinate must be at least it's initial value
                # - TE Spar coordinate must be at least it's initial value
                flightPointProb.model.add_constraint(
                    "PlanformCons.forwardCGX",
                    upper=wingGeometry["wing"]["forwardCGX"],
                )
                flightPointProb.model.add_constraint(
                    "PlanformCons.aftCGX",
                    lower=wingGeometry["wing"]["aftCGX"],
                )
                flightPointProb.model.add_constraint(
                    "PlanformCons.landingGearTEX",
                    lower=wingGeometry["wing"]["landingGearTEX"],
                )
                landingGearChord = abs(wingGeometry["wing"]["landingGearLEX"] - wingGeometry["wing"]["landingGearTEX"])
                flightPointProb.model.add_constraint(
                    "PlanformCons.landingGearTESparX",
                    upper=wingGeometry["wing"]["landingGearLEX"]
                    + wingGeometry["wingbox"]["TESparFrac"] * landingGearChord,
                )

    # --- Buffet constraints ---
    if "buffet" in ctx.localFlightPoint.name.lower():
        flightPointProb.model.add_constraint(
            f"{ctx.localFlightPoint.name}BuffetCon",
            upper=0.0,
            scaler=1 / (0.04 * wingGeometry["wing"]["planformArea"]),
            cache_linear_solution=True,
        )
    # --- Balanced field length constraint ---
    # This should only be applied if the flight point is a cruise point and the balanced field
    if ctx.isCruisePoint and ctx.args.includeBFL:
        flightPointProb.model.add_constraint(
            "takeoff.bfl.distance_continue",
            upper=ctx.args.maxBFL,
            scaler=1.0 / ctx.args.maxBFL,
            cache_linear_solution=True,
        )


# ==============================================================================
# Setup the OpenMDAO models
# ==============================================================================
# Setup the model for this proc's flight point and get the names of its outputs
flightPointProb.setup(force_alloc_complex=isComplex, mode="rev")
tmp = ctx.ptComm.bcast(flightPointProb.model.list_outputs(out_stream=None), root=0)
flightPointProbOutputs = {}
for output in tmp:
    flightPointProbOutputs[output[1]["prom_name"]] = output[1]

# Setup the performance model and get the names of its inputs. Each performance model input takes its value from the
# flight point output with the same name, so check that some flight point model has an output with each name.
performanceProb.setup(force_alloc_complex=True)
performanceInputs = getPromotedInputNames(performanceProb)
checkInputsHaveSources(performanceInputs, flightPointProbOutputs.keys(), ctx.globalComm)

if ctx.ptRank == 0:
    printBanner({"Performance inputs": performanceInputs})

# The coupled adjoint solver's rhs_checking ("auto") only creates its solution cache if, at final_setup, some declared
# responses depend on others. The flight point outputs that feed the performance model (masses, fuel burn, lift) are
# such responses, and caching their adjoints saves most of the cruise adjoint solves, so we declare them as unbounded
# constraints. They can only be declared after setup, because that's when we know which outputs this flight point has.
# These constraints are skipped when building gradFuncs and the pyOptSparse problem.
PERFORMANCE_OUTPUT_ALIAS_PREFIX = "_performanceOutput"
existingResponses = (
    list(flightPointProb.model._responses.keys())
    + list(flightPointProb.model._static_responses.keys())
    + list(flightPointProb.model._design_vars.keys())
    + list(flightPointProb.model._static_design_vars.keys())
)
with allowResponsesAfterSetup(flightPointProb):
    for outputName in performanceInputs:
        if outputName in flightPointProbOutputs and outputName not in existingResponses:
            flightPointProb.model.add_constraint(outputName, alias=f"{PERFORMANCE_OUTPUT_ALIAS_PREFIX}{outputName}")


# Need to call final_setup so that sizes of inputs and outputs are figured out correctly, otherwise calling
# `list_outputs` can fail (see https://github.com/OpenMDAO/OpenMDAO/issues/3560 for details)
flightPointProb.final_setup()
performanceProb.final_setup()

# ==============================================================================
# Potentially set initial DVs from a previous run
# ==============================================================================
if len(ctx.args.initDVs) != 0:
    setValsFromFiles(ctx.args.initDVs, flightPointProb)

om.n2(
    flightPointProb,
    show_browser=False,
    outfile=os.path.join(ctx.localOutputDir, "AeroStruct-N2-Pre-Run.html"),
)
om.n2(
    performanceProb,
    show_browser=False,
    outfile=os.path.join(ctx.outputDir, "Performance-N2-Pre-Run.html"),
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
# gradFuncs, apart from the performance output constraints, which are only added to gradFuncs below if a performance
# constraint depends on them
for con in flightPointProb.model.get_constraints().values():
    isPerformanceOutput = con["alias"] is not None and con["alias"].startswith(PERFORMANCE_OUTPUT_ALIAS_PREFIX)
    if not con["linear"] and not isPerformanceOutput:
        conName = getPromName(flightPointProb.model, con["source"])
        gradFuncs.append(conName)

for obj in flightPointProb.model.get_objectives().values():
    gradFuncs.append(obj["name"])

# Up to now, we'll have caught any gradFuncs from the MPhys models that we need to compute the derivatives of
# because they're directly used as a constraint/objective, now we need to add the functions that we need to compute
# the gradients of because they're inputs to the performance model that computes further objectives/constraints.
relevantFlightPointOutputs = {}
dvsForObjCon = []
performanceInputSources = [getSource(performanceProb.model, inpName) for inpName in performanceInputs]
# Objectives are included too, so that any flight point outputs a performance model objective depends on get gradients
performanceResponses = list(performanceProb.model.get_constraints().values()) + list(
    performanceProb.model.get_objectives().values()
)
for con in performanceResponses:
    # Step 1: find which performance model inputs this constraint depends on, getRelevantInputs takes the candidate
    # inputs explicitly so they don't need to be declared as design variables
    fullConName = con["source"]
    promConName = getPromName(performanceProb.model, fullConName)
    relevantInputs = getRelevantInputs(performanceProb.model, fullConName, performanceInputSources)
    relevantInputs = [getPromName(performanceProb.model, inp) for inp in relevantInputs]
    # Step 2: each performance model input has the same name as the flight point output that feeds it, so outputs is
    # a list of all the variables from all the flight point problems whose derivatives are needed to compute this
    # constraint in the performance model.
    outputs = relevantInputs

    # Next save these outputs for later when we need to figure out which design variables these constraints depend on
    relevantFlightPointOutputs[con["name"]] = list(set(outputs))

    # For each output, we either need to add it to gradFuncs so that we know to compute it's derivatives or, if it is a
    # design variable, we need to tell multipoint to pass that design variable to the objCon function.
    for out in outputs:
        if out in flightPointProbOutputs:
            if out in designVariables:
                dvsForObjCon.append(out)
            elif out not in gradFuncs:
                gradFuncs.append(out)
# We need to call addDVsAsFunctions with the same inputs on all procs because I don't think multipoint does any
# collection across procs
dvsForObjCon = gatherUnion(ctx.globalComm, dvsForObjCon)
ctx.multiPoint.addDVsAsFunctions(dvsForObjCon)

# broadcast gradFuncs to all procs in this set
gradFuncs = ctx.ptComm.bcast(gradFuncs, root=0)

if ctx.ptRank == 0:
    printBanner({"Grad funcs": gradFuncs})

# --- Disp funcs ---
# First add all the grad funcs, and all the functions we need to pass to the performance model
dispFuncs = gradFuncs.copy()
dispFuncs += [f for f in performanceInputs if f in flightPointProbOutputs]

# Now add any additional functions we want to track in the history file even though they're not required for
# optimisation
for output in flightPointProbOutputs:
    # Quantities that exist on all flight points but we only want to save them from a single proc set
    if ctx.ptID == 0:
        patterns = [
            "*mass",  # Component-wise masses
            "wingboxVolume",  # Wingbox/rib bay volumes
            "RibBay-Volume",
            "bayFuelMasses",
        ]
        for pattern in patterns:
            if fnmatch.fnmatch(output, pattern):
                dispFuncs.append(output)
    patterns = [
        "PlanformValues.*",  # WimpressCalc outputs
        "*SectionToC",  # Section t/c values
        "*aero_post.c*",  # aero coefficients
        "*correctedDrag",  # Corrected drag values
        "avgToC",  # Average wing t/c
        "*CLmax_*",  # CLmax values (clean and flapped)
        "*takeoff|v*",  # Takeoff speeds
        "*takeoff.bfl.*",  # Balanced field length speeds and distances
        "*ksFailure",  # TACS failure values
    ]
    for pattern in patterns:
        if fnmatch.fnmatch(output, pattern):
            dispFuncs.append(output)

# Remove duplicate entries from dispFuncs
dispFuncs = list(set(dispFuncs))

# broadcast dispFuncs to all procs in this set
dispFuncs = ctx.ptComm.bcast(dispFuncs, root=0)

if ctx.ptRank == 0:
    printBanner({"Disp funcs": dispFuncs})


# ==============================================================================
# Define functions to be used in the optimisation/analysis
# ==============================================================================
def writeAeroStructSolution():
    scenario = getattr(flightPointProb.model, ctx.localFlightPoint.name)
    scenario.struct_post.write_solution()
    # The TACS counter is only incremented by the write above, so it is in sync across all flight points
    iterNum = scenario.struct_post.eval_funcs.solution_counter - 1
    if ctx.isAeroStruct:
        if not ctx.usingVLM:
            scenario.aero_post.nom_write_solution()
            if ctx.ptID == 0:
                dummyAeroSolver = flightPointProb.model.dummyAeroSolver
                dummyAeroSolver.setAeroProblem(ctx.localFlightPoint)
                dummyAeroSolver(ctx.localFlightPoint, writeSolution=False)
                dummyAeroSolver.writeSolution(baseName="jigshape", number=iterNum)
        geometryComp = flightPointProb.model.geometry
        DVCon = geometryComp.nom_getDVCon()
        wimpressCalc = flightPointProb.model.PlanformValues.wimpressCalc
        if ctx.globalRank == 0:
            DVCon.writeTecplot(os.path.join(ctx.geoOutputDir, f"GeoConstraints_{iterNum:03d}.dat"))
            wimpressCalc.writeTecplot(os.path.join(ctx.geoOutputDir, f"WimpressCalc_{iterNum:03d}.dat"))


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
        setLocalVals(flightPointProb, x)
    fail = False
    try:
        flightPointProb.run_model()
    except om.AnalysisError:
        fail = True

    funcs = {"fail": fail}
    if evalFuncs is not None:
        for func in evalFuncs:
            funcs[func] = flightPointProb.get_val(func)
            if ctx.ptRank == 0:
                print(f"{func} = {funcs[func]}")

    funcRunTime = time.time() - funcStartTime
    if ctx.ptRank == 0:
        with open(ctx.funcTimingFile, "a") as f:
            f.write(f"{funcRunTime:.16e}\n")

    if writeSolution and not ctx.args.noFiles:
        writeAeroStructSolution()

    # Print out some interesting values
    if ctx.ptRank == 0:
        for funcName, funcVal in funcs.items():
            if np.size(funcVal) > 100:
                print(f"{funcName}: max={np.max(funcVal)}, min={np.min(funcVal)}", flush=True)
            else:
                print(f"{funcName}: {funcVal}", flush=True)

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
        setLocalVals(flightPointProb, x)
    if gradFuncs is None:
        gradFuncs = []

    if dispFuncs is None:
        dispFuncs = []

    funcSens = {}
    if len(gradFuncs) != 0:
        openMDAOTotals = flightPointProb.compute_totals(of=gradFuncs, return_format="dict", debug_print=ctx.ptRank == 0)
        for of, sens in openMDAOTotals.items():
            ofName = getPromName(flightPointProb.model, of)
            funcSens[ofName] = {}
            for wrt, val in sens.items():
                wrtName = getPromName(flightPointProb.model, wrt)
                funcSens[ofName][wrtName] = val
    funcRunTime = time.time() - funcStartTime
    if ctx.ptRank == 0:
        with open(ctx.funcSensTimingFile, "a") as f:
            f.write(f"{funcRunTime:.16e}\n")

    # HACK: We need to provide bogus empty derivatives for the functions that are in dispFuncs but not gradFuncs
    # otherwise multipoint will complain
    for func in dispFuncs:
        if func not in funcSens:
            funcSens[func] = {}

    if writeSolution and not ctx.args.noFiles:
        writeAeroStructSolution()

    if ctx.ptRank == 0:
        for ofName, sens in funcSens.items():
            print(f"Gradients of {ofName}:", flush=True)
            for wrtName, val in sens.items():
                if val.shape[0] < 100:
                    print(f"  d{ofName}/d{wrtName} = {val}", flush=True)
                else:
                    print(f"  d{ofName}/d{wrtName} = Too big, not printing", flush=True)

    return funcSens


# This is the function that takes the function values from the aerostructural analyses and computes any remaining
# objective/constraints. In our case this involves running the performance model.
def objCon(funcs, printOK, passThroughFuncs):
    # Multiploint computes the derivatives through this objCOn function using complex step, printOK is False when objCon
    # is being complex-stepped
    performanceProb.set_complex_step_mode(not printOK)

    # Each performance model input takes the value of the flight point output with the same name
    for inputName in performanceInputs:
        performanceProb.set_val(inputName, funcs[inputName])

    performanceProb.run_model()

    outputs = ctx.globalComm.bcast(
        performanceProb.model.list_outputs(return_format="dict", print_arrays=False, out_stream=None), root=0
    )
    for output in outputs.items():
        funcs[output[1]["prom_name"]] = output[1]["val"]

    if ctx.ptRank == 0 and printOK:
        print("\n==================================================", flush=True)
        print("OBJCON Functions:", flush=True)
        pp(funcs)
        print("==================================================\n", flush=True)

    return funcs


ctx.multiPoint.setObjCon(objCon)


# ==============================================================================
# Create wrapped functions to be used by multipoint sparse
# ==============================================================================
def procSetObj(x=None):
    return runAeroStructAnalyses(
        x,
        evalFuncs=gradFuncs if ctx.settings.doEvaluateGradFuncsOnly else dispFuncs,
        writeSolution=ctx.settings.doWriteSolutionInObj,
    )


ctx.multiPoint.addProcSetObjFunc("all", procSetObj)


# I set writeSolution to true on the sensitivity function so that we only write solution files on major iterations
# (this assumes you're using SNOPT's derivative-free line search)
def procSetSens(x=None, funcs=None):
    return computeSens(
        x,
        funcs,
        gradFuncs=gradFuncs,
        dispFuncs=gradFuncs if ctx.settings.doEvaluateGradFuncsOnly else dispFuncs,
        writeSolution=ctx.settings.doWriteSolutionInSens,
    )


ctx.multiPoint.addProcSetSensFunc("all", procSetSens)


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
        name = ctx.localFlightPoint.name
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


if ctx.settings.doInitialRun:
    # ==============================================================================
    # Run the model to initialize everything
    # ==============================================================================
    funcs = runAnalysesRobustly(
        ctx.args.initDVs, evalFuncs=dispFuncs, writeSolution=ctx.settings.doWriteInitialSolution
    )
    # Before proceeding, combine the function values from all flight points and send them to every proc
    funcs = gatherMergedDict(ctx.globalComm, funcs)
    funcs = objCon(funcs, True, None)
    if ctx.ptRank == 0:
        pp(funcs)

# If we have DVs that were supposed to be set after initialisation, we can set those now and re-run the model
if len(ctx.args.postInitDVs) != 0:
    print(f"Proc {ctx.globalRank}: Running again with postInitDVs", flush=True)
    setValsFromFiles(ctx.args.postInitDVs, flightPointProb)
    if ctx.settings.doInitialRun:
        funcs = runAnalysesRobustly(
            ctx.args.postInitDVs,
            evalFuncs=dispFuncs,
            writeSolution=ctx.settings.doWriteInitialSolution,
        )
        funcs = gatherMergedDict(ctx.globalComm, funcs)
        funcs = objCon(funcs, True, None)

if ctx.task == "polar":
    alphaPert = 1.0
    machPert = 0.02
    numPoints = 11
    alphas = ctx.localFlightPoint.alpha + np.linspace(-alphaPert, alphaPert, numPoints)
    machs = ctx.localFlightPoint.mach + np.linspace(-machPert, machPert, numPoints)
    for alphaIndex, alpha in enumerate(alphas):
        for machIndex, mach in enumerate(machs):
            ctx.localFlightPoint.mach = mach
            # We have to set alpha through the dvs otherwise it will be overwritten by the default DV value
            x = {f"dvs.{ctx.localFlightPoint.name}_AOA": alpha}
            if ctx.usingVLM:
                # Setting the Mach number updates the AeroProblem's other flow conditions, VLM reads them from inputs
                for suffix, fpAttr, _ in VLM_FLOW_CONDITIONS:
                    x[f"dvs.{ctx.localFlightPoint.name}-{suffix}"] = getattr(ctx.localFlightPoint, fpAttr)
            funcs = runAeroStructAnalyses(x=x, evalFuncs=dispFuncs, writeSolution=True)
            writeOutputs(
                flightPointProb,
                outputDir=ctx.localOutputDir,
                fileName=f"Mach-{machIndex}-Alpha-{alphaIndex}-Outputs",
            )
if ctx.task == "rawPolar":
    RAW_POLAR_DEFAULT_ALPHA_RANGE = 1.0  # degrees either side of the flight point's alpha
    alphaMin = (
        ctx.localFlightPoint.alpha - RAW_POLAR_DEFAULT_ALPHA_RANGE if ctx.args.alphaMin is None else ctx.args.alphaMin
    )
    alphaMax = (
        ctx.localFlightPoint.alpha + RAW_POLAR_DEFAULT_ALPHA_RANGE if ctx.args.alphaMax is None else ctx.args.alphaMax
    )
    alphas = np.linspace(alphaMin, alphaMax, ctx.args.numAlpha)
    for alphaIndex, alpha in enumerate(alphas):
        # We have to set alpha through the dvs otherwise it will be overwritten by the default DV value
        x = {f"dvs.{ctx.localFlightPoint.name}_AOA": alpha}
        funcs = runAeroStructAnalyses(x=x, evalFuncs=dispFuncs, writeSolution=True)
        writeOutputs(
            flightPointProb,
            outputDir=ctx.localOutputDir,
            fileName=f"Alpha-{alphaIndex}-Outputs",
        )

sol = ""
if ctx.task in ["check", "opt", "trim", "derivCheck"]:
    # ==============================================================================
    # Setup optimization problem
    # ==============================================================================
    optProb = Optimization("Aero-Structural Optimization", ctx.multiPoint.obj)

    # ==============================================================================
    # Define design variables
    # ==============================================================================
    addDesignVarsFromOpenMDAO(optProb, flightPointProb)

    # ==============================================================================
    # Define constraints
    # ==============================================================================
    # Add all constraints from the flight point model, addConstraintFromOpenMDAO will automatically figure out which DVs
    # each constraint depends on using the OpenMDAO model. The performance output constraints are only there for adjoint
    # caching, so they're skipped.
    for con in flightPointProb.model.get_constraints().values():
        isPerformanceOutput = con["alias"] is not None and con["alias"].startswith(PERFORMANCE_OUTPUT_ALIAS_PREFIX)
        if not isPerformanceOutput:
            addConstraintFromOpenMDAO(con, optProb, flightPointProb, wrt="auto")

    # For the constraints coming from the performance model, things are more complicated because the design variables
    # are not in the performance model. We therefore need to do the following for each constraint in the performance
    # model:
    # 1) Figure out which inputs to the performance model it depends on
    # 2) Find the flight point outputs that feed those inputs, which have the same names
    # 3) Figure out which design variables those flight point outputs depend on using the flight point OpenMDAO model
    # This is complicated by the fact that some constraints in the performance model might depend on outputs from
    # multiple flight points
    for con in performanceProb.model.get_constraints().values():
        # We already did steps 1 and 2 earlier when we figured out what should go in gradFuncs, now Step 3:
        # Every proc has the full list of the outputs that are needed, so we can work through that and, if any of the
        # outputs are from this proc's flight point, we can use the flight point OpenMDAO model to figure out which DVs
        # they depend on

        if ctx.ptRank == 0:
            print(f"\n\nPerformance constraint {con['name']} depends on performance inputs:", flush=True)

        relevantDVs = []
        for output in relevantFlightPointOutputs[con["name"]]:
            if ctx.ptRank == 0:
                print(f"- {output}", flush=True)
            if output in flightPointProbOutputs:
                if output in flightPointProb.model.get_design_vars():
                    # If the output is itself a design variable then we can just add it straight away
                    relevantDVs.append(output)
                else:
                    # Otherwise we need to figure out which DVs it depends on
                    relevantDVs += getRelevantInputs(
                        flightPointProb.model,
                        getSource(flightPointProb.model, output),
                        getDesignVarSources(flightPointProb.model),
                    )
        relevantDVs = [getPromName(flightPointProb.model, inp) for inp in relevantDVs]
        # Gather the relevant DVs from all procs, combine them into a single list, then send back to all procs
        relevantDVs = gatherUnion(ctx.globalComm, relevantDVs)
        if ctx.ptRank == 0:
            print("and thus on design variables:", flush=True)
            for dv in relevantDVs:
                print(f"- {dv}", flush=True)

        addConstraintFromOpenMDAO(con, optProb, performanceProb, wrt=relevantDVs)

    # ==============================================================================
    # Define objective
    # ==============================================================================
    # Let's hope we don't have more than one objective defined
    objectives = flightPointProb.model.get_objectives()
    objectives.update(performanceProb.model.get_objectives())

    for obj in objectives.values():
        optProb.addObj(obj["name"], scale=obj["scaler"] if obj["scaler"] is not None else 1.0)

    # Print out some useful info about the optimization problem
    if ctx.ptRank == 0:
        printBanner(
            {"Design variables": optProb.variables, "Constraints": optProb.constraints, "Objectives": objectives}
        )

    optProb.printSparsity(verticalPrint=True)

    ctx.multiPoint.setOptProb(optProb)

    # If any of the objective or constraint functions are also required as inputs for the performance model, we need to
    # explicitly tell this to multipoint so that it passes them to the objCon function
    for func in list(optProb.constraints.keys()) + list(optProb.objectives.keys()):
        if func in performanceInputs:
            if ctx.ptRank == 0:
                print(f"Adding {func} as an objCon input function", flush=True)
            ctx.multiPoint.addConsAsObjConInputs(func)

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
    optHistFilename = os.path.join(ctx.outputDir, "AeroStructOpt.hst")
    optimiserOptions = getOptOptions(
        ctx.args.optimiser,
        ctx.outputDir,
        ctx.args.optIter,
        ctx.args.hessianUpdate,
        ctx.args.initPenalty,
        ctx.args.violLimit,
        ctx.args.stepLimit,
        ctx.args.feasibility,
        ctx.args.optimality,
    )

    # SNOPT settings for the trim task
    TRIM_STEP_LIMIT = 1e30
    TRIM_FEASIBILITY_TOL = 1e-4
    TRIM_LINESEARCH_TOL = 1.0
    TRIM_PENALTY_PARAMETER = 1e3

    restartDict = None
    timeLimit = ctx.args.timeLimit
    if ctx.args.optimiser == "snopt":
        optimiserOptions["Return work arrays"] = True
        if ctx.task == "trim":
            # Ignore the objective, only satisfy the trim constraints
            optimiserOptions["Problem Type"] = "Feasible point"
            # Avoid the step limit
            optimiserOptions["Major step limit"] = TRIM_STEP_LIMIT
            optimiserOptions["Major feasibility tolerance"] = TRIM_FEASIBILITY_TOL
            # Avoid the line search as much as possible
            optimiserOptions["Linesearch tolerance"] = TRIM_LINESEARCH_TOL
            # Only care about satisfying the constraints, not the DV change norm objective SNOPT uses in feasible point
            # mode
            optimiserOptions["Penalty parameter"] = TRIM_PENALTY_PARAMETER
        if timeLimit is not None:
            # Correct the time limit for the time that has elapsed already
            timeLimit = ctx.globalComm.bcast(timeLimit - (time.time() - startTime), root=0)
            optimiserOptions["Time limit"] = int(timeLimit)
        if ctx.args.restartDict is not None:
            with open(ctx.args.restartDict, "rb") as restartFile:
                restartDict = dill.load(restartFile)
                optimiserOptions["Start"] = "Hot"

    optimiser = OPT(optimiserMap[ctx.args.optimiser], options=optimiserOptions)

    # ==============================================================================
    # Run the optimisation
    # ==============================================================================
    if ctx.task == "trim":
        sol = optimiser(
            optProb,
            ctx.multiPoint.sens,
            storeHistory=optHistFilename,
        )
        if ctx.args.optimiser == "snopt":
            # SNOPT also returns its working arrays, which trim doesn't need
            sol = sol[0]

    elif ctx.task == "opt":
        if restartDict is not None:
            sol = optimiser(
                optProb,
                ctx.multiPoint.sens,
                storeHistory=optHistFilename,
                restartDict=restartDict,
                timeLimit=timeLimit,
            )
        else:
            sol = optimiser(
                optProb,
                ctx.multiPoint.sens,
                storeHistory=optHistFilename,
                timeLimit=timeLimit,
            )
        if ctx.args.optimiser == "snopt":
            # SNOPT Returns it's working arrays in a restart dictionary that we should save for future hot starts
            restartDict = sol[-1]
            sol = sol[0]
            if ctx.globalRank == 0:
                with open(os.path.join(ctx.outputDir, "SNOPTRestart.pkl"), "wb") as f:
                    dill.dump(restartDict, f)

    elif ctx.task == "derivCheck":
        # Check derivatives of all objectives and constraints with respect to all design variables
        obj = ctx.multiPoint.obj
        sens = ctx.multiPoint.sens
        origDVs = optProb.getDVs()
        origFuncs, _ = obj(origDVs)
        analyticSens, _ = sens(origDVs, origFuncs)

        # Exclude the large outputs (geometric and panel length constraints) from the derivative check, I'm confident they're accurate
        outputsToTest = [func for func in origFuncs if np.size(origFuncs[func]) < 10]

        # FD settings
        stepSize = 1e-4
        directionalThreshold = 3  # If a DV has a length greater than this we will test the directional derivative only

        fdSens = {}
        for func in outputsToTest:
            fdSens[func] = {}

        dvScales = {}
        for dvName, dvs in optProb.variables.items():
            numDV = len(dvs)
            dvScales[dvName] = np.ones(numDV)
            for ii in range(numDV):
                if dvs[ii].scale is not None:
                    dvScales[dvName][ii] = dvs[ii].scale

            testDirectional = numDV > directionalThreshold

            for func in outputsToTest:
                outputSize = np.size(origFuncs[func])
                inpSize = 1 if testDirectional else numDV
                fdSens[func][dvName] = np.zeros((outputSize, inpSize))

            if testDirectional:
                if ctx.ptRank == 0:
                    print(f"Testing directional derivative w.r.t {dvName} (size {numDV})", flush=True)
                pertDVs = copy.deepcopy(origDVs)
                dvPert = stepSize / dvScales[dvName]
                pertDVs[dvName] += dvPert

                pertFuncs, _ = obj(pertDVs)
                pertDVs[dvName] -= 2 * dvPert
                pertFuncs2, _ = obj(pertDVs)

                for func in outputsToTest:
                    fdSens[func][dvName][:, 0] = (pertFuncs[func] - pertFuncs2[func]) / (2 * stepSize)

            else:
                for ii in range(numDV):
                    if ctx.ptRank == 0:
                        print(f"Testing derivative w.r.t {dvName}[{ii}]", flush=True)
                    pertDVs = copy.deepcopy(origDVs)
                    dvPert = stepSize / dvScales[dvName][ii]
                    pertDVs[dvName][ii] += dvPert

                    pertFuncs, _ = obj(pertDVs)
                    pertDVs[dvName][ii] -= 2 * dvPert
                    pertFuncs2, _ = obj(pertDVs)

                    for func in outputsToTest:
                        fdSens[func][dvName][:, ii] = (pertFuncs[func] - pertFuncs2[func]) / (2 * dvPert)

        derivCheckData = {}
        for func in outputsToTest:
            derivCheckData[func] = {}
            if ctx.ptRank == 0:
                print(f"\n\nDerivative check for function {func}:", flush=True)

            if func not in analyticSens:
                if ctx.ptRank == 0:
                    print(f"  No analytic derivatives found for {func}", flush=True)
                continue
            for dvName in fdSens[func]:
                if ctx.ptRank == 0:
                    print(f"  DV: {dvName}", flush=True)

                fd = fdSens[func][dvName]
                if dvName in analyticSens[func]:
                    analytic = analyticSens[func][dvName]
                else:
                    if ctx.ptRank == 0:
                        print(f"    No analytic derivative found w.r.t {dvName}", flush=True)
                    continue
                derivCheckData[func][dvName] = {
                    "analytic": analytic,
                    "fd": fd,
                }
                if analytic.shape != fd.shape:
                    analyticSensProd = analytic @ (1 / dvScales[dvName])
                    derivCheckData[func][dvName]["analyticProd"] = analyticSensProd
                    absError = np.abs(analyticSensProd - fd.reshape(analyticSensProd.shape))
                    relError = absError / (np.abs(fd.reshape(analyticSensProd.shape)) + 1e-16)
                else:
                    absError = np.abs(analytic - fd)
                    relError = absError / (np.abs(fd) + 1e-16)
                derivCheckData[func][dvName]["absError"] = absError
                derivCheckData[func][dvName]["relError"] = relError

                if ctx.ptRank == 0:
                    for key, val in derivCheckData[func][dvName].items():
                        print(f"    {key:>25}: {val}", flush=True)

        if ctx.globalRank == 0:
            outFileName = os.path.join(ctx.outputDir, "DerivCheck.pkl")
            with open(outFileName, "wb") as f:
                dill.dump(derivCheckData, f, protocol=-1)


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
        if np.size(data) < 10000:
            outputData[output] = data
    except TypeError:
        pass

# Add wingbox tip displacement to the outputs
tipZDisp, tipTwist = getTipDisplacement(flightPointProb, ctx.localFlightPoint.name)
outputData[f"{ctx.localFlightPoint.name}-TipZDisp"] = tipZDisp
outputData[f"{ctx.localFlightPoint.name}-TipTwist"] = tipTwist

# Accumulate the data from all flight points on the root proc
outputData = gatherMergedDict(ctx.globalComm, outputData, doBroadcast=False)

# Add outputs from the performance problem
performanceOutputs = performanceProb.model.list_outputs(return_format="dict", print_arrays=False)

if ctx.globalRank == 0:
    outputData.update(performanceOutputs)

if MPI.COMM_WORLD.rank == 0:
    outFileName = os.path.join(ctx.outputDir, "Outputs.pkl")
    with open(outFileName, "wb") as f:
        dill.dump(outputData, f, protocol=-1)

om.n2(
    flightPointProb,
    show_browser=False,
    outfile=os.path.join(ctx.localOutputDir, "AeroStruct-N2-Post-Run.html"),
)
om.n2(
    performanceProb,
    show_browser=False,
    outfile=os.path.join(ctx.outputDir, "Performance-N2-Post-Run.html"),
)

if ctx.globalRank == 0:
    print(sol)

if ctx.args.profile:
    pr.disable()
    pr.dump_stats(os.path.join(ctx.outputDir, f"{os.path.basename(ctx.args.output)}_{ctx.globalRank:03d}.prof"))
