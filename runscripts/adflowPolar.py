"""
==============================================================================
ADflow baseline polar
==============================================================================
@File    :   adflowPolar.py
@Date    :   2024/06/19
@Author  :   Alasdair Christison Gray
@Description : This script uses ADflow to run an aerodynamic polar on the rigid
baseline geometry as specified in the problem description document. With
--includeTail the rigid horizontal tail is included, at the cruise flight
point's tail rotation or, with --trim, at the rotation that makes the pitching
moment about the cruise CG zero at each angle of attack.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import argparse
import os

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import dill
from adflow import ADFLOW
from idwarp import USMesh
from mpi4py import MPI
from pygeo import DVGeometry

# ==============================================================================
# Extension modules
# ==============================================================================
from SETUP.setupADflow import getADflowOptions, addTailAeroFunctions, WING_FAMILY, HTAIL_FAMILY
from SETUP.setupDVGeo import addTailRotation, getTailRotationDVName, TAIL_ROTATION_LOWER, TAIL_ROTATION_UPPER
from SETUP.setupIDWarp import getIDWarpOptions
from utils import (
    getAeroMeshPath,
    getOutputDir,
    getGeometryData,
    getFlightPointSet,
    getFFDPath,
    getTailFFDPath,
    setMomentReferenceToCG,
    solveBoundedSecant,
    writeMultiBlockFFD,
)

# The wing FFD level whose block goes into the combined wing and tail FFD, the aerostructural runscript's default
FFD_LEVEL = "coarse"
# Trimming with --trim: the first change in tail rotation in degrees, the largest |cmy| accepted as trimmed, and the
# most flow solves at one angle of attack, including both starting solves
TRIM_ROTATION_STEP = 0.5
TRIM_CMY_TOL = 1e-6
TRIM_MAX_SOLVES = 10
# Functions recorded at every angle of attack, and those added with the tail: the pitching moment coefficient about
# the CG, and the wing's and the tail's shares of the lift and drag coefficients
POLAR_FUNC_NAMES = ["cl", "cd"]
TAIL_POLAR_FUNC_NAMES = ["cmy"] + [
    f"{funcName}_{family}" for funcName in POLAR_FUNC_NAMES for family in [WING_FAMILY, HTAIL_FAMILY]
]

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="debug")
parser.add_argument("--level", type=int, default=3, choices=[1, 2, 3])
parser.add_argument("--task", choices=["analysis", "polar"], default="polar")
parser.add_argument(
    "--alphas",
    type=float,
    nargs="+",
    default=None,
    help="Angles of attack of --task polar in degrees, by default 21 angles from 0 to 5",
)
parser.add_argument(
    "--includeTail",
    action="store_true",
    help="Include the rigid horizontal tail, using the overset wing and tail mesh",
)
parser.add_argument(
    "--trim",
    action="store_true",
    help="At each angle of attack, rotate the tail to make the pitching moment about the CG zero (needs --includeTail)",
)
args = parser.parse_args()

if args.trim and not args.includeTail:
    raise ValueError("--trim needs --includeTail, because the tail rotation is what trims the pitching moment")

comm = MPI.COMM_WORLD

# --- Figure out where to put the results ---
outputDir = os.path.join(getOutputDir(), args.output)

if comm.rank == 0:
    os.makedirs(outputDir, exist_ok=True)

meshFile = getAeroMeshPath(args.level, includeTail=args.includeTail)
aeroOptions = getADflowOptions(meshFile=meshFile, outputDir=outputDir, isOverset=args.includeTail)

if args.level == 1:
    aeroOptions["anksecondordswitchtol"] = 1e-4
    aeroOptions["nsubiterturb"] = 10

# Create solver
CFDSolver = ADFLOW(options=aeroOptions)

flightPoint = getFlightPointSet("cruise")[0]
wingGeometry = getGeometryData()
flightPoint.areaRef = wingGeometry["wing"]["planformArea"]
flightPoint.chordRef = wingGeometry["wing"]["meanAerodynamicChord"]
# Take moments about the cruise CG
setMomentReferenceToCG(flightPoint)
flightPoint.addDV("alpha", value=flightPoint.alpha, lower=0, upper=10.0, scale=0.1)

if args.includeTail:
    # The tail rotates through a DVGeometry made from the combined FFD, whose first block is the wing's. Nothing moves
    # the wing block, but it is needed: a DVGeometry embeds every surface node in its nearest block, so with the tail
    # block alone the wing nodes would move with the tail
    ffdFile = os.path.join(outputDir, "Geometry", f"wingAndHTail-ffd-{FFD_LEVEL}.xyz")
    if comm.rank == 0:
        os.makedirs(os.path.dirname(ffdFile), exist_ok=True)
        writeMultiBlockFFD([getFFDPath(FFD_LEVEL), getTailFFDPath()], ffdFile)
    # Every proc reads the combined FFD, so it must be complete first
    comm.Barrier()
    DVGeo = DVGeometry(ffdFile)
    # The tail rotation is the only DV, and it starts at the cruise flight point's tailRotation
    addTailRotation(DVGeo, flightPoint)
    tailRotationDVName = getTailRotationDVName(flightPoint)
    # IDWarp warps the volume mesh around the rotated tail
    mesh = USMesh(options=getIDWarpOptions(meshFile), comm=comm)
    CFDSolver.setMesh(mesh)
    CFDSolver.setDVGeo(DVGeo)
    # This appends to the flight point's evalFuncs, so it must only be called once
    addTailAeroFunctions(CFDSolver, flightPoint)


def solveWithTailRotation(tailRotation: float) -> float:
    """Solve the flow with the tail at the given rotation in degrees, and return cmy about the CG"""
    DVGeo.setDesignVars({tailRotationDVName: tailRotation})
    CFDSolver(flightPoint)
    funcs: dict = {}
    CFDSolver.evalFunctions(flightPoint, funcs, evalFuncs=["cmy"])
    # Every proc must take the same trim decisions, so they all use the root proc's value
    return comm.bcast(funcs[f"{flightPoint.name}_cmy"], root=0)


if args.task == "analysis":
    alphas = [flightPoint.alpha]
elif args.task == "polar":
    if args.alphas is None:
        alphas = list(np.linspace(0, 5, 21))
    else:
        alphas = args.alphas

cl = []
cd = []
# Only recorded with the tail
tailFuncs = {funcName: [] for funcName in TAIL_POLAR_FUNC_NAMES}
tailRotations = []
trimConverged = []
# The tail stays at the cruise flight point's rotation, unless it is trimmed, in which case each angle's trim starts
# from the previous angle's rotation
tailRotation = flightPoint.tailRotation
for ii, alpha in enumerate(alphas):
    if comm.rank == 0:
        print("\n\n")
        print("=" * 80)
        print(f"Running alpha = {alpha}")
        print("=" * 80)

    # --- Solve ---
    flightPoint.alpha = alpha
    if args.trim:
        # The flow solution is left at the last rotation solved, which is the one returned
        tailRotation, cmy, isTrimmed, numSolves = solveBoundedSecant(
            solveWithTailRotation,
            tailRotation,
            TRIM_ROTATION_STEP,
            TAIL_ROTATION_LOWER,
            TAIL_ROTATION_UPPER,
            TRIM_CMY_TOL,
            TRIM_MAX_SOLVES,
        )
        trimConverged.append(isTrimmed)
        if not isTrimmed and comm.rank == 0:
            print(
                f"Warning: the tail could not trim alpha = {alpha:.2f}, cmy = {cmy:.6e} at a tail rotation of "
                f"{tailRotation:.6f} degrees after {numSolves} solves"
            )
    else:
        CFDSolver(flightPoint)

    hist = CFDSolver.getConvergenceHistory()
    with open(os.path.join(outputDir, f"history_{ii:03d}.pkl"), "wb") as file:
        dill.dump(hist, file)

    # --- Compute functions ---
    funcs: dict = {}
    evalFuncs = POLAR_FUNC_NAMES + TAIL_POLAR_FUNC_NAMES if args.includeTail else POLAR_FUNC_NAMES
    CFDSolver.evalFunctions(flightPoint, funcs, evalFuncs=evalFuncs)
    cl.append(funcs[f"{flightPoint.name}_cl"])
    cd.append(funcs[f"{flightPoint.name}_cd"])
    if args.includeTail:
        for funcName in TAIL_POLAR_FUNC_NAMES:
            tailFuncs[funcName].append(funcs[f"{flightPoint.name}_{funcName}"])
        tailRotations.append(tailRotation)

data = {"alpha": alphas, "cl": cl, "cd": cd}
if args.includeTail:
    data.update(tailFuncs)
    data["tailRotation"] = tailRotations
    if args.trim:
        data["trimConverged"] = trimConverged
with open(os.path.join(outputDir, "polarData.pkl"), "wb") as file:
    dill.dump(data, file)

if comm.rank == 0:
    print("\n\nPolar Results:")
    for ii in range(len(alphas)):
        resultLine = f"Alpha = {alphas[ii]:.2f}, CL = {cl[ii]:17.11e}, CD = {cd[ii]:17.11e}"
        if args.includeTail:
            for funcName in TAIL_POLAR_FUNC_NAMES:
                resultLine += f", {funcName} = {tailFuncs[funcName][ii]:17.11e}"
            resultLine += f", tailRotation = {tailRotations[ii]:.6f}"
            if args.trim:
                resultLine += f", trimConverged = {trimConverged[ii]}"
        print(resultLine)
