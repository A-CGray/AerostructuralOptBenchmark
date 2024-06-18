import numpy as np
import argparse
import os
from adflow import ADFLOW
from baseclasses import AeroProblem
from mpi4py import MPI
from pprint import pprint
import pickle
from SETUP.setupADflow import getADflowOptions
from utils import getAeroMeshPath, whereAmIRunning, getGeometryData, getFlightPointSet

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="debug")
parser.add_argument("--level", type=int, default=3, choices=[1, 2, 3])
parser.add_argument("--task", choices=["analysis", "polar"], default="polar")
args = parser.parse_args()

comm = MPI.COMM_WORLD

# --- Figure out where to put the results ---
machineName = whereAmIRunning()
parentDirs = {
    "albert": "/home/ali/BigBoi/ali/AerostructuralOptBenchmark",
    "greatlakes": "/scratch/jrram_root/jrram1/alachris/AerostructuralOptBenchmark",
    "stampede": "/work2/07488/acgray/stampede2/AerostructuralOptBenchmark",
}
if machineName in parentDirs:
    OUTPUT_PARENT_DIR = parentDirs[machineName]
else:
    OUTPUT_PARENT_DIR = "Output"
outputDir = os.path.join(OUTPUT_PARENT_DIR, args.output)

if comm.rank == 0:
    os.makedirs(outputDir, exist_ok=True)

meshFile = getAeroMeshPath(args.level)
aeroOptions = getADflowOptions(meshFile=meshFile, outputDir=outputDir)

# Create solver
CFDSolver = ADFLOW(options=aeroOptions)

flightPoint = getFlightPointSet("cruise")[0]
wingGeometry = getGeometryData()
flightPoint.areaRef = wingGeometry["wing"]["planformArea"]
flightPoint.chordRef = wingGeometry["wing"]["meanAerodynamicChord"]
flightPoint.addDV("alpha", value=flightPoint.alpha, lower=0, upper=10.0, scale=0.1)

if args.task == "analysis":
    alphas = [flightPoint.alpha]
elif args.task == "polar":
    alphas = list(np.linspace(0, 5, 6))

cl = []
cd = []
for ii, alpha in enumerate(alphas):
    if comm.rank == 0:
        print("\n\n")
        print("=" * 80)
        print(f"Running alpha = {alpha}")
        print("=" * 80)

    # --- Solve ---
    flightPoint.alpha = alpha
    CFDSolver(flightPoint)

    hist = CFDSolver.getConvergenceHistory()
    with open(os.path.join(outputDir, f"history_{ii:03d}.pkl"), "wb") as file:
        pickle.dump(hist, file)

    # --- Compute functions ---
    funcs = {}
    CFDSolver.evalFunctions(flightPoint, funcs, evalFuncs=["cl", "cd"])
    cl.append(funcs[f"{flightPoint.name}_cl"])
    cd.append(funcs[f"{flightPoint.name}_cd"])

if comm.rank == 0:
    print("\n\nPolar Results:")
    for ii in range(len(alphas)):
        print(f"Alpha = {alphas[ii]:.2f}, CL = {cl[ii]:17.11e}, CD = {cd[ii]:17.11e}")
