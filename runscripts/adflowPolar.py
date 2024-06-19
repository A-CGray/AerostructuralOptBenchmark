"""
==============================================================================
ADflow baseline polar
==============================================================================
@File    :   adflowPolar.py
@Date    :   2024/06/19
@Author  :   Alasdair Christison Gray
@Description : This script uses ADflow to run an aerodynamic polar on the rigid
baseline geometry as specified in the problem description document
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
from mpi4py import MPI

# ==============================================================================
# Extension modules
# ==============================================================================
from SETUP.setupADflow import getADflowOptions
from utils import getAeroMeshPath, getOutputDir, getGeometryData, getFlightPointSet

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="debug")
parser.add_argument("--level", type=int, default=3, choices=[1, 2, 3])
parser.add_argument("--task", choices=["analysis", "polar"], default="polar")
args = parser.parse_args()

comm = MPI.COMM_WORLD

# --- Figure out where to put the results ---
outputDir = os.path.join(getOutputDir(), args.output)

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
    alphas = list(np.linspace(0, 5, 21))

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
        dill.dump(hist, file)

    # --- Compute functions ---
    funcs: dict = {}
    CFDSolver.evalFunctions(flightPoint, funcs, evalFuncs=["cl", "cd"])
    cl.append(funcs[f"{flightPoint.name}_cl"])
    cd.append(funcs[f"{flightPoint.name}_cd"])

data = {"alpha": alphas, "cl": cl, "cd": cd}
with open(os.path.join(outputDir, "polarData.pkl"), "wb") as file:
    dill.dump(data, file)

if comm.rank == 0:
    print("\n\nPolar Results:")
    for ii in range(len(alphas)):
        print(f"Alpha = {alphas[ii]:.2f}, CL = {cl[ii]:17.11e}, CD = {cd[ii]:17.11e}")
