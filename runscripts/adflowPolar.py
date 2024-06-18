import numpy as np
import argparse
import os
from adflow import ADFLOW
from baseclasses import AeroProblem
from mpi4py import MPI
from pprint import pprint
import pickle
from .SETUP.setupADflow import getADflowOptions

parser = argparse.ArgumentParser()
parser.add_argument("--output", type=str, default="output")
parser.add_argument("--level", type=int, default=2, choices=list(range(4)))
parser.add_argument("--task", choices=["analysis", "polar"], default="analysis")
args = parser.parse_args()

comm = MPI.COMM_WORLD

if not os.path.exists(args.output):
    if comm.rank == 0:
        os.makedirs(args.output, exist_ok=True)

aeroOptions = getADflowOptions(meshFile=, outputDir=args.output)

# Create solver
CFDSolver = ADFLOW(options=aeroOptions)

ap = AeroProblem(name="wing", mach=0.8, altitude=10000, alpha=2.5, areaRef=45.5, chordRef=3.25, evalFuncs=["cl", "cd"])
ap.addDV("alpha", value=2.5, lower=0, upper=10.0, scale=0.1)

# --- Solve ---
CFDSolver(ap)

hist = CFDSolver.getConvergenceHistory()
with open(os.path.join(args.output, "history.pkl"), "wb") as file:
    pickle.dump(hist, file)

# --- Compute functions ---
funcs = {}
CFDSolver.evalFunctions(ap, funcs)

# --- Compute derivatives ---
funcSens = {}
CFDSolver.evalFunctionsSens(ap, funcSens)

if comm.rank == 0:
    print("Functions:")
    pprint(funcs)

    print("\nSensitivities:")
    pprint(funcSens)
