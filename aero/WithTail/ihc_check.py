import sys
import argparse
from adflow import ADFLOW

sys.path.append("../../")
from AircraftSpecs.STWFlightPoints import standardCruise
from geometry.wingGeometry import wingGeometry

# ======================================================================
#         Input Information
# ======================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--level", default="L3")
parser.add_argument("--solve", action="store_true")
args = parser.parse_args()

# File name of the mesh
gridFile = f"combined_vol_{args.level}.cgns"

# Common aerodynamic problem description and design variables
ap = standardCruise
standardCruise.areaRef = 45.5
standardCruise.chordRef = 3.25

# dictionary with name of the zone as a key and a factor to multiply it with.
oversetpriority = {}

aeroOptions = {
    # Common Parameters
    "gridFile": gridFile,
    "outputDirectory": "./",
    "mgcycle": "sg",
    "monitorvariables": [
        "cpu",
        "resrho",
        "resmom",
        "resturb",
        "cl",
        "cd",
        "yplus",
        "sepsensor",
    ],
    "surfaceVariables": [
        "cp",
        "vx",
        "vy",
        "vz",
        "mach",
        "yplus",
        "cf",
        "cfx",
        "cfy",
        "cfz",
        "sepsensor",
        "sepsensorks",
        "sepsensorksarea",
        "blank",
    ],
    "volumevariables": [
        "blank",
    ],
    "isosurface": {"vx": -0.001, "shock": 1.0},
    "writeTecplotSurfaceSolution": True,
    "writesurfacesolution": True,
    # Physics Parameters
    "equationType": "RANS",
    "useQCR": True,
    "liftindex": wingGeometry["verticalIndex"] + 1,  # z is the lift direction
    # Solver Parameters
    "smoother": "DADI",
    "CFL": 1.5,
    "CFLCoarse": 1.25,
    "MGCycle": "sg",
    "MGStartLevel": -1,
    "nCyclesCoarse": 250,
    "rkReset": True,  # args.task == "derivCheck",
    "nRKReset": 20,
    "infchangecorrection": True,
    "infChangeCorrectionType": "rotate",
    "useBlockettes": True,
    # Separation sensor options
    # Old sepsensor formulation
    "sepSensorOffset": -0.1,
    # New sepsenson formulation
    "computeSepSensorKs": True,
    "sepSensorKsRho": 1000.0,
    "sepSensorKsPhi": 60.0,
    # Debugging/overset parameters
    "debugzipper": True,
    "usezippermesh": False,
    "nrefine": 10,  # number of times to run IHC cycle
    "nearwalldist": 0.1,
    "oversetpriority": oversetpriority,
}

# Create solver
CFDSolver = ADFLOW(options=aeroOptions, debug=True)

# Uncoment this if just want to check flooding
CFDSolver.setAeroProblem(ap)

name = ".".join(gridFile.split(".")[0:-1])
CFDSolver.writeVolumeSolutionFile(name + "_IHC.cgns", writeGrid=True)

if args.solve:
    CFDSolver(ap)
