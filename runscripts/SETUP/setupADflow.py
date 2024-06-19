import sys
import os
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../geometry"))
from wingGeometry import wingGeometry  # noqa: E402


def getADflowOptions(meshFile: str, outputDir: str, aerostructural: Optional[bool] = False):
    aero_options = {
        # I/O Parameters
        "gridFile": meshFile,
        "outputDirectory": outputDir,
        "monitorvariables": ["cpu", "resrho", "resmom", "resturb", "cl", "cd", "yplus", "sepsensor"],
        "surfaceVariables": ["cp", "vx", "vy", "vz", "mach", "yplus", "cf", "cfx"],
        "isosurface": {"vx": -0.001, "shock": 1.0},
        "writeTecplotSurfaceSolution": True,
        "writevolumesolution": False,
        "writesurfacesolution": False,
        # Printing Parameters
        "setMonitor": True,
        "printTiming": False,
        # Load balancing parameters
        "loadImbalance": 0.1,
        "loadBalanceIter": 10,
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
        "nRKReset": 5,
        "infchangecorrection": True,
        "infChangeCorrectionType": "rotate",
        "useBlockettes": True,
        # ANK Solver Parameters
        "useANKSolver": True,
        "ANKCFLCutback": 0.25,
        "ankswitchtol": 1.0,
        "anksecondordswitchtol": 1e-5,
        "ankcoupledswitchtol": 1e-10,
        # NK Solver Parameters
        # "useNKSolver": True,
        "nkswitchtol": 1e-8,
        "NKJacobianLag": 5,
        "nkinnerpreconits": 2,
        "NKASMOverlap": 2,
        # Termination Criteria
        "L2Convergence": 1e-14,
        "L2ConvergenceCoarse": 1e-2,
        "L2ConvergenceRel": 1e-2 if aerostructural else 1e-14,
        "nCycles": 1000 if aerostructural else 10000,
        # Adjoint parameters
        "adjointMaxIter": 500,
        "adjointSubspaceSize": 500,
        "adjointL2ConvergenceRel": (
            1e-1 if aerostructural else 1e-14
        ),  # Only converge the ADflow adjoint 1 order of magnitudefor aerostructural models because we are only using the ADflow adjoint solver as a preconditioner for the coupled adjoint GMRES solver
        "adjointL2Convergence": 1e-14,
        "restartAdjoint": True,
        "ADPC": False,
        # force integration
        "forcesAsTractions": False,  # Because we're using MELD not RLT
    }
    return aero_options
