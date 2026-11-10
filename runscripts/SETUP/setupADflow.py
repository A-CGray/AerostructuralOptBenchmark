import sys
import os
from typing import Optional

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../geometry"))
from wingGeometry import wingGeometry  # noqa: E402

# ADflow lower-cases the family names of the mesh file and looks them up case-sensitively, so these are lower case
# The only wall family of the wing-only meshes
WING_ONLY_WALL_FAMILY = "wall"
# The wall families of the overset wing and horizontal tail meshes, aero/WithTail/combined_vol_L*.cgns, whose CGNS
# families are wing and hTail
WING_FAMILY = "wing"
HTAIL_FAMILY = "htail"


def getADflowOptions(meshFile: str, outputDir: str, aerostructural: Optional[bool] = False, isOverset: bool = False):
    """Get the ADflow options shared by every script

    Parameters
    ----------
    meshFile : str
        Path to the volume mesh
    outputDir : str
        Directory ADflow writes its solution files to
    aerostructural : bool, optional
        Whether ADflow is part of an aerostructural analysis, which needs looser convergence tolerances, by default
        False
    isOverset : bool, optional
        Whether the mesh is an overset mesh, by default False

    Returns
    -------
    dict
        ADflow options
    """
    aero_options = {
        # I/O Parameters
        "gridFile": meshFile,
        "outputDirectory": outputDir,
        "monitorvariables": [
            "cpu",
            "resrho",
            "resmom",
            "resturb",
            "cl",
            "cd",
            "cmy",
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
        ],
        "isosurface": {"vx": -0.001, "shock": 1.0},
        "writeTecplotSurfaceSolution": True,
        "writevolumesolution": False,
        "writesurfacesolution": True,
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
        "nRKReset": 5 if aerostructural else 20,
        "infchangecorrection": True,
        "infChangeCorrectionType": "rotate",
        "useBlockettes": True,
        # ANK Solver Parameters
        "useANKSolver": True,
        # "nSubiterTurb": 5,
        "ANKCFLCutback": 0.1,  # be more aggressive in reducing the CFL when necessary
        "ankswitchtol": 1.0,
        "anksecondordswitchtol": 1e-4,
        # "ankcoupledswitchtol": 1e-10,
        # NK Solver Parameters
        "useNKSolver": False,
        "nkswitchtol": 1e-10,
        "NKJacobianLag": 5,
        "nkinnerpreconits": 2,
        "NKASMOverlap": 2,
        # Termination Criteria
        "L2Convergence": 1e-14,
        "L2ConvergenceCoarse": 1e-2,
        "L2ConvergenceRel": 1e-3 if aerostructural else 1e-14,
        "nCycles": 1000 if aerostructural else 20000,
        # Adjoint parameters
        "adjointMaxIter": 500,
        "adjointSubspaceSize": 500,
        "adjointL2ConvergenceRel": (
            1e-1 if aerostructural else 1e-14
        ),  # Only converge the ADflow adjoint 1 order of magnitudefor aerostructural models because we are only using the ADflow adjoint solver as a preconditioner for the coupled adjoint GMRES solver
        "adjointL2Convergence": 1e-14,
        "restartAdjoint": True,
        "ADPC": False,
        # The assembled dRdW matrix takes some time to build but then each adjoint solve is faster. On overset meshes
        # it gives wrong gradients with no warning, so they have to use the matrix-free form
        "useMatrixFreedrdw": isOverset,
        "ILUFill": 2,  # 2 gave faster coupled adjoint solves than 3
        # force integration
        "forcesAsTractions": False,  # Because we're using MELD not RLT
        # Overset parameters, which have no effect on non-overset meshes
        "useZipperMesh": False,  # No wall surfaces overlap
        "oversetUpdateMode": "frozen",  # IDWarp warps the whole mesh together, so the connectivity stays valid
        # Separation sensor options
        # Old sepsensor formulation
        "sepSensorOffset": -0.1,
        # New sepsenson formulation
        "computeSepSensorKs": True,
        "sepSensorKsRho": 1000.0,
        "sepSensorKsPhi": 60.0,
    }
    return aero_options
