"""
==============================================================================
Volume mesh extrusion
==============================================================================
@File    :   genVolMesh.py
@Date    :   2024/03/28
@Author  :   Anil Yildirim, slightly modified by Alasdair Christision Gray
@Description : Program to extrude CFD volume meshes using pyHyp
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import argparse
import sys

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from mpi4py import MPI
from pyhyp import pyHyp
from pyhyp.utils import simpleOCart
from cgnsutilities.cgnsutilities import readGrid, combineGrids

# ==============================================================================
# Extension modules
# ==============================================================================
sys.path.append("../")
from ExtrusionSettings import getExtrusionOptions

sys.path.append("../../")
from geometry.wingGeometry import wingGeometry
from AircraftSpecs.STWFlightPoints import standardCruise


def computeFirstCellHeight(rho: float, u: float, L: float, mu: float, yPlus=1.0) -> float:
    """Compute first cell height to achieve a desired y+ value

    Equations from https://www.cadence.com/en_US/home/tools/system-analysis/computational-fluid-dynamics/y-plus.html

    Parameters
    ----------
    rho : float
        Density
    u : float
        Velocity
    L : float
        Characteristic length
    mu : float
        Dynamic viscosity
    yPlus : float, optional
        Desired y+ value, by default 1.0

    Returns
    -------
    float
        First cell height
    """
    re = rho * u * L / mu
    cf = 0.026 / re ** (1 / 7)
    tau = 0.5 * rho * u**2 * cf
    uFric = np.sqrt(tau / rho)
    return yPlus * mu / (rho * uFric)


def computeNumLayers(totalDist: float, s0: float, growthRatio: float) -> int:
    """Compute the number of layers needed to extrude a given distance with a given first cell height and growth ratio

    Parameters
    ----------
    totalDist : float
        Total distance to extrude
    s0 : float
        First cell height
    marchDist : float
        Marching distance for the extrusion
    nConstantStart : int
        Number of constant layers at the start of the extrusion
    nConstantEnd : int
        Number of constant layers at the end of the extrusion

    Returns
    -------
    int
        Total number of layers needed for the extrusion
    """
    return int(np.log(1 - (totalDist * (1 - growthRatio)) / (s0)) / np.log(growthRatio))


parser = argparse.ArgumentParser()
parser.add_argument("--level", default="L3")
args = parser.parse_args()

rank = MPI.COMM_WORLD.rank

# process the arguments and set up the variables
comm = MPI.COMM_WORLD

surfaceNames = ["wing", "hTail"]
extrusionData = {name: {} for name in surfaceNames}

# print all arguments
if comm.rank == 0:
    print("Arguments are:")
    for argname in vars(args):
        print(argname, ":", getattr(args, argname))

if args.level in ["L1", "L2", "L3"]:
    surface_family = "S1"
else:
    surface_family = "S0.7"

# factor for spacings
levelFact = {
    "L3": 0.5,
    "L2": 1.0,
    "L1.4": 1.4,
    "L1": 2.0,
    "L0.7": 2.8,
}
fact = levelFact[args.level]

# Extrude overset meshes to some multiple of the distance between the wing and tail, with the distance for each mesh being proportional to the root chord of the corresponding surface.
distFactor = {
    "L3": 1.0,
    "L2": 1.0,
    "L1.4": 1.0,
    "L1": 1.0,
    "L0.7": 1.0,
}[args.level]
chordOffset = wingGeometry["hTail"]["sectionChordwiseOffset"][0]
verticalOffset = wingGeometry["hTail"]["sectionVerticalOffset"][0]
offset = np.sqrt(chordOffset**2 + verticalOffset**2)

totalChord = 0.0
for surfaceName in surfaceNames:
    rootChord = wingGeometry[surfaceName]["sectionChord"][0]
    extrusionData[surfaceName]["rootChord"] = rootChord
    totalChord += rootChord

for surfaceName in surfaceNames:
    rootChord = extrusionData[surfaceName]["rootChord"]
    extrusionData[surfaceName]["marchDist"] = distFactor * offset * rootChord / totalChord

# levels of coarsening for the surface meshes
coarsen = {
    "L3": 3,
    "L2": 2,
    "L1.4": 2,
    "L1": 1,
    "L0.7": 1,
}[args.level]

growthRatio = {
    "L3": 1.3,
    "L2": 1.2,
    "L1.4": 1.3,
    "L1": 1.1,
    "L0.7": 1.1,
}[args.level]

targetYPlus = 1.0 / fact

nConstantStart = {"L3": 1, "L2": 2, "L1.4": 2, "L1": 3, "L0.7": 3}[args.level]
nConstantEnd = 5

smoothingLevel = {
    "L3": "maxatend",
    "L2": "maxatend",
    "L1.4": "maxatend",
    "L1": "maxatend",
    "L0.7": "maxatend",
}[args.level]

surfaceGrids = []
for surfaceName in ["wing", "hTail"]:
    mesh_name = f"{surfaceName.lower()}_vol_{args.level}"
    rootChord = wingGeometry[surfaceName]["sectionChord"][0]

    s0 = computeFirstCellHeight(
        rho=standardCruise.rho,
        u=standardCruise.V,
        L=rootChord,
        mu=standardCruise.mu,
        yPlus=targetYPlus,
    )

    numLayers = (
        computeNumLayers(
            totalDist=extrusionData[surfaceName]["marchDist"] - nConstantStart * s0,
            s0=s0,
            growthRatio=growthRatio,
        )
        + nConstantStart
        # + nConstantEnd
    )

    options = getExtrusionOptions(smoothingLevel=smoothingLevel)
    options.update(
        {
            # ---------------------------
            #   General options
            # ---------------------------
            "inputFile": f"{surfaceName.lower()}_surf_{surface_family}.cgns",
            "outerFaceBC": "overset",
            "families": surfaceName,
            # ---------------------------
            #   Grid Parameters
            # ---------------------------
            "N": numLayers,
            "s0": s0,
            "marchDist": extrusionData[surfaceName]["marchDist"],
            "nConstantStart": nConstantStart,
            "nConstantEnd": nConstantEnd,
            "coarsen": coarsen,
        }
    )

    hyp = pyHyp(options=options)
    hyp.run()
    hyp.writeCGNS(f"{mesh_name}.cgns")
    surfaceGrids.append(f"{mesh_name}.cgns")

    if rank == 0:
        # finally, print information about the grid
        finalGrid = readGrid(f"{mesh_name}.cgns")
        print("\nGrid info:")
        finalGrid.printInfo()

# Combine wing and tail meshes into single cgns file
wingTailMesh = combineGrids([readGrid(grid) for grid in surfaceGrids], useOldNames=False)

# Create background mesh and combine
backgroundFileName = f"background_vol_{args.level}.cgns"
sym = "y"

# farfield distance. this is adjusted to compensate for the mesh levels
marchDist = 300.0 - extrusionData["wing"]["marchDist"]
mgcycle = 1
OCartGrowthRatio = 1.25

OCartBounds = np.array(((-4.0, 27.0), (0.0, 19.0), (-7.3, 11.0))).T

cellSize = np.array([1.0, 1.5, 1.0]) * 0.65 / fact
nExtra = computeNumLayers(totalDist=marchDist, s0=cellSize[0], growthRatio=OCartGrowthRatio)
OCartxtrusionOptions = {
    "epsE": 1.0,
    "epsI": 6.0,
    "volCoef": 1.0,
    "volBlend": 1e-3,
    "volSmoothIter": 1000,
}
simpleOCart(
    wingTailMesh,
    cellSize,
    marchDist,
    nExtra,
    sym,
    mgcycle,
    backgroundFileName,
    userOptions=OCartxtrusionOptions,
    xBounds=OCartBounds,
)

if rank == 0:
    backgroundGrid = readGrid(backgroundFileName)

    # combine background and overset meshes into single cgns file
    fullGrid = combineGrids([wingTailMesh, backgroundGrid], useOldNames=False)
    fullGrid.writeToCGNS(f"combined_vol_{args.level}.cgns")

    print("\nGrid info:")
    fullGrid.printInfo()
