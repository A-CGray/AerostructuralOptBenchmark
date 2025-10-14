"""
==============================================================================

==============================================================================
@File    :   generateFuelMassRBEs.py
@Date    :   2025/09/09
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import argparse

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from tacs import pyTACS
import sympy as sp

# ==============================================================================
# Extension modules
# ==============================================================================


def computeCentroidFraction(A1, A2):
    """Given the areas of two consecutive ribs, compute the approximate location of the centroid of the rib bay between them as a fraction of the distance from rib 1 to rib 2.

    This calculation assumes that the area varies quadraticly between the two ribs.

    Parameters
    ----------
    A1 : float
        Area of the first rib.
    A2 : float
        Area of the second rib.
    """
    x = sp.symbols("x", real=True)
    A = (sp.sqrt(A1) + (sp.sqrt(A2) - sp.sqrt(A1)) * x) ** 2
    centroid = sp.integrate(x * A, (x, 0, 1)) / sp.integrate(A, (x, 0, 1))
    return float(centroid)


parser = argparse.ArgumentParser()
parser.add_argument("--level", type=int, default=4, help="Mesh level", choices=list(range(1, 5)))

args = parser.parse_args()

origMeshFileName = f"wingbox-L{args.level}-Order2.bdf"
newMeshFileName = f"wingbox-L{args.level}-Order2-wRBEs.bdf"

# Open the mesh with TACS so we can access it through TACS and pyNastran
FEAAssembler = pyTACS(origMeshFileName)
meshLoader = FEAAssembler.meshLoader
bdf = meshLoader.getBDFInfo()
bdf.cross_reference()
compDescripts = meshLoader.getComponentDescripts()

structGroups = ["U_SKIN", "L_SKIN", "RIB", "SPAR.00", "SPAR.01"]

# Get the compIDs for each group and sort them in from root to tip
#
componentData = {}

for groupName in structGroups:
    componentData[groupName] = []
    groupCompDescripts = [desc for desc in compDescripts if groupName in desc]
    groupCompDescripts.sort()
    for compName in groupCompDescripts:
        compID = FEAAssembler.selectCompIDs(compName)[0][0]
        compNodeInds = meshLoader.getGlobalNodeIDsForComps([compID], nastranOrdering=True)
        compElemInds = meshLoader.getGlobalElementIDsForComps([compID], nastranOrdering=True)
        compCentre = np.zeros(3)
        compArea = 0.0
        for elemInd in compElemInds:
            elem = bdf.elements[elemInd]
            area, centroid = elem.AreaCentroid()
            compArea += area
            compCentre += area * centroid
        compCentre /= compArea

        compNodeCoords = meshLoader.getBDFNodes(compNodeInds, nastranOrdering=True)
        componentData[groupName].append(
            {
                "compID": compID,
                "compDescript": compName,
                "nodeInds": compNodeInds,
                "centre": compCentre,
                "area": compArea,
            }
        )

# Now we have all the component data we need, for each rib bay we want to do the following:
# 1. Compute the centroid of the bay
# 2. Find the nodes at the intersection of the two ribs and two spars bounding the bay
# 3. Create a node at the centroid of the bay
# 4. Create and RBE3 element connecting the centroid node to the intersection nodes

numRibs = len(componentData["RIB"])
numBays = numRibs - 1

# There are 69963 nodes in the finest mesh, so start the new nodes from 100000
massNodeInd = 100000
conm2Ind = 200000
rbeInd = 300000

for bayInd in range(numBays):
    startRibData = componentData["RIB"][bayInd]
    endRibData = componentData["RIB"][bayInd + 1]
    startSparData = componentData["SPAR.00"][bayInd]
    endSparData = componentData["SPAR.01"][bayInd]

    centroidFraction = computeCentroidFraction(startRibData["area"], endRibData["area"])
    bayCentroid = startRibData["centre"] + centroidFraction * (endRibData["centre"] - startRibData["centre"])

    # Create the new node
    bdf.add_grid(massNodeInd, bayCentroid, comment=f"Centroid node for rib bay {bayInd}")

    intersectionNodeInds = []
    for rib in [startRibData, endRibData]:
        for spar in [startSparData, endSparData]:
            commonNodeInds = list(set(rib["nodeInds"]).intersection(set(spar["nodeInds"])))
            if len(commonNodeInds) == 0:
                raise ValueError(f"No common nodes found between {rib['compDescript']} and {spar['compDescript']}")
            intersectionNodeInds += commonNodeInds
    # Create the CONM2 mass at the new node
    bdf.add_conm2(conm2Ind, massNodeInd, 1.0, comment=f"Fuel mass for rib bay {bayInd}")
    # Connect the centroid node to the intersection nodes with an RBE3
    elemInd = elemInd + bayInd
    bdf.add_rbe3(
        eid=rbeInd,
        refgrid=massNodeInd,
        refc="123456",
        weights=[1.0],
        comps=["123"],
        Gijs=[intersectionNodeInds],
        comment=f"RBE3 for rib bay {bayInd}",
    )

    massNodeInd += 1
    conm2Ind += 1
    rbeInd += 1

# Write out the new mesh
with open(newMeshFileName, "w") as f:
    bdf.write_bdf(f)

# Go through file and remove MAT1 and PBAR cards
with open(newMeshFileName, "r") as f:
    lines = f.readlines()
with open(newMeshFileName, "w") as f:
    skipNext = False
    for line in lines:
        if not (line.startswith("MAT1") or line.startswith("PBAR")):
            f.write(line)
