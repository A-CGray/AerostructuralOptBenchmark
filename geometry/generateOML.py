"""
==============================================================================
OML Geometry generation
==============================================================================
@File    :   generateOML.py
@Date    :   2023/03/29
@Author  :   Alasdair Christison Gray
@Description : This file generates the surface geometry of the wing's OML and a series of fitted FFD's
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import argparse

# ==============================================================================
# External Python modules
# ==============================================================================
from pygeo import pyGeo
from pygeo.geo_utils import createFittedWingFFD, createMidsurfaceMesh
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================
from wingGeometry import wingGeometry
from writeOASMesh import writeOASMesh

spanIndex = wingGeometry["spanIndex"]
chordIndex = wingGeometry["chordIndex"]
verticalIndex = wingGeometry["verticalIndex"]

# How far the rounded tip cap of the CFD tail mesh reaches outboard of the tail's semi-span, in metres. It is measured
# from aero/WithTail/htail_surf_S1.cgns (max y of 5.6218 against a semi-span of 5.6) because the cap exists only in
# the ICEM CFD mesh, not in the OML built here. tests/test_tailFFD.py fails if the mesh is remade with a longer cap
HTAIL_TIP_CAP_REACH = 0.0218
# Spanwise coordinate of the tail FFD's root face in metres. It is not zero because the tail's symmetry-plane nodes
# are at |y| <= 1e-17 and round-off could leave them just outside a face at zero
HTAIL_FFD_ROOT_FACE_Y = -1e-5
# Distance of the tail FFD's tip face outboard of the tip cap in metres
HTAIL_FFD_TIP_CLEARANCE = 1e-3
# Number of spanwise sections of an FFD whose spanwise edges are straight lines
NUM_STRAIGHT_FFD_SECTIONS = 2


def generateOML(partName: str):
    airfoilFiles = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
        for file in wingGeometry[partName]["sectionProfiles"]
    ]

    # Airfoil rotations, this is a bit complicated because the wing we want to make is not in the orientation pyGeo expects
    rot_x = [90.0, 90.0]
    rot_y = wingGeometry[partName]["sectionTwist"]
    rot_z = [0.0, 0.0]

    surface = pyGeo(
        "liftingSurface",
        xsections=airfoilFiles,
        scale=wingGeometry[partName]["sectionChord"],
        X=wingGeometry[partName]["LECoords"],
        rotX=rot_x,
        rotY=rot_y,
        rotZ=rot_z,
        # tip="pinched",
        bluntTe=True,
        squareTeTip=False,
        teHeight=wingGeometry[partName]["teHeight"],
    )

    surface.writeTecplot(f"{partName}.dat")
    surface.writeIGES(f"{partName}.igs")
    surface.writeTin(f"{partName}.tin")
    return surface


def generateVLMMesh(
    surface: pyGeo, LEList: np.ndarray, TEList: np.ndarray, numSpanPanels: int, numChordPanels: int, meshName: str
):
    vlmMesh = createMidsurfaceMesh(
        surface,
        surfFormat="point-vector",
        leList=LEList,
        teList=TEList,
        nSpan=[n + 1 for n in numSpanPanels],
        nChord=numChordPanels + 1,
        liftIndex=wingGeometry["verticalIndex"] + 1,
        chordCosSpacing=0.75,
    )

    # OpenAeroStruct expects the mesh to be order tip to root for some godforsaken reason, so flip the spanwise axis
    vlmMesh = np.flip(vlmMesh, 1)
    np.save(f"../aero/{meshName}.npy", vlmMesh)
    writeOASMesh(vlmMesh, f"{meshName}.dat")


def extendFFDSpanwise(ffdFile: str, rootFaceY: float, tipFaceY: float):
    """Move the root and tip faces of an FFD written by createFittedWingFFD to given spanwise positions

    Each node of the two faces slides along the straight spanwise edge of the FFD it lies on, so the other four faces
    of the box stay in their planes. The file is overwritten.

    Parameters
    ----------
    ffdFile : str
        Plot3d file holding one FFD block with two spanwise sections, the spanwise direction being the block's second
        index
    rootFaceY : float
        Spanwise coordinate to put the root face at
    tipFaceY : float
        Spanwise coordinate to put the tip face at
    """
    with open(ffdFile) as f:
        values = f.read().split()
    numBlocks = int(values[0])
    ffdSize = [int(numNodes) for numNodes in values[1:4]]
    if numBlocks != 1 or ffdSize[1] != NUM_STRAIGHT_FFD_SECTIONS:
        raise ValueError(f"{ffdFile} must hold one block with {NUM_STRAIGHT_FFD_SECTIONS} spanwise sections")
    # Plot3d files list every x coordinate, then every y, then every z, each with the first index running fastest
    ffdCoords = np.array(values[4:], dtype=float).reshape((*ffdSize, 3), order="F")

    rootCoords = ffdCoords[:, 0].copy()
    edgeVectors = ffdCoords[:, -1] - rootCoords
    for sectionIndex, faceY in [(0, rootFaceY), (-1, tipFaceY)]:
        edgeFractions = (faceY - rootCoords[..., spanIndex]) / edgeVectors[..., spanIndex]
        ffdCoords[:, sectionIndex] = rootCoords + edgeFractions[..., np.newaxis] * edgeVectors

    # The same layout as createFittedWingFFD writes, without its trailing spaces
    with open(ffdFile, "w") as f:
        f.write("1\n")
        f.write(f"{ffdSize[0]} {ffdSize[1]} {ffdSize[2]}\n")
        for dim in range(3):
            for kk in range(ffdSize[2]):
                for jj in range(ffdSize[1]):
                    f.write(" ".join("%.15f" % coord for coord in ffdCoords[:, jj, kk, dim]) + "\n")


parser = argparse.ArgumentParser()
parser.add_argument("--ffdType", type=str, choices=["basic", "advanced", "none"], default="advanced")
args = parser.parse_args()

wingSurface = generateOML("wing")
tailSurface = generateOML("hTail")

# ==============================================================================
# Create VLM meshes
# ==============================================================================
wingLEList = wingGeometry["wing"]["LECoords"]
wingTEList = wingGeometry["wing"]["TECoords"]

# Add SOB point to LE and TE lists
SOB_LE = wingLEList[0] + wingGeometry["wingbox"]["SOB"] / wingGeometry["wing"]["semiSpan"] * (
    wingLEList[-1] - wingLEList[0]
)
SOB_TE = wingTEList[0] + wingGeometry["wingbox"]["SOB"] / wingGeometry["wing"]["semiSpan"] * (
    wingTEList[-1] - wingTEList[0]
)
wingLEList = np.vstack([wingLEList[0], SOB_LE, wingLEList[-1]])
wingTEList = np.vstack([wingTEList[0], SOB_TE, wingTEList[-1]])

numChordPanels = [
    4,
    8,
    16,
]
numSpanPanels = [
    [1, 8],
    [2, 16],
    [4, 32],
]
for ii in range(len(numSpanPanels)):
    generateVLMMesh(
        wingSurface,
        wingLEList,
        wingTEList,
        numSpanPanels=numSpanPanels[ii],
        numChordPanels=numChordPanels[ii],
        meshName=f"wing-VLMSurf-L{3 - ii}",
    )
    generateVLMMesh(
        tailSurface,
        wingGeometry["hTail"]["LECoords"],
        wingGeometry["hTail"]["TECoords"],
        numSpanPanels=[numSpanPanels[ii][-1] // 2],
        numChordPanels=numChordPanels[ii],
        meshName=f"hTail-VLMSurf-L{3 - ii}",
    )
# ==============================================================================
# Generate FFDs
# ==============================================================================
numFFDSpan = [6, 9, 12]
numFFDChord = [8, 12, 16]

# --- We need to shift the LE and TE coordinate slightly inside the wing for the fitted FFD creation to work ---
wingLEList = wingGeometry["wing"]["LECoords"]
wingTEList = wingGeometry["wing"]["TECoords"]
wingLEList += 0.01 * (wingTEList - wingLEList)
wingTEList += 0.01 * (wingLEList - wingTEList)
wingLEList[0, spanIndex] += 1e-3
wingTEList[0, spanIndex] += 1e-3

if args.ffdType == "basic":
    # ==============================================================================
    # Generate fitted FFD
    # ==============================================================================
    fileNames = ["wing-ffd-coarse.xyz", "wing-ffd-med.xyz", "wing-ffd-fine.xyz"]
    for nSpan, nChord, file in zip(numFFDSpan, numFFDChord, fileNames, strict=True):
        createFittedWingFFD(
            wingSurface,
            surfFormat="point-vector",
            outFile=file,
            leList=wingLEList,
            teList=wingTEList,
            nSpan=nSpan,
            nChord=nChord,
            absMargins=[0.01, 0.005, 0.05],
            relMargins=[0.015, 0.0035, 0.01],
            liftIndex=wingGeometry["verticalIndex"] + 1,
        )

elif args.ffdType == "advanced":
    # ==============================================================================
    # Fancier FFD
    # ==============================================================================
    # This is a more complex FFD that should allw me to stop the wing root and SOB moving in the spanwise direction
    fileNames = [
        "wing-ffd-advanced-coarse.xyz",
        "wing-ffd-advanced-med.xyz",
        "wing-ffd-advanced-fine.xyz",
    ]

    # The first segment of the FFD will contain 3 sections around the root, purely to enforce that the wing root does
    # not move away from the symmetry plane
    ffdLEList = np.zeros((6, 3))
    ffdTEList = np.zeros((6, 3))
    ffdLEList[0] = wingLEList[0]
    ffdTEList[0] = wingTEList[0]

    ffdLEList[1] = ffdLEList[0]
    ffdTEList[1] = ffdTEList[0]
    ffdLEList[1, spanIndex] = 1e-2
    ffdTEList[1, spanIndex] = 1e-2

    # Next we place 3 FFD sections at the side-of-body location so we can fix that in place
    sobLE = wingLEList[0] + wingGeometry["wingbox"]["SOB"] / wingGeometry["wing"]["semiSpan"] * (
        wingLEList[1] - wingLEList[0]
    )
    sobTE = wingTEList[0] + wingGeometry["wingbox"]["SOB"] / wingGeometry["wing"]["semiSpan"] * (
        wingTEList[1] - wingTEList[0]
    )
    ffdLEList[2] = sobLE
    ffdTEList[2] = sobTE

    ffdLEList[3] = ffdLEList[2] + 2.5e-4 * (ffdLEList[2] - ffdLEList[1])
    ffdTEList[3] = ffdTEList[2] + 2.5e-4 * (ffdTEList[2] - ffdTEList[1])

    ffdLEList[4] = ffdLEList[2] + 5e-4 * (ffdLEList[2] - ffdLEList[1])
    ffdTEList[4] = ffdTEList[2] + 5e-4 * (ffdTEList[2] - ffdTEList[1])
    # ffdLEList[3, spanIndex] += 1e-3
    # ffdTEList[3, spanIndex] += 1e-3

    # The final FFD section is placed at the wing tip
    ffdLEList[5] = wingLEList[-1]
    ffdTEList[5] = wingTEList[-1]

    for nSpan, nChord, file in zip(numFFDSpan, numFFDChord, fileNames, strict=True):
        createFittedWingFFD(
            wingSurface,
            surfFormat="point-vector",
            outFile=file,
            leList=ffdLEList,
            teList=ffdTEList,
            nSpan=[2, 1, 1, 1, nSpan],
            nChord=nChord,
            absMargins=[0.01, 0.005, 0.05],
            relMargins=[0.015, 0.0035, 0.01],
            liftIndex=wingGeometry["verticalIndex"] + 1,
        )

# ==============================================================================
# Create basic tail FFD
# ==============================================================================
tailFFDName = "hTail-ffd.xyz"
tailFFDLEList = wingGeometry["hTail"]["LECoords"]
tailFFDLEList[:, chordIndex] += 1e-4 * wingGeometry["hTail"]["sectionChord"]
tailFFDLEList[0, spanIndex] += 1e-4
tailFFDLEList[-1, spanIndex] -= 1e-4

tailFFDTEList = wingGeometry["hTail"]["TECoords"]
tailFFDTEList[:, chordIndex] -= 1e-4 * wingGeometry["hTail"]["sectionChord"]
tailFFDTEList[0, spanIndex] += 1e-4
tailFFDTEList[-1, spanIndex] -= 1e-4

# There is no spanwise margin, because createFittedWingFFD adds the same margin at the root and the tip and the tail
# needs different ones. extendFFDSpanwise places the root and tip faces afterwards
absMargins = np.zeros((3,))
relMargins = np.zeros((3,))
absMargins[verticalIndex] = 0.2
relMargins[chordIndex] = 0.01

createFittedWingFFD(
    tailSurface,
    surfFormat="point-vector",
    outFile=tailFFDName,
    leList=tailFFDLEList,
    teList=tailFFDTEList,
    nSpan=2,
    nChord=2,
    absMargins=absMargins,
    relMargins=relMargins,
    liftIndex=wingGeometry["verticalIndex"] + 1,
)

# Extend each of the box's four spanwise edges along its own line, so the root face lies just inboard of the symmetry
# plane and the tip face just outboard of the CFD mesh's tip cap
extendFFDSpanwise(
    tailFFDName,
    rootFaceY=HTAIL_FFD_ROOT_FACE_Y,
    tipFaceY=wingGeometry["hTail"]["semiSpan"] + HTAIL_TIP_CAP_REACH + HTAIL_FFD_TIP_CLEARANCE,
)
