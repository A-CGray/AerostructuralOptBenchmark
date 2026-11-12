"""Tests that the wing and horizontal tail FFDs, combined into one file, enclose every mesh used with the tail"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import pytest
from cgnsutilities.cgnsutilities import readGrid
from pygeo import DVGeometry

# ==============================================================================
# Extension modules
# ==============================================================================
from utils import getFFDPath, getTailFFDPath, writeMultiBlockFFD

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
AERO_DIR = os.path.join(REPO_ROOT, "aero")
WING_FFD_LEVELS = ["coarse", "med", "fine"]
# The surface meshes that genVolMesh.py coarsens into every level of the overset CFD mesh, which are tracked by git
CFD_SURFACE_MESHES = ["WithTail/wing_surf_S1.cgns", "WithTail/htail_surf_S1.cgns"]
VLM_LEVELS = [1, 2, 3]
# The VLM meshes are ignored by git and written by geometry/generateOML.py
VLM_MESHES = [f"{part}-VLMSurf-L{level}.npy" for part in ["wing", "hTail"] for level in VLM_LEVELS]
# How closely the embedded points must reproduce the mesh coordinates, in metres
EMBEDDING_ATOL = 1e-10


def readMeshPoints(meshFile: str) -> np.ndarray:
    """Read every node of a CGNS surface mesh or a VLM mesh as an (N, 3) array"""
    if meshFile.endswith(".npy"):
        return np.load(meshFile).reshape(-1, 3)
    grid = readGrid(meshFile)
    return np.vstack([block.coords.reshape(-1, 3) for block in grid.blocks])


@pytest.mark.parametrize("meshName", CFD_SURFACE_MESHES + VLM_MESHES)
@pytest.mark.parametrize("ffdLevel", WING_FFD_LEVELS)
def test_combined_ffd_reproduces_mesh_coordinates(ffdLevel, meshName, tmp_path):
    """
    Given the wing FFD of one level and the tail FFD combined into one file by writeMultiBlockFFD,
    when a mesh used in runs with the tail is embedded in a plain DVGeometry built from that file,
    then the undeformed DVGeometry returns the mesh's own coordinates to within EMBEDDING_ATOL, so every node lies
    inside an FFD block. If the tail CFD mesh fails after being remade, check HTAIL_TIP_CAP_REACH in
    geometry/generateOML.py.
    """
    meshFile = os.path.join(AERO_DIR, meshName)
    if meshName in VLM_MESHES and not os.path.isfile(meshFile):
        pytest.skip(f"{meshFile} doesn't exist, run geometry/generateOML.py to create it")
    # cgnsutilities ends the whole test session without a message if the file is missing
    assert os.path.isfile(meshFile), f"{meshFile} is tracked by git and must exist"
    points = readMeshPoints(meshFile)

    combinedFFDFile = str(tmp_path / f"wingAndHTail-ffd-{ffdLevel}.xyz")
    writeMultiBlockFFD([getFFDPath(ffdLevel), getTailFFDPath()], combinedFFDFile)
    DVGeo = DVGeometry(combinedFFDFile)
    DVGeo.addPointSet(points, "mesh")
    embeddedPoints = DVGeo.update("mesh")

    np.testing.assert_allclose(embeddedPoints, points, rtol=0.0, atol=EMBEDDING_ATOL)
