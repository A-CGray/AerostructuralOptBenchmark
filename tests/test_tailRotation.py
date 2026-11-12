"""Tests of the horizontal tail rotation design variable on a DVGeometry made from the combined wing and tail FFD"""

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats, integers
from pygeo import DVGeometry

# ==============================================================================
# Extension modules
# ==============================================================================
from AircraftSpecs.FlightPoint import FlightPoint
from geometry.wingGeometry import wingGeometry
from SETUP.setupDVGeo import (
    TAIL_FFD_VOLUME,
    TAIL_ROTATION_LOWER,
    TAIL_ROTATION_UPPER,
    WING_FFD_VOLUME,
    addTailRotation,
    getTailRotationDVName,
)
from utils import NUM_SPATIAL_DIMS, getFFDPath, getTailFFDPath, writeMultiBlockFFD

# The wing FFD level runs use by default
WING_FFD_LEVEL = "coarse"
# The test flight point. A flight point needs a flow condition, so it is at the cruise Mach number and altitude
TEST_FLIGHT_POINT_NAME = "testPoint"
TEST_LOAD_FACTOR = 1.0
TEST_MACH = 0.77
TEST_ALTITUDE = 10.4e3  # metres
POINT_SET_NAME = "wingAndTailPoints"
MAX_POINTS_PER_BLOCK = 10
COMPLEX_STEP_SIZE = 1e-40
# Tolerances on point coordinates in metres and on their derivatives in metres per degree. The rigid rotation is
# reproduced to about 1e-14 m at -15 degrees, and the derivatives from pyGeo agree with the complex step to round-off
RIGID_ROTATION_ATOL = 1e-10
DERIVATIVE_RTOL = 1e-10
DERIVATIVE_ATOL = 1e-12
# Parametric coordinates of the leading and trailing edges of the tail FFD block, whose first index runs chordwise,
# at the middle of the block's span and height
TAIL_LEADING_EDGE_PARAMS = np.array([[0.0, 0.5, 0.5]])
TAIL_TRAILING_EDGE_PARAMS = np.array([[1.0, 0.5, 0.5]])
# Parametric coordinates of the middle of the wing FFD block
WING_CENTRE_PARAMS = np.array([[0.5, 0.5, 0.5]])
CHORD_INDEX = wingGeometry["chordIndex"]
VERTICAL_INDEX = wingGeometry["verticalIndex"]

tailRotations = floats(TAIL_ROTATION_LOWER, TAIL_ROTATION_UPPER)
# Parametric coordinates (u, v, w) in an FFD block of between 1 and MAX_POINTS_PER_BLOCK points. Points at v = 0 or
# v = 1 in the tail block lie beyond the ends of the hinge line, which runs from y = 0 to the tail's semi-span
blockParams = arrays(
    np.float64,
    integers(1, MAX_POINTS_PER_BLOCK).map(lambda numPoints: (numPoints, NUM_SPATIAL_DIMS)),
    elements=floats(0.0, 1.0),
)


@pytest.fixture(scope="module")
def combinedFFDFile(tmp_path_factory) -> str:
    """Write the wing FFD and the tail FFD into one file, as runs with the tail do"""
    ffdFile = str(tmp_path_factory.mktemp("ffd") / f"wingAndHTail-ffd-{WING_FFD_LEVEL}.xyz")
    writeMultiBlockFFD([getFFDPath(WING_FFD_LEVEL), getTailFFDPath()], ffdFile)
    return ffdFile


def createFlightPoint(tailRotation: float = 0.0) -> FlightPoint:
    """Create a flight point at the cruise Mach number and altitude with the given initial tail rotation"""
    return FlightPoint(
        TEST_FLIGHT_POINT_NAME,
        loadFactor=TEST_LOAD_FACTOR,
        mach=TEST_MACH,
        altitude=TEST_ALTITUDE,
        tailRotation=tailRotation,
    )


def evaluateBlockPoints(DVGeo: DVGeometry, volume: int, params: np.ndarray) -> np.ndarray:
    """Get the points at the given (u, v, w) parametric coordinates of one block of an undeformed FFD, as an (N, 3)
    array"""
    points = DVGeo.FFD.vols[volume].getValue(params[:, 0], params[:, 1], params[:, 2])
    return np.real(points).reshape(-1, NUM_SPATIAL_DIMS)


def createTailRotationDVGeo(
    ffdFile: str, wingParams: np.ndarray, tailParams: np.ndarray, isComplex: bool = False
) -> DVGeometry:
    """Create a DVGeometry from the combined FFD with the tail rotation from addTailRotation, and a point set holding
    the points at the given parametric coordinates of the wing block, then those of the tail block"""
    DVGeo = DVGeometry(ffdFile, isComplex=isComplex)
    addTailRotation(DVGeo, createFlightPoint())
    wingPoints = evaluateBlockPoints(DVGeo, WING_FFD_VOLUME, wingParams)
    tailPoints = evaluateBlockPoints(DVGeo, TAIL_FFD_VOLUME, tailParams)
    DVGeo.addPointSet(np.vstack([wingPoints, tailPoints]), POINT_SET_NAME)
    return DVGeo


def updatePoints(DVGeo: DVGeometry, tailRotation: float) -> np.ndarray:
    """Set the test flight point's tail rotation and return the real part of the deformed points"""
    DVGeo.setDesignVars({getTailRotationDVName(createFlightPoint()): tailRotation})
    return np.real(DVGeo.update(POINT_SET_NAME))


def rotateAboutHingeLine(points: np.ndarray, tailRotation: float) -> np.ndarray:
    """Rotate points about the tail hinge line in wingGeometry, positive nose up, without pyGeo"""
    hingeX = wingGeometry["hTail"]["hingeLineCoords"][0]
    hingeZ = wingGeometry["hTail"]["hingeLineCoords"][1]
    angle = np.deg2rad(tailRotation)
    chordwiseOffset = points[:, CHORD_INDEX] - hingeX
    verticalOffset = points[:, VERTICAL_INDEX] - hingeZ
    rotatedPoints = points.copy()
    rotatedPoints[:, CHORD_INDEX] = hingeX + np.cos(angle) * chordwiseOffset + np.sin(angle) * verticalOffset
    rotatedPoints[:, VERTICAL_INDEX] = hingeZ - np.sin(angle) * chordwiseOffset + np.cos(angle) * verticalOffset
    return rotatedPoints


def computeHingeRotationDerivative(points: np.ndarray, tailRotation: float) -> np.ndarray:
    """Differentiate rotateAboutHingeLine(points, tailRotation) with respect to tailRotation, in metres per degree"""
    hingeX = wingGeometry["hTail"]["hingeLineCoords"][0]
    hingeZ = wingGeometry["hTail"]["hingeLineCoords"][1]
    angle = np.deg2rad(tailRotation)
    dAngleDRotation = np.deg2rad(1.0)
    chordwiseOffset = points[:, CHORD_INDEX] - hingeX
    verticalOffset = points[:, VERTICAL_INDEX] - hingeZ
    derivs = np.zeros_like(points)
    derivs[:, CHORD_INDEX] = dAngleDRotation * (-np.sin(angle) * chordwiseOffset + np.cos(angle) * verticalOffset)
    derivs[:, VERTICAL_INDEX] = dAngleDRotation * (-np.cos(angle) * chordwiseOffset - np.sin(angle) * verticalOffset)
    return derivs


# Each Hypothesis example builds its own DVGeometry, which can take longer than Hypothesis's default 200 ms deadline,
# so the property tests below have no deadline
@settings(deadline=None)
@given(tailRotation=tailRotations)
def test_tail_rotation_dv_starts_at_flight_point_value(combinedFFDFile, tailRotation):
    """Assert that addTailRotation adds one DV, named <flight point name>_tailRotation, whose initial value is the
    flight point's tailRotation"""
    DVGeo = DVGeometry(combinedFFDFile)
    addTailRotation(DVGeo, createFlightPoint(tailRotation))

    designVars = DVGeo.getDesignVars()

    assert list(designVars) == [f"{TEST_FLIGHT_POINT_NAME}_tailRotation"]
    np.testing.assert_array_equal(designVars[f"{TEST_FLIGHT_POINT_NAME}_tailRotation"], [tailRotation])


@settings(deadline=None)
@given(tailRotation=tailRotations, wingParams=blockParams, tailParams=blockParams)
def test_tail_rotation_is_rigid_about_hinge_line(combinedFFDFile, tailRotation, wingParams, tailParams):
    """
    Given a DVGeometry on the combined wing and tail FFD with the tail rotation from addTailRotation, and points
    anywhere in the tail block, including beyond the ends of the hinge line,
    when the tail rotation is set,
    then the tail points move as a rigid rotation about the hinge line in wingGeometry, positive nose up.
    """
    DVGeo = createTailRotationDVGeo(combinedFFDFile, wingParams, tailParams)
    numWingPoints = len(wingParams)
    unrotatedTailPoints = updatePoints(DVGeo, 0.0)[numWingPoints:]

    rotatedTailPoints = updatePoints(DVGeo, tailRotation)[numWingPoints:]

    np.testing.assert_allclose(
        rotatedTailPoints,
        rotateAboutHingeLine(unrotatedTailPoints, tailRotation),
        rtol=0.0,
        atol=RIGID_ROTATION_ATOL,
    )


@settings(deadline=None)
@given(tailRotation=tailRotations, wingParams=blockParams, tailParams=blockParams)
def test_tail_rotation_leaves_wing_points_unmoved(combinedFFDFile, tailRotation, wingParams, tailParams):
    """Assert that setting the tail rotation leaves every point in the wing block exactly where it is at zero tail
    rotation"""
    DVGeo = createTailRotationDVGeo(combinedFFDFile, wingParams, tailParams)
    numWingPoints = len(wingParams)
    unrotatedWingPoints = updatePoints(DVGeo, 0.0)[:numWingPoints]

    wingPoints = updatePoints(DVGeo, tailRotation)[:numWingPoints]

    np.testing.assert_array_equal(wingPoints, unrotatedWingPoints)


def test_positive_tail_rotation_raises_leading_edge(combinedFFDFile):
    """Assert that a positive tail rotation raises the front face of the tail FFD block and lowers its rear face at
    the middle of the block's span, where the front face is ahead of the hinge line"""
    tailParams = np.vstack([TAIL_LEADING_EDGE_PARAMS, TAIL_TRAILING_EDGE_PARAMS])
    DVGeo = createTailRotationDVGeo(combinedFFDFile, WING_CENTRE_PARAMS, tailParams)
    numWingPoints = len(WING_CENTRE_PARAMS)
    unrotatedPoints = updatePoints(DVGeo, 0.0)[numWingPoints:]

    rotatedPoints = updatePoints(DVGeo, TAIL_ROTATION_UPPER)[numWingPoints:]

    verticalMovement = rotatedPoints[:, VERTICAL_INDEX] - unrotatedPoints[:, VERTICAL_INDEX]
    assert verticalMovement[0] > 0.0
    assert verticalMovement[1] < 0.0


@settings(deadline=None)
@given(tailRotation=tailRotations, wingParams=blockParams, tailParams=blockParams)
def test_tail_rotation_sensitivities_match_complex_step(combinedFFDFile, tailRotation, wingParams, tailParams):
    """
    Given a complex DVGeometry on the combined wing and tail FFD with the tail rotation from addTailRotation, and a
    point set with points in both blocks,
    when the derivatives of the points with respect to the tail rotation are computed in reverse mode
    (totalSensitivity) and forward mode (totalSensitivityProd),
    then both match the complex step of the points with respect to the tail rotation.
    """
    DVGeo = createTailRotationDVGeo(combinedFFDFile, wingParams, tailParams, isComplex=True)
    dvName = getTailRotationDVName(createFlightPoint())
    numPoints = len(wingParams) + len(tailParams)
    numCoords = numPoints * NUM_SPATIAL_DIMS
    DVGeo.setDesignVars({dvName: tailRotation})
    # One reverse-mode seed per point coordinate gives the whole Jacobian column
    reverseSeeds = np.eye(numCoords).reshape(numCoords, numPoints, NUM_SPATIAL_DIMS)

    reverseDerivs = DVGeo.totalSensitivity(reverseSeeds, POINT_SET_NAME)[dvName].flatten()
    forwardDerivs = DVGeo.totalSensitivityProd({dvName: np.ones(1)}, POINT_SET_NAME).flatten()
    DVGeo.setDesignVars({dvName: tailRotation + COMPLEX_STEP_SIZE * 1j})
    complexStepDerivs = np.imag(DVGeo.update(POINT_SET_NAME)).flatten() / COMPLEX_STEP_SIZE

    np.testing.assert_allclose(reverseDerivs, complexStepDerivs, rtol=DERIVATIVE_RTOL, atol=DERIVATIVE_ATOL)
    np.testing.assert_allclose(np.real(forwardDerivs), complexStepDerivs, rtol=DERIVATIVE_RTOL, atol=DERIVATIVE_ATOL)


@settings(deadline=None)
@given(tailRotation=tailRotations, wingParams=blockParams, tailParams=blockParams)
def test_tail_rotation_complex_step_matches_rigid_rotation(combinedFFDFile, tailRotation, wingParams, tailParams):
    """
    Given a complex DVGeometry on the combined wing and tail FFD with the tail rotation from addTailRotation, and a
    point set with points in both blocks,
    when the tail rotation is perturbed by a complex step,
    then the tail points' derivatives are those of a rigid rotation about the hinge line and the wing points'
    derivatives are zero, so the rotation function carries the complex perturbation through to the points.
    """
    DVGeo = createTailRotationDVGeo(combinedFFDFile, wingParams, tailParams, isComplex=True)
    dvName = getTailRotationDVName(createFlightPoint())
    numWingPoints = len(wingParams)
    unrotatedTailPoints = updatePoints(DVGeo, 0.0)[numWingPoints:]

    DVGeo.setDesignVars({dvName: tailRotation + COMPLEX_STEP_SIZE * 1j})
    complexStepDerivs = np.imag(DVGeo.update(POINT_SET_NAME)) / COMPLEX_STEP_SIZE

    np.testing.assert_array_equal(complexStepDerivs[:numWingPoints], 0.0)
    np.testing.assert_allclose(
        complexStepDerivs[numWingPoints:],
        computeHingeRotationDerivative(unrotatedTailPoints, tailRotation),
        rtol=DERIVATIVE_RTOL,
        atol=DERIVATIVE_ATOL,
    )
