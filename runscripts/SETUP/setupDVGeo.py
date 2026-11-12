"""
==============================================================================
Geometry parameterisation and constraint setup
==============================================================================
@File    :   steupDVGeo.py
@Date    :   2023/03/31
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import sys
import os

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from pyspline import Curve

# ==============================================================================
# Extension modules
# ==============================================================================
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../geometry"))
from wingGeometry import wingGeometry  # noqa: E402

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
from AircraftSpecs.FlightPoint import FlightPoint  # noqa: E402

# Index of the wing's block in the FFD file. The wing must be the first block: its reference axis, getLocalIndex and
# the indexing of the local shape DVs rely on it. A second block, for the horizontal tail, may follow it
WING_FFD_VOLUME = 0
# Index of the horizontal tail's block in the FFD file, which follows the wing's
TAIL_FFD_VOLUME = 1
# Name of the horizontal tail's reference axis, its hinge line, matching the tail's wall family in the CFD mesh
TAIL_AXIS_NAME = "hTail"
# Bounds on the tail rotation in degrees, positive nose up: a conservative widening of the MD-80's limits of 12.2
# degrees nose down and 2.1 degrees nose up
TAIL_ROTATION_LOWER = -15.0
TAIL_ROTATION_UPPER = 5.0
# The tail rotation has a fixed scaling, like the angle of attack, rather than one multiplied by --geoScalingFactor
TAIL_ROTATION_SCALER = 0.1


def c_atan2(x, y):
    a = x.real
    b = x.imag
    c = y.real
    d = y.imag
    return complex(np.arctan2(a, c), (c * b - a * d) / (a**2 + c**2))


def getTailRotationDVName(flightPoint: FlightPoint) -> str:
    """Get the name of a flight point's tail rotation DV, shared by the DVGeo global DV and the OpenMDAO design variable

    Parameters
    ----------
    flightPoint : FlightPoint
        The flight point the tail rotation belongs to

    Returns
    -------
    str
        The DV name, ``<flight point name>_tailRotation``
    """
    return f"{flightPoint.name}_tailRotation"


def createTailHingeCurve() -> Curve:
    """Create the horizontal tail's hinge line, the reference axis the tail rotates about

    The hinge line is parallel to y and runs from the symmetry plane to the tail tip through
    ``wingGeometry["hTail"]["hingeLineCoords"]``.

    Returns
    -------
    Curve
        The hinge line as a straight, linear pySpline curve
    """
    # hingeLineCoords holds the hinge line's x and z coordinates
    hingeX = wingGeometry["hTail"]["hingeLineCoords"][0]
    hingeZ = wingGeometry["hTail"]["hingeLineCoords"][1]
    tailSemiSpan = wingGeometry["hTail"]["semiSpan"]
    return Curve(x=[hingeX, hingeX], y=[0.0, tailSemiSpan], z=[hingeZ, hingeZ], k=2)


def rotateTail(val, geo) -> None:
    """Rotate the horizontal tail rigidly about its hinge line, the function of the tail rotation global DV

    Parameters
    ----------
    val : array
        The tail rotation in degrees, positive nose up (increasing the tail's incidence)
    geo : DVGeometry
        The DVGeometry object whose tail reference axis is rotated
    """
    # Every point of the hinge line gets the same rotation about y, so the tail rotates rigidly. The tail FFD
    # coefficients beyond the ends of the hinge line attach to its end points, which lie on the same line, so they
    # rotate rigidly too
    geo.rot_y[TAIL_AXIS_NAME].coef[:] = val[0]


def addTailRotation(DVGeo, flightPoint: FlightPoint) -> None:
    """Add the horizontal tail's hinge line and a flight point's tail rotation DV to a plain DVGeometry

    ``setupDVGeo`` adds the same reference axis and DV to an OpenMDAO geometry component.

    Parameters
    ----------
    DVGeo : DVGeometry
        DVGeometry object made from an FFD file with the wing block then the tail block
    flightPoint : FlightPoint
        The flight point whose tail rotation is added, starting at its ``tailRotation``
    """
    DVGeo.addRefAxis(TAIL_AXIS_NAME, curve=createTailHingeCurve(), volumes=[TAIL_FFD_VOLUME])
    DVGeo.addGlobalDV(getTailRotationDVName(flightPoint), value=flightPoint.tailRotation, func=rotateTail)


def setupDVGeo(
    args,
    top,
    DVGeoComp,
    dvComponent,
    dvScaleFactor,
    geoCompName=None,
    addGeoDVs=None,
    addGeoConstraints=None,
    computeRibBayVolumes=True,
    computeToC=False,
    tailRotationPoint: FlightPoint | None = None,
):
    if geoCompName is None:
        geoCompName = "geometry"
    if addGeoDVs is None:
        addGeoDVs = True
    if addGeoConstraints is None:
        addGeoConstraints = True

    DVGeo = DVGeoComp.nom_getDVGeo()

    spanIndex = wingGeometry["spanIndex"]
    chordIndex = wingGeometry["chordIndex"]
    verticalIndex = wingGeometry["verticalIndex"]

    # Create reference axis along the leading edge of the wing block only. With the one-block wing FFD this is the
    # same as giving no volumes
    numRefAxPts = DVGeoComp.nom_addRefAxis(name="wing", xFraction=0.00675, alignIndex="j", volumes=[WING_FFD_VOLUME])
    numRootSections = 3
    numSOBSections = 3
    numMovingSections = numRefAxPts - numRootSections - (numSOBSections)

    # Figure out the original chord lengths at the FFD sections, we need these so that we can keep a linear taper when
    # changing the root and tip chords

    refAxisCoords = DVGeo.axis["wing"]["curve"].X
    ffdSectionSpanwiseCoords = refAxisCoords[:, spanIndex]
    ffdSectionEta = ffdSectionSpanwiseCoords / ffdSectionSpanwiseCoords[-1]
    sectionEta = wingGeometry["wing"]["sectionEta"]
    sectionChord = wingGeometry["wing"]["sectionChord"]
    ffdSectionChords = np.interp(ffdSectionEta, sectionEta, sectionChord)

    # Figure out the chordwise distance between the reference axis and the leading edge spar at the root and SOB
    wingboxLECoords = wingGeometry["wingbox"]["LESparCoords"]
    centreBoxLECoord = wingboxLECoords[0, chordIndex]
    RootRefAxisCoord = refAxisCoords[0, chordIndex]
    SOBRefAxisCoord = refAxisCoords[numRootSections, chordIndex]

    RootLESparOffset = centreBoxLECoord - RootRefAxisCoord
    SOBLESparOffset = centreBoxLECoord - SOBRefAxisCoord

    # Set up global design variables
    if args.twist:

        def twist(val, geo):
            if spanIndex == 1:
                twistArray = geo.rot_y["wing"]
            elif spanIndex == 2:
                twistArray = geo.rot_z["wing"]
            # We don't twist the wing at the root because we have an angle of attack DV, we will also not twist at the
            # SOB because this would make the centre wingbox section very unrealistic
            sobStartInd = numRootSections
            sobEndInd = sobStartInd + numSOBSections
            # twistArray.coef[sobStartInd:sobEndInd] = val[0]
            for i in range(sobEndInd, numRefAxPts):
                twistArray.coef[i] = val[i - sobEndInd]

        DVGeoComp.nom_addGlobalDV(dvName="twist", value=[0.0] * numMovingSections, func=twist)
        try:
            dvComponent.add_output("twist", val=[0.0] * numMovingSections, units="deg")
        except ValueError:
            pass
        top.connect("twist", f"{geoCompName}.twist")
        if addGeoDVs:
            top.add_design_var(
                "twist",
                lower=-20.0,
                upper=20.0,
                scaler=dvScaleFactor * 1e-1,
                units="deg",
            )

    if args.taper:
        dvName = "taper"

        def taper(val, geo):
            # We scale the root and tip chords by the two DV values, then linearly interpolate the chord length (not the
            # scaling factor!) to the other FFD sections
            s = geo.extractS("wing")
            sobInd = numRootSections + numSOBSections - 1

            rootChord = ffdSectionChords[sobInd] * val[0]
            tipChord = ffdSectionChords[-1] * val[1]

            # Scale all of the root and SOB sections by the root chord scaling factor
            geo.scale_x["wing"].coef[:sobInd] = val[0]

            for ii in range(sobInd, numRefAxPts):
                spanwiseCoord = (s[ii] - s[sobInd]) / (s[-1] - s[sobInd])
                origChord = ffdSectionChords[ii]
                newChord = rootChord + (tipChord - rootChord) * spanwiseCoord
                geo.scale_x["wing"].coef[ii] = newChord / origChord
            # We need to shift the root section in the chordwise section to keep the centre wingbox section straight. To
            # do this, we need to know the distance between the reference axis and the leading edge spar at both the
            # root and SOB
            C = geo.extractCoef("wing")
            C[:numRootSections, chordIndex] += (val[0] - 1.0) * (SOBLESparOffset - RootLESparOffset)
            geo.restoreCoef(C, "wing")

        DVGeoComp.nom_addGlobalDV(dvName=dvName, func=taper, value=[1.0] * 2)
        try:
            dvComponent.add_output(dvName, val=np.array([1, 1]), units=None)
        except ValueError:
            pass
        top.connect(dvName, f"{geoCompName}.{dvName}")
        if addGeoDVs:
            top.add_design_var(dvName, lower=0.25, upper=2.0, scaler=dvScaleFactor * 1)

    if args.span:
        dvName = "span"

        def span(val, geo):
            C = geo.extractCoef("wing")
            s = geo.extractS("wing")
            # The DV value defines how far the tip section moves in the spanwise direction, this is then linearly
            # interpolated to zero at the SOB
            sobInd = numRootSections + numSOBSections - 1
            for i in range(sobInd + 1, numRefAxPts):
                frac = (s[i] - s[sobInd]) / (s[-1] - s[sobInd])
                C[i, spanIndex] += val[0] * frac
            geo.restoreCoef(C, "wing")

        DVGeoComp.nom_addGlobalDV(dvName=dvName, value=0.0, func=span)
        try:
            dvComponent.add_output(dvName, val=0.0, units=None)
        except ValueError:
            pass
        top.connect(dvName, f"{geoCompName}.{dvName}")
        if addGeoDVs:
            top.add_design_var(dvName, lower=-10.0, upper=20.0, scaler=dvScaleFactor * 0.1)

    if args.sweep:
        dvName = "sweep"

        def sweep(val, geo):
            C = geo.extractCoef("wing")
            s = geo.extractS("wing")
            sobInd = numRootSections + numSOBSections - 1
            for i in range(sobInd + 1, numRefAxPts):
                frac = (s[i] - s[sobInd]) / (s[-1] - s[sobInd])
                C[i, chordIndex] += val[0] * frac
            geo.restoreCoef(C, "wing")

        DVGeoComp.nom_addGlobalDV(dvName=dvName, value=0.0, func=sweep)
        try:
            dvComponent.add_output(dvName, val=0.0, units=None)
        except ValueError:
            pass
        top.connect(dvName, f"{geoCompName}.{dvName}")
        if addGeoDVs:
            top.add_design_var(dvName, lower=-10.0, upper=10.0, scaler=dvScaleFactor * 1)

    if args.xShift:
        dvName = "xShift"

        def xShift(val, geo):
            C = geo.extractCoef("wing")
            for i in range(numRefAxPts):
                C[i, chordIndex] += val[0]
            geo.restoreCoef(C, "wing")

        DVGeoComp.nom_addGlobalDV(dvName=dvName, value=0.0, func=xShift)
        try:
            dvComponent.add_output(dvName, val=0.0, units=None)
        except ValueError:
            pass
        top.connect(dvName, f"{geoCompName}.{dvName}")
        if addGeoDVs:
            top.add_design_var(dvName, lower=-10.0, upper=10.0, scaler=dvScaleFactor * 1)

    if args.shape:
        dvName = "local"

        shapes = []
        ffdLocalInds = DVGeo.getLocalIndex(WING_FFD_VOLUME)

        numChordwisePoints = ffdLocalInds.shape[0]

        direction = np.zeros(3)
        direction[verticalIndex] = 1.0

        # For each chordwise point, excluding the first and last (which are the leading and trailing edge nodes)
        for chordInd in range(1, numChordwisePoints - 1):
            # For the upper and lower nodes
            for verticalInd in range(ffdLocalInds.shape[2]):
                # Add one shape function to move this point in the root sections together
                rootShape = {}
                for spanInd in range(numRootSections):
                    rootShape[ffdLocalInds[chordInd, spanInd, verticalInd]] = direction
                shapes.append(rootShape)

                # Add one shape function to move this point in the SOB sections together
                sobShape = {}
                for spanInd in range(numRootSections, numRootSections + numSOBSections):
                    sobShape[ffdLocalInds[chordInd, spanInd, verticalInd]] = direction
                shapes.append(sobShape)

                # Add separate shape functions for each of the remaining sections
                for spanInd in range(numRootSections + numSOBSections, numRefAxPts):
                    shape = {ffdLocalInds[chordInd, spanInd, verticalInd]: direction}
                    shapes.append(shape)

        # Now add the leading edge DVs
        rootShape = {}
        for spanInd in range(numRootSections):
            rootShape[ffdLocalInds[0, spanInd, 0]] = -direction
            rootShape[ffdLocalInds[0, spanInd, -1]] = direction
        shapes.append(rootShape)
        sobShape = {}
        for spanInd in range(numRootSections, numRootSections + numSOBSections):
            sobShape[ffdLocalInds[0, spanInd, 0]] = -direction
            sobShape[ffdLocalInds[0, spanInd, -1]] = direction
        shapes.append(sobShape)

        for spanInd in range(numRootSections + numSOBSections, numRefAxPts):
            shape = {}
            shape[ffdLocalInds[0, spanInd, 0]] = -direction
            shape[ffdLocalInds[0, spanInd, -1]] = direction
            shapes.append(shape)

        DVGeoComp.nom_addShapeFunctionDV(dvName, shapes)
        try:
            dvComponent.add_output(dvName, val=0.0, units=None, shape_by_conn=True)
        except ValueError:
            pass
        top.connect(dvName, f"{geoCompName}.{dvName}")
        if addGeoDVs:
            top.add_design_var(dvName, lower=-0.5, upper=0.5, scaler=dvScaleFactor * 1)

    # The horizontal tail's rotation about its hinge line, for runs with the tail (tailRotationPoint is None without
    # it). It is a design variable whatever args and addGeoDVs are, so that every task, trim included, can use it
    if tailRotationPoint is not None:
        dvName = getTailRotationDVName(tailRotationPoint)
        DVGeoComp.nom_addRefAxis(TAIL_AXIS_NAME, curve=createTailHingeCurve(), volumes=[TAIL_FFD_VOLUME])
        DVGeoComp.nom_addGlobalDV(dvName=dvName, value=tailRotationPoint.tailRotation, func=rotateTail)
        try:
            dvComponent.add_output(dvName, val=tailRotationPoint.tailRotation, units="deg")
        except ValueError:
            pass
        top.connect(dvName, f"{geoCompName}.{dvName}")
        top.add_design_var(
            dvName,
            lower=TAIL_ROTATION_LOWER,
            upper=TAIL_ROTATION_UPPER,
            scaler=TAIL_ROTATION_SCALER,
            units="deg",
        )

    # ==============================================================================
    # Set up the DVConstraints
    # ==============================================================================
    chordDir = np.zeros(3)
    chordDir[chordIndex] = -1.0
    projectionDir = np.zeros(3)
    projectionDir[verticalIndex] = 1.0

    LECoords = wingGeometry["wing"]["LECoords"]
    LECoords[:, chordIndex] += 1e-2  # Need to be slightly behind the LE
    LECoords[0, spanIndex] += 1e-4  # Need to be in from the symmetry plane
    LECoords[-1, spanIndex] -= 1e-2  # Need to be in from the tip

    TECoords = wingGeometry["wing"]["TECoords"]
    TECoords[:, chordIndex] -= 1e-2  # Need to be slightly ahead of the TE
    TECoords[0, spanIndex] += 1e-4  # Need to be in from the symmetry plane
    TECoords[-1, spanIndex] -= 1e-2  # Need to be in from the tip

    # --- Wingbox volume ---
    if computeRibBayVolumes:
        # We'll add a volume constraint for each rib bay in the wingbox
        LESparCoords = wingGeometry["wingbox"]["LESparCoords"]
        TESparCoords = wingGeometry["wingbox"]["TESparCoords"]
        numRibsCentrebody = wingGeometry["wingbox"]["numRibsCentrebody"]
        numRibsOuter = wingGeometry["wingbox"]["numRibsOuter"]

        bayCount = 0

        # First the bays between the root and the SOB
        spanFrac = np.linspace(0, 1, numRibsCentrebody)
        for ii in range(numRibsCentrebody - 1):
            ribLE = LESparCoords[0] + spanFrac[ii] * (LESparCoords[1] - LESparCoords[0])
            ribTE = TESparCoords[0] + spanFrac[ii] * (TESparCoords[1] - TESparCoords[0])
            nextRibLE = LESparCoords[0] + spanFrac[ii + 1] * (LESparCoords[1] - LESparCoords[0])
            nextRibTE = TESparCoords[0] + spanFrac[ii + 1] * (TESparCoords[1] - TESparCoords[0])
            DVGeoComp.nom_addVolumeConstraint(
                f"RibBay-Volume_{bayCount}",
                [ribLE, nextRibLE],
                [ribTE, nextRibTE],
                nSpan=3,
                nChord=20,
                scaled=False,
            )
            bayCount += 1

        # Then the bays between the SOB and the tip
        spanFrac = np.linspace(0, 1, numRibsOuter + 1)
        for ii in range(numRibsOuter):
            ribLE = LESparCoords[1] + spanFrac[ii] * (LESparCoords[2] - LESparCoords[1])
            ribTE = TESparCoords[1] + spanFrac[ii] * (TESparCoords[2] - TESparCoords[1])
            nextRibLE = LESparCoords[1] + spanFrac[ii + 1] * (LESparCoords[2] - LESparCoords[1])
            nextRibTE = TESparCoords[1] + spanFrac[ii + 1] * (TESparCoords[2] - TESparCoords[1])
            DVGeoComp.nom_addVolumeConstraint(
                f"RibBay-Volume_{bayCount}",
                [ribLE, nextRibLE],
                [ribTE, nextRibTE],
                nSpan=5,
                nChord=20,
                scaled=False,
            )
            bayCount += 1

    # --- Add section t/c computation ---
    if computeToC:
        # NOTE: I decided against the approach below of only computing the t/c over the flapped region of the wing,
        # because the t/c value is also used to compute the overall drag coefficient of the wing for the takeoff analysis

        # Since we are using the t/c values for the openconcept CL max estimation, we only want to compute the t/c over the
        # flapped span of the wing
        # FLAP_INBOARD_SPAN_FRAC = wingGeometry["wing"]["flapInboardSpanFrac"]
        # FLAP_INBOARD_SPAN = FLAP_INBOARD_SPAN_FRAC * wingGeometry["wing"]["semiSpan"]
        # FLAP_OUTBOARD_SPAN_FRAC = wingGeometry["wing"]["flapOutboardSpanFrac"]
        # FLAP_OUTBOARD_SPAN = FLAP_OUTBOARD_SPAN_FRAC * wingGeometry["wing"]["semiSpan"]

        # # Find the last section Inboard of the flaps
        # inboardInd = np.where(LECoords[:, spanIndex] <= FLAP_INBOARD_SPAN)[0][-1]
        # # Find the first section outboard of the flaps
        # outboardInd = np.where(LECoords[:, spanIndex] >= FLAP_OUTBOARD_SPAN)[0][0]

        # flapLECoords = LECoords[inboardInd : outboardInd + 1].copy()
        # flapTECoords = TECoords[inboardInd : outboardInd + 1].copy()
        # # Now interpolate the first and last sections to be exactly at the flap edges
        # for coords in [flapLECoords, flapTECoords]:
        #     # Inboard
        #     firstSectionFrac = (FLAP_INBOARD_SPAN - coords[0, spanIndex]) / (
        #         coords[1, spanIndex] - coords[0, spanIndex]
        #     )
        #     coords[0, :] = coords[0, :] + firstSectionFrac * (coords[1, :] - coords[0, :])
        #     # Outboard
        #     lastSectionFrac = (FLAP_OUTBOARD_SPAN - coords[-2, spanIndex]) / (
        #         coords[-1, spanIndex] - coords[-2, spanIndex]
        #     )
        #     coords[-1, :] = coords[-2, :] + lastSectionFrac * (coords[-1, :] - coords[-2, :])

        # Rough calculation that should give us at least 1 ToC section per FFD section over the span
        numToCSections = int(1.5 * numMovingSections) + 1
        DVGeoComp.nom_addThicknessToChordConstraints2D(
            "SectionToC",
            leList=LECoords,
            teList=TECoords,
            nSpan=numToCSections,
            nChord=60,
            sectionMax=True,
            ksRho=1e3,
            scaled=False,
        )

    if addGeoConstraints:
        if args.shape:
            # --- Leading edge radius constraint ---
            DVGeoComp.nom_addLERadiusConstraints(
                "LERadius",
                LECoords,
                nSpan=20,
                axis=projectionDir,
                chordDir=chordDir,
            )
            top.add_constraint(f"{geoCompName}.LERadius", lower=0.9)

            # --- Thickness constraints ---
            # We will add two forms of thickness constraints here:

            # First, thickness constraints along the leading and trailing edge spars that limit how much the optimiser can
            # reduce their height (so there's still space to mount actuators etc to them)
            DVGeoComp.nom_addThicknessConstraints1D("LESparThickness", LESparCoords, nCon=20, axis=projectionDir)
            DVGeoComp.nom_addThicknessConstraints1D("TESparThickness", TESparCoords, nCon=20, axis=projectionDir)
            top.add_constraint(f"{geoCompName}.LESparThickness", lower=0.75)
            top.add_constraint(f"{geoCompName}.TESparThickness", lower=0.75)

            # Second a grid of constraints over the region between the wingbox trailing edge and the wing trailing edge,
            # to stop the optimiser thinning out the trailing edge of the wing too much, which is a common issue
            DVGeoComp.nom_addThicknessConstraints2D("TEThickness", TESparCoords, TECoords, nSpan=20, nChord=20)
            top.add_constraint(f"{geoCompName}.TEThickness", lower=0.5)
        if args.addPlanformCons:
            # Constraint that the rear spar of the centre wingbox cannot move any further backward than its original location
            centreBoxTECoords = wingGeometry["wingbox"]["TESparCoords"][:2]
            numPoints = centreBoxTECoords.shape[0]
            DVGeoComp.nom_addLocationConstraints1D("TESparLocation", centreBoxTECoords, numPoints, scaled=False)
            # The constraint output holds the flattened (x, y, z) of each point, only the chordwise components are
            # bounded, the infinite bounds on the others leave them unconstrained
            upperBound = np.full_like(centreBoxTECoords, np.inf)
            upperBound[:, chordIndex] = centreBoxTECoords[:, chordIndex]
            top.add_constraint(f"{geoCompName}.TESparLocation", upper=upperBound.flatten())
