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

# ==============================================================================
# Extension modules
# ==============================================================================
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../geometry"))
from wingGeometry import wingGeometry  # noqa: E402


def c_atan2(x, y):
    a = x.real
    b = x.imag
    c = y.real
    d = y.imag
    return complex(np.arctan2(a, c), (c * b - a * d) / (a**2 + c**2))


def setupDVGeo(
    args, top, DVGeoComp, dvComponent, dvScaleFactor, geoCompName=None, addGeoDVs=None, addGeoConstraints=None
):
    if geoCompName is None:
        geoCompName = "geometry"
    if addGeoDVs is None:
        addGeoDVs = True
    if addGeoConstraints is None:
        addGeoConstraints = True

    DVGeo = DVGeoComp.nom_getDVGeo()
    DVCon = DVGeoComp.nom_getDVCon()

    spanIndex = wingGeometry["spanIndex"]
    chordIndex = wingGeometry["chordIndex"]
    verticalIndex = wingGeometry["verticalIndex"]

    # Create reference axis along the leading edge
    numRefAxPts = DVGeoComp.nom_addRefAxis(name="wing", xFraction=0.00675, alignIndex="j")
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
            top.add_design_var("twist", lower=-20.0, upper=20.0, scaler=dvScaleFactor * 1e-1, units="deg")

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

    if args.shape:
        dvName = "local"

        shapes = []
        ffdLocalInds = DVGeo.getLocalIndex(0)

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

    # ==============================================================================
    # Set up the DVConstraints
    # ==============================================================================
    # We can only add geometric constraints if we have a triangulated surface
    if len(DVCon.surfaces) > 0 and addGeoConstraints:
        chordDir = np.zeros(3)
        chordDir[chordIndex] = -1.0
        projectionDir = np.zeros(3)
        projectionDir[verticalIndex] = 1.0

        # --- Wingbox volume ---
        # This will be used to constrain the volume of the wingbox to be greater than the required fuel volume
        LESparCoords = wingGeometry["wingbox"]["LESparCoords"]
        TESparCoords = wingGeometry["wingbox"]["TESparCoords"]
        DVGeoComp.nom_addVolumeConstraint(
            "WingboxVolume", LESparCoords, TESparCoords, nSpan=20, nChord=20, scaled=False
        )

        # --- Area constraint ---
        # This will be used to compute and potentially constrain the wing loading
        directions = ["x", "y", "z"]
        DVGeoComp.nom_addProjectedAreaConstraint("WingArea", axis=directions[verticalIndex], scaled=False)

        if args.shape:
            # --- Leading/Trailing edge constraints ---
            # DVGeoComp.nom_add_LETEConstraint("LEConstraint", 0, "iLow")
            # DVGeoComp.nom_add_LETEConstraint("TEConstraint", 0, "iHigh")

            # top.add_constraint(f"{geoCompName}.LEConstraint", equals=0.0, scaler=1.0, linear=True)
            # top.add_constraint(f"{geoCompName}.TEConstraint", equals=0.0, scaler=1.0, linear=True)

            # --- Leading edge radius constraint ---
            LECoords = wingGeometry["wing"]["LECoords"]
            LECoords[:, chordIndex] += 2e-2  # Need to be slightly behind the LE
            LECoords[0, spanIndex] += 1e-4  # Need to be in from the symmetry plane
            try:  # This fails on the super coarse mesh so we'll just ignore it
                DVGeoComp.nom_addLERadiusConstraints(
                    "LERadius", LECoords, nSpan=20, axis=projectionDir, chordDir=chordDir
                )
                top.add_constraint(f"{geoCompName}.LERadius", lower=0.9)
            except Exception:
                pass

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
            TECoords = wingGeometry["wing"]["TECoords"]
            TECoords[:, chordIndex] -= 1e-2  # Need to be slightly ahead of the TE
            TECoords[0, spanIndex] += 1e-2  # Need to be in from the symmetry plane
            TECoords[-1, spanIndex] -= 1e-2  # Need to be in from the tip
            DVGeoComp.nom_addThicknessConstraints2D("TEThickness", TESparCoords, TECoords, nSpan=20, nChord=20)
            top.add_constraint(f"{geoCompName}.TEThickness", lower=0.5)

        if top.comm.rank == 0 and addGeoConstraints:
            print("Added DV constraints")
            DVCon.writeTecplot("DVConstraints.dat")
            DVCon.writeSurfaceTecplot("DVConstraintsSurface.dat")
            DVCon.writeSurfaceSTL("DVConstraintsSurface.stl")
            DVGeo.writeRefAxes("FFDRefAxis")
