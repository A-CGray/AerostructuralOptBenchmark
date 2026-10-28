"""
==============================================================================
WimpressCalc Setup Script
==============================================================================
@File    :   setupWimpress.py
@Date    :   2025/10/06
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
from wimpresscalc import WimpressCalc, WimpressCalcComp, WimpressCoordsComp
import numpy as np
import openmdao.api as om

# ==============================================================================
# Extension modules
# ==============================================================================
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../geometry"))
from wingGeometry import wingGeometry  # noqa: E402

LG_STATION_NAME = "LandingGear"
PLANFORM_CON_INPUT_NAMES = ["MAC", "xLEMAC", f"xLE_{LG_STATION_NAME}", f"chord_{LG_STATION_NAME}"]


def setupWimpress(addLandingGearCon=False):
    liftIndex = wingGeometry["verticalIndex"] + 1
    spanIndex = wingGeometry["spanIndex"]

    # Get leading and trailing edge points from wingGeometry, we want to use points at the root, SOB, and tip
    leCoords = wingGeometry["wing"]["LECoords"]
    teCoords = wingGeometry["wing"]["TECoords"]

    # Interpolate to get the SOB point
    SOBSpanCoord = wingGeometry["wingbox"]["SOB"]
    SOBLECoord = np.zeros(3)
    SOBLECoord[spanIndex] = SOBSpanCoord
    for ii in range(3):
        if ii != spanIndex:
            SOBLECoord[ii] = np.interp(
                SOBSpanCoord,
                leCoords[:, spanIndex],
                leCoords[:, ii],
            )

    leCoords = np.vstack((leCoords[0], SOBLECoord, leCoords[1:]))

    wp = WimpressCalc(liftIndex=liftIndex)
    wp.addTrapSegment(leadingEdges=leCoords, trailingEdges=teCoords, nSegment=1)

    # Add landing gear station
    if addLandingGearCon:
        wp.addSpanwiseStation(location=wingGeometry["wing"]["landingGearSpan"], name="LandingGear", isRelative=False)

    return wp, WimpressCalcComp(wimpresscalc=wp), WimpressCoordsComp(wimpresscalc=wp)


class PlanformConGroup(om.Group):
    def initialize(self):
        self.options.declare("wingGeometry", types=dict)

    def setup(self):
        wingGeometry = self.options["wingGeometry"]

        # MAC location/CG constraints
        forwardCGComp = om.AddSubtractComp(
            output_name="forwardCGX",
            input_names=["xLEMAC", "MAC"],
            scaling_factors=[1.0, wingGeometry["wing"]["forwardCGMAC"]],
        )

        self.add_subsystem("forwardCGComp", forwardCGComp, promotes=["*"])

        aftCGComp = om.AddSubtractComp(
            output_name="aftCGX",
            input_names=["xLEMAC", "MAC"],
            scaling_factors=[1.0, wingGeometry["wing"]["aftCGMAC"]],
        )

        self.add_subsystem("aftCGComp", aftCGComp, promotes=["*"])

        # Landing gear constraints
        landingGearTESparComp = om.AddSubtractComp(
            output_name="landingGearTESparX",
            input_names=["xLE_LandingGear", "chord_LandingGear"],
            scaling_factors=[1.0, wingGeometry["wingbox"]["TESparFrac"]],
        )

        self.add_subsystem("landingGearTESparComp", landingGearTESparComp, promotes=["*"])

        landingGearTEComp = om.AddSubtractComp(
            output_name="landingGearTEX",
            input_names=["xLE_LandingGear", "chord_LandingGear"],
            scaling_factors=[1.0, 1.0],
        )

        self.add_subsystem("landingGearTEComp", landingGearTEComp, promotes=["*"])


if __name__ == "__main__":
    wp, calcComp, coordsComp = setupWimpress()
    wp.writeTecplot("wimpressTest")
