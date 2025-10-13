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

# ==============================================================================
# Extension modules
# ==============================================================================
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../geometry"))
from wingGeometry import wingGeometry  # noqa: E402


def setupWimpress():
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
    wp.addTrapSegment(leadingEdges=leCoords, trailingEdges=teCoords, nSegment=10)

    return wp, WimpressCalcComp(wimpresscalc=wp), WimpressCoordsComp(wimpresscalc=wp)


if __name__ == "__main__":
    wp, calcComp, coordsComp = setupWimpress()
    wp.writeTecplot("wimpressTest")
