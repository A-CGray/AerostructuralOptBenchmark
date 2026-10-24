import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import niceplots

from splinetoolbox.fitting import curveLMSApprox

from splinetoolbox.smoothing import smoothCurve
from splinetoolbox.export import writeIGES
from foilgen.smoothing import plotCurvatureComb

sys.path.append("../")

plt.style.use(niceplots.get_style())
colours = niceplots.get_colors()
colourList = niceplots.get_colors_list()


def getCosinePoints(curve, n):
    theta = np.linspace(0, np.pi, n)
    # cosine spacing from s=0 -> 1/2
    s = (1 - np.cos(-theta)) / 4
    upperPoints = curve(s)
    lowerPoints = np.flip(np.copy(upperPoints), axis=0)
    lowerPoints[:, 1] *= -1
    points = np.vstack((upperPoints, lowerPoints[1:]))
    return points


curvatureScale = 1e-1

airfoilFiles = [
    "CRMTailCADRoot-Fixed.csv",
    "CRMTailCADTip-Fixed.csv",
]

for ii, airfoilFile in enumerate(airfoilFiles):
    # Create figure with one wide axis on top row, 3 axes in second row
    fig = plt.figure(figsize=(16, 10), constrained_layout=True)
    gs = GridSpec(2, 3, figure=fig)
    axes = []
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])
    axes = [ax1, ax2, ax3, ax4]
    # Define x and y limits for each axis
    xlims = [
        [-0.2, 1.02],
        [None, None],
        [-0.02, 0.1],
        [0.8, 1.0],
    ]
    ylims = [
        [-0.25, 0.25],
        [None, None],
        [-0.05, 0.05],
        [-0.04, 0.04],
    ]

    curvatureLineArgs = {"c": colourList[ii], "linewidth": 0.5}

    # csv file only contains upper surface
    points = np.loadtxt(airfoilFile, delimiter=",")

    for ax in axes:
        ax.plot(
            points[:, 0],
            points[:, 1],
            "o",
            label="Original Points",
            color=colourList[ii],
            clip_on=True,
            markersize=6,
        )
    weights = np.ones(points.shape[0])
    LEPointInd = np.argmin(points[:, 0])
    weights[LEPointInd] = (
        1e3  # Give the leading edge point a very high weight to make sure the fitted curve goes through it
    )
    airfoil, _ = curveLMSApprox(points=points, degree=5, nCtl=201, maxIter=1, tol=1e-6, nParamIters=10, weights=weights)
    airfoil.computeData()
    splinePoints = airfoil.data
    for ax in axes:
        ax.plot(
            splinePoints[:, 0],
            splinePoints[:, 1],
            "-",
            label="Fitted Spline",
            color=colourList[ii],
            clip_on=True,
            markersize=6,
        )
        ax.plot(
            airfoil.ctrlPnts[:, 0],
            airfoil.ctrlPnts[:, 1],
            "x",
            label="Fitted Spline control points",
            color=colourList[ii],
            clip_on=True,
            markersize=6,
            markeredgecolor=colourList[ii],
        )
        plotCurvatureComb(
            ax, airfoil, scale=curvatureScale, dims=[0, 1], argsComb=curvatureLineArgs, argsCurve=curvatureLineArgs
        )

    smooth = True

    if smooth:
        # --- Smoothing ---
        numCtl = airfoil.nCtl
        numDims = airfoil.nDim

        # Use all five point stencil for smoothing, Shounak says it gives better results. Five point stencils cannot be
        # applied at the first two or last two points. Three point stencils cannot be applied at the first or last point.
        fivePointInds = []
        threePointInds = []
        for jj in range(numCtl):
            if jj >= 2 and jj <= numCtl - 3:
                fivePointInds.append(jj)
            elif jj > 0 and jj < numCtl - 1:
                threePointInds.append(jj)

        # Fix the trailing edge points, and the max t/c points to maintain t/c
        LEPointInd = np.argmin(airfoil.ctrlPnts[:, 0])
        maxYPointInd = np.argmax(airfoil.ctrlPnts[:, 1])
        minYPointInd = np.argmin(airfoil.ctrlPnts[:, 1])
        bcIdxs = []
        ptsToFix = [0, maxYPointInd, minYPointInd, numCtl - 1]
        for ind in ptsToFix:
            bcIdxs.append((ind, 0))
            bcIdxs.append((ind, 1))

        # also fix the points adjacent to the leading edge in x to maintain LE position
        numFixedLEPts = 2
        for jj in range(numFixedLEPts):
            bcIdxs.append((LEPointInd + jj, 0))
            bcIdxs.append((LEPointInd - jj, 0))

        # Only do 1 smoothing iterations involving the first n points, to maintain LE shape, then do many more on rest of airfoil
        smoothedAirfoil, _ = smoothCurve(
            airfoil,
            threePointInds,
            fivePointInds,
            bcIdxs,
            fivePointWeight=0.25 * 1 / 16,
            smoothingIters=4,
        )
        numFixedLEPts = 6
        for jj in range(numFixedLEPts):
            bcIdxs.append((LEPointInd + jj, 0))
            bcIdxs.append((LEPointInd + jj, 1))
            bcIdxs.append((LEPointInd - jj, 0))
            bcIdxs.append((LEPointInd - jj, 1))
        smoothedAirfoil, _ = smoothCurve(
            smoothedAirfoil,
            threePointInds,
            fivePointInds,
            bcIdxs,
            smoothingIters=40,
        )

        # Do a another bout of smoothing on the last 25% of the chord
        bcIdxs = []
        for jj in range(numCtl):
            if smoothedAirfoil.ctrlPnts[jj, 0] <= 0.75 or jj in [0, numCtl - 1]:
                bcIdxs.append((jj, 0))
                bcIdxs.append((jj, 1))
        smoothedAirfoil, _ = smoothCurve(
            smoothedAirfoil,
            threePointInds,
            fivePointInds,
            bcIdxs,
            smoothingIters=200,
        )

        smoothedAirfoil.computeData(recompute=True)
        smoothedSplineColour = colourList[ii + 1]
        for ax in axes:
            ax.plot(
                smoothedAirfoil.data[:, 0],
                smoothedAirfoil.data[:, 1],
                "-",
                label="Smoothed Spline",
                color=smoothedSplineColour,
                clip_on=True,
                markersize=6,
            )
            ax.plot(
                smoothedAirfoil.ctrlPnts[:, 0],
                smoothedAirfoil.ctrlPnts[:, 1],
                "x",
                label="Smoothed Spline control points",
                color=smoothedSplineColour,
                clip_on=True,
                markersize=6,
                markeredgecolor=smoothedSplineColour,
            )
            curvatureLineArgs["c"] = smoothedSplineColour
            plotCurvatureComb(
                ax,
                smoothedAirfoil,
                scale=curvatureScale,
                dims=[0, 1],
                argsComb=curvatureLineArgs,
                argsCurve=curvatureLineArgs,
            )

    axes[0].legend(labelcolor="linecolor", loc="upper left", bbox_to_anchor=(1.02, 1))
    for ii, ax in enumerate(axes):
        niceplots.adjust_spines(ax)
        ax.set_xlim(left=xlims[ii][0], right=xlims[ii][1])
        ax.set_ylim(bottom=ylims[ii][0], top=ylims[ii][1])
        ax.set_aspect("equal")

    writeFiles = True
    if writeFiles:
        # --- Write out files ---
        fileNameBase = os.path.split(airfoilFile)[-1].replace("_processed", "")
        fileNameBase = os.path.splitext(fileNameBase)[0]
        fig.suptitle(fileNameBase)

        # Fitted spline
        # writeTecplot(
        #     airfoil,
        #     fileNameBase + "_spline.dat",
        #     controlPoints=False,
        #     orig=False,
        # )
        writeIGES(airfoil, fileNameBase + "_spline.igs")
        points = getCosinePoints(airfoil, 101)
        np.savetxt(fileNameBase + "_spline.dat", points, delimiter=" ")

        if smooth:
            # Smoothed spline
            # writeTecplot(
            #     smoothedAirfoil,
            #     fileNameBase + "_spline_smoothed.dat",
            #     controlPoints=False,
            #     orig=False,
            # )
            writeIGES(smoothedAirfoil, fileNameBase + "_spline_smoothed.igs")
            points = getCosinePoints(smoothedAirfoil, 101)
            np.savetxt(fileNameBase + "_spline_smoothed.dat", points, delimiter=" ")

    niceplots.save_figs(fig, f"{fileNameBase}_smoothed", ["pdf", "png"])
# plt.show()
