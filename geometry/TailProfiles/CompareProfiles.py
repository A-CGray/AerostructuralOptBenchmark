import sys

import numpy as np
import matplotlib.pyplot as plt
import niceplots

sys.path.append("../")
from wingGeometry import wingGeometry

plt.style.use(niceplots.get_style())
colours = niceplots.get_colors()
colourList = niceplots.get_colors_list()

curvatureScale = 1e-1

CADProfiles = [
    [
        "CRMTailCADRootUpper.csv",
        # "CRMTailCADRootLower.csv",
    ],
    [
        "CRMTailCADTipUpperRear.csv",
        "CRMTailCADTipUpperFront.csv",
        # "CRMTailCADTipLowerFront.csv",
        # "CRMTailCADTipLowerRear.csv",
    ],
]
profileNames = [
    "CRMTailCADRoot",
    "CRMTailCADTip",
]
CADTEHeight = 13.17
maxThicknesses = [0.05, 0.04]

digitizedProfiles = [
    "CRMTailRoot.csv",
    "CRMTailTip.csv",
]

fig, axes = plt.subplots(len(digitizedProfiles), 1, figsize=(16, 6), sharex=True, sharey=True)

for ii, (CADProfile, digitizedProfile, maxThickness, profileName) in enumerate(
    zip(CADProfiles, digitizedProfiles, maxThicknesses, profileNames, strict=True)
):
    ax = axes[ii]
    niceplots.adjust_spines(ax)
    CADPoints = np.zeros((0, 2))
    for prof in CADProfile:
        upperCADPoints = np.loadtxt(prof, delimiter=",")  # , skiprows=1)
        CADPoints = np.vstack([CADPoints, upperCADPoints])

    # The CAD points are offset from the origin, in mm and have a shearing twist applied, they are also not necessarily
    # the right thickness, so to get the normalized points we need to do the following:
    # 1. Subtract the leading edge point from all points to move the origin to the leading edge
    # 2. Apply a shearing twist to the points to get the trailing edge at the correct y-coordinate
    # 2. Scale the points to a chord length of 1
    # 3. Iteratively scale and shear the points to get the maximum thickness to match the target maximum thickness, while maintaining the correct trailing edge height

    # First, sort in descending order of x-coordinate so that the leading edge point is the last point in the array
    CADPoints = CADPoints[np.argsort(CADPoints[:, 0])[::-1], :]
    leadingEdgePoint = CADPoints[-1, :]
    CADPoints -= leadingEdgePoint  # move origin to leading edge

    # Apply inverse of shearing twist
    chordLength = np.max(CADPoints[:, 0]) - np.min(CADPoints[:, 0])
    TEYCoord = CADPoints[0, 1]
    CADPoints[:, 1] -= (TEYCoord - CADTEHeight / 2) * CADPoints[:, 0] / chordLength

    # Scale airfoil to chord length of 1
    CADPoints /= chordLength  # scale to chord length of 1

    # Now we want to match the maximum thickness of the CAD profile to the target maximum thickness, but scaling the
    # thickness will also change the trailing edge height, so after each scaling we need to apply a shearing twist to
    # get the trailing edge back to the correct height. This will change the max thickness again so we iterate until we
    # converge to the target maximum thickness
    maxIter = 10
    tol = 1e-12
    # Compute the target trailing edge height as a fraction of the chord length, so that it's the right size when we scale the airfoil to the right chord length when lofting the tail
    targetTEHeight = wingGeometry["hTail"]["teHeight"] / wingGeometry["hTail"]["sectionChord"][ii] / 2
    for ii in range(maxIter):
        currentMaxThickness = np.max(CADPoints[:, 1])
        print(
            f"Iteration {ii}: Current max thickness = {(100 * currentMaxThickness):11.7g}, target max thickness = {(100 * maxThickness):11.7g}"
        )
        if abs(currentMaxThickness - maxThickness) < tol:
            break
        # Scale points to get the correct max thickness
        CADPoints[:, 1] *= maxThickness / currentMaxThickness
        # Apply shearing twist to get the correct trailing edge height
        TEYCoord = CADPoints[0, 1]
        chordLength = np.max(CADPoints[:, 0]) - np.min(CADPoints[:, 0])
        CADPoints[:, 1] -= (CADPoints[0, 1] - targetTEHeight) * CADPoints[:, 0] / chordLength

    # Now create the lower surface points by mirroring the upper surface points about the x-axis, and flip the order of the points so that they go from leading edge to trailing edge
    lowerCADPoints = CADPoints.copy()
    lowerCADPoints[:, 1] *= -1.0
    lowerCADPoints = np.flip(lowerCADPoints, 0)
    CADPoints = np.vstack((CADPoints, lowerCADPoints[1:, :]))

    np.savetxt(f"{profileName}-Fixed.csv", CADPoints, delimiter=",")

    ax.plot(CADPoints[:, 0], CADPoints[:, 1], "-", label="CAD Profile", clip_on=False)

    # Compare with digitized points
    digitizedPoints = np.loadtxt(digitizedProfile, delimiter=",", skiprows=1)
    ax.plot(
        digitizedPoints[:, 0],
        digitizedPoints[:, 1],
        "o",
        label="Digitized Points",
        clip_on=False,
        markersize=6,
    )
    ax.legend(labelcolor="linecolor")
    ax.set_aspect("equal")
    niceplots.adjust_spines(ax)
niceplots.save_figs(fig, "FixedCADProfiles", formats=["png", "pdf"])
plt.show()
