def getExtrusionOptions(smoothingLevel):
    if smoothingLevel.lower() not in ["min", "max", "maxatend"]:
        raise ValueError(f"Invalid smoothing level: {smoothingLevel}. Must be one of 'none', 'max', or 'maxAtEnd'.")

    # Common extrusion settings for all extrusions
    options = {
        # ---------------------------
        #   General options
        # ---------------------------
        "fileType": "CGNS",
        "unattachedEdgesAreSymmetry": True,
        "autoConnect": True,
        "BC": {},
        "families": "wall",
        # ---------------------------
        #   Pseudo Grid Parameters
        # ---------------------------
        "ps0": -1.0,
        "pGridRatio": -1.0,
        "cMax": 1.0,
        # ---------------------------
        #   Smoothing parameters
        # ---------------------------
        "epsE": 1.0,
        "epsI": 2.0,
        "theta": 3.0,
        "volCoef": 0.25,
        "volBlend": 1e-4,
        "volSmoothIter": 100,
        # ---------------------------
        #   Solution Parameters
        # ---------------------------
        "kspRelTol": 1e-10,
        "kspMaxIts": 1500,
        "kspSubspaceSize": 50,
    }

    if smoothingLevel.lower() == "max":
        options.update(
            {
                "epsE": 1.0,
                "epsI": 6.0,
                "volCoef": 1.0,
                "volBlend": 1e-3,
                "volSmoothIter": 500,
            }
        )

    if smoothingLevel.lower() == "maxatend":
        switchFrac = 0.5
        options.update(
            {
                "epsE": [[0.0, 1.0], [switchFrac, 1.0], [1.0, 1.0]],
                "epsI": [[0.0, 2.0], [switchFrac, 2.0], [1.0, 6.0]],
                "volCoef": [[0.0, 0.1], [switchFrac, 0.1], [1.0, 1.0]],
                "volBlend": [[0.0, 1e-6], [switchFrac, 1e-6], [1.0, 1e-3]],
                "volSmoothIter": [[0.0, 100], [switchFrac, 100], [1.0, 500]],
            }
        )

    return options
