def getIDWarpOptions(meshFile):
    warpOptions = {
        "gridFile": meshFile,
        "fileType": "CGNS",
        "aExp": 3.0,
        "bExp": 5.0,
        "LdefFact": 1000.0,
        "alpha": 0.25,
        "errTol": 0.0005,
        "evalMode": "fast",
        "symmTol": 1e-6,
        "useRotations": True,
        "zeroCornerRotations": False,
        "cornerAngle": 30.0,
        "bucketSize": 8,
    }
    return warpOptions
