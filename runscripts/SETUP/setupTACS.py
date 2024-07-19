"""
==============================================================================
Structural setup functions
==============================================================================
@File    :   setupTACS.py
@Date    :   2023/04/21
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import sys
from typing import Iterable

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from tacs import elements, constitutive, functions, TACS

# ==============================================================================
# Extension modules
# ==============================================================================
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../INPUT/geometry"))
from wingGeometry import wingGeometry  # noqa: E402

# ==============================================================================
# Aluminium 7075-T6 Properties
# ==============================================================================
# Taken from https://asm.matweb.com/search/SpecificMaterial.asp?bassnum=ma7075t6
aluProperties = {
    "rho": 2.81e3,  # density kg/m^3
    "E": 71.7e9,  # Young's modulus (Pa)
    "nu": 0.33,  # Poisson's ratio
    "ys": 503e6,  # yield stress
}

# ==============================================================================
# Composite properties
# ==============================================================================
# The material properties are taken from "Aerostructural tradeoffs for tow-steered
# composite wings" by Tim Brooks, https://doi.org/10.2514/1.C035699
# The ply fractions are taken from chapter 7 of the PhD thesis of Johannes Dillinger,
# available at https://repository.tudelft.nl/islandora/object/uuid%3A20484651-fd5d-49f2-9c56-355bc680f2b7
compositeProperties = {
    "E11": 117.9e9,  # Young's modulus in 11 direction (Pa)
    "E22": 9.7e9,  # Young's modulus in 22 direction (Pa)
    "G12": 4.8e9,  # in-plane 1-2 shear modulus (Pa)
    "G13": 4.8e9,  # Transverse 1-3 shear modulus (Pa)
    "G23": 4.8e9,  # Transverse 2-3 shear modulus (Pa)
    "nu12": 0.35,  # 1-2 poisson's ratio
    "rho": 1.55e3,  # density kg/m^3
    "T1": 1648e6,  # Tensile strength in 1 direction (Pa)
    "C1": 1034e6,  # Compressive strength in 1 direction (Pa)
    "T2": 64e6,  # Tensile strength in 2 direction (Pa)
    "C2": 228e6,  # Compressive strength in 2 direction (Pa)
    "S12": 71e6,  # Shear strength direction (Pa)
}
skinPlyAngles = np.deg2rad(np.array([0.0, -45.0, 45.0, 90.0]))
skinPlyFracs = np.array([44.41, 22.2, 22.2, 11.19]) / 100.0
sparRibPlyAngles = np.deg2rad(np.array([0.0, -45.0, 45.0, 90.0]))
sparRibPlyFracs = np.array([10.0, 35.0, 35.0, 20.0]) / 100.0

kcorr = 5.0 / 6.0  # shear correction factor

# --- Design variable values ---
flangeFraction = 1.0
# Panel length
defaultPanelLengthMax = np.inf
panelLengthMin = 0.0
panelLengthScale = 1.0

# Stiffener pitch
defaultStiffenerPitch = 0.15  # m
defaultStiffenerPitchMax = 0.5  # m
defaultStiffenerPitchMin = 0.15  # m
defaultStiffenerPitchScale = 1.0

# Panel thickness
defaultPanelThickness = 0.0065  # m
defaultPanelThicknessMax = 0.1  # m
defaultPanelThicknessMin = 0.6e-3  # m
defaultPanelThicknessScale = 100.0

# ply fraction bounds
defaultPlyFractionMax = 1.0
defaultPlyFractionMin = 0.1
defaultPlyFractionScale = 1.0

# Stiffener height
defaultStiffenerHeight = 0.05  # m
defaultStiffenerHeightMax = 0.15  # m
defaultStiffenerHeightMin = max(25e-3 * flangeFraction, 18e-3)  # m
defaultStiffenerHeightScale = 10.0

# Stiffener thickness
defaultStiffenerThickness = 0.006  # m
defaultStiffenerThicknessMax = 0.1  # m
defaultStiffenerThicknessMin = 0.6e-3  # m
defaultStiffenerThicknessScale = 100.0


# --- Figure out the rib spacing ---
spanIndex = wingGeometry["spanIndex"]
verticalIndex = wingGeometry["verticalIndex"]
LESparCoords = wingGeometry["wingbox"]["LESparCoords"]
TESparCoords = wingGeometry["wingbox"]["TESparCoords"]
numRibsCentrebody = wingGeometry["wingbox"]["numRibsCentrebody"]
numRibsOuter = wingGeometry["wingbox"]["numRibsOuter"]
inboardRibSpacing = (LESparCoords[1, spanIndex] - LESparCoords[0, spanIndex]) / numRibsCentrebody
outboardRibSpacing = np.linalg.norm(LESparCoords[2] - LESparCoords[1]) / numRibsOuter
TESparDirection = TESparCoords[2] - TESparCoords[1]
TESparDirection /= np.linalg.norm(TESparDirection)


def computeDVNums(dvNum, usePanelLengthDVs, usePlyFractionDVs, useStiffenerPitchDVs, numPlies):
    dvNums = {}
    currDVNum = dvNum
    if usePanelLengthDVs:
        dvNums["panelLength"] = currDVNum
        currDVNum += 1
    else:
        dvNums["panelLength"] = -1

    if useStiffenerPitchDVs:
        dvNums["stiffenerPitch"] = currDVNum
        currDVNum += 1
    else:
        dvNums["stiffenerPitch"] = -1

    dvNums["panelThickness"] = currDVNum
    currDVNum += 1

    if usePlyFractionDVs:
        dvNums["panelPlyFractions"] = np.arange(currDVNum, currDVNum + numPlies).astype(np.intc)
        currDVNum += numPlies
    else:
        dvNums["panelPlyFractions"] = -np.ones(numPlies).astype(np.intc)

    dvNums["stiffenerHeight"] = currDVNum
    currDVNum += 1

    dvNums["stiffenerThickness"] = currDVNum
    currDVNum += 1

    if usePlyFractionDVs:
        dvNums["stiffenerPlyFractions"] = np.arange(currDVNum, currDVNum + numPlies).astype(np.intc)
        currDVNum += numPlies
    else:
        dvNums["stiffenerPlyFractions"] = -np.ones(numPlies).astype(np.intc)

    return dvNums


def computeDVScales(usePanelLengthDVs, usePlyFractionDVs, useStiffenerPitchDVs, numPlies):
    DVScales = []
    if usePanelLengthDVs:
        DVScales.append(panelLengthScale)
    if useStiffenerPitchDVs:
        DVScales.append(defaultStiffenerPitchScale)
    DVScales.append(defaultPanelThicknessScale)
    if usePlyFractionDVs:
        DVScales += [defaultPlyFractionScale] * numPlies
    DVScales.append(defaultStiffenerHeightScale)
    DVScales.append(defaultStiffenerThicknessScale)
    if usePlyFractionDVs:
        DVScales += [defaultPlyFractionScale] * numPlies
    return DVScales


# Callback function used to setup TACS element objects and DVs
def element_callback(
    dvNum,
    compID,
    compDescript,
    elemDescripts,
    globalDVs,
    args,
    nonlinear,
    useComposite,
    usePanelLengthDVs,
    usePlyFractionDVs,
    useStiffenerPitchDVs,
    panelLengths=None,
    **kwargs,
):
    if args.oldSizingRules:
        stiffenerPitchMin = 0.05
        panelThicknessMin = 2e-3
        stiffenerHeightMin = 2e-3
        stiffenerThicknessMin = 2e-3
    else:
        stiffenerPitchMin = defaultStiffenerPitchMin
        panelThicknessMin = defaultPanelThicknessMin
        stiffenerHeightMin = defaultStiffenerHeightMin
        stiffenerThicknessMin = defaultStiffenerThicknessMin

    if useComposite:
        prop = constitutive.MaterialProperties(
            rho=compositeProperties["rho"],
            E1=compositeProperties["E11"],
            E2=compositeProperties["E22"],
            G12=compositeProperties["G12"],
            G13=compositeProperties["G13"],
            G23=compositeProperties["G23"],
            nu12=compositeProperties["nu12"],
            T1=compositeProperties["T1"],
            C1=compositeProperties["C1"],
            T2=compositeProperties["T2"],
            C2=compositeProperties["C2"],
            S12=compositeProperties["S12"],
        )
        ply = constitutive.OrthotropicPly(1.0, prop)

        # Use a 0-deg biased layup for the skin and a +-45-deg biased layup spars and ribs
        if "SKIN" in compDescript:
            plyAngles = skinPlyAngles
            panelPlyFractions = skinPlyFracs
        else:
            plyAngles = sparRibPlyAngles
            panelPlyFractions = sparRibPlyFracs

        # Always use the 0-deg biased layup for the stiffeners
        stiffenerPlyFractions = skinPlyFracs
    else:
        # Setup (isotropic) property and constitutive objects
        prop = constitutive.MaterialProperties(
            rho=aluProperties["rho"],
            E=aluProperties["E"],
            nu=aluProperties["nu"],
            ys=aluProperties["ys"],
        )
        ply = constitutive.OrthotropicPly(1.0, prop)
        plyAngles = np.zeros(1)
        panelPlyFractions = stiffenerPlyFractions = np.ones(1)
    numPlies = len(plyAngles)

    # ==============================================================================
    # Blade stiffened shell constitutive model setup
    # ==============================================================================
    # Figure out the reference axis and panel length
    if "SKIN" in compDescript:
        isInboard = any([name in compDescript for name in ["SKIN.000", "SKIN.001", "SKIN.002"]])
        if isInboard:
            refAxis = np.array([0.0, 0.0, 0.0])
            refAxis[spanIndex] = 1.0
            panelLength = inboardRibSpacing
        else:
            refAxis = TESparDirection
            panelLength = outboardRibSpacing
    else:
        refAxis = np.array([0.0, 0.0, 0.0])
        refAxis[verticalIndex] = 1.0
        panelLength = 0.5  # This is a conservative estimate of the panel length for the ribs and spars, based on the depth of the leading edge spar at the wing root
    if panelLengths is not None:
        panelLength = panelLengths[compDescript]

    currDVNum = dvNum
    DVScales = []
    if usePanelLengthDVs:
        panelLengthNum = currDVNum
        DVScales.append(panelLengthScale)
        currDVNum += 1
    else:
        panelLengthNum = -1

    if useStiffenerPitchDVs:
        if "RIB" in compDescript:
            stiffenerPitchNum = currDVNum
            stiffenerPitch = defaultStiffenerPitch
            stiffenerPitchMax = defaultStiffenerPitchMax
            DVScales.append(defaultStiffenerPitchScale)
            currDVNum += 1
        else:
            if "SPAR.00" in compDescript:
                pitchVarName = "le_spar_stiffenerPitch"
            elif "SPAR.01" in compDescript:
                pitchVarName = "te_spar_stiffenerPitch"
            elif "U_SKIN" in compDescript:
                pitchVarName = "u_skin_stiffenerPitch"
            elif "L_SKIN" in compDescript:
                pitchVarName = "l_skin_stiffenerPitch"

            stiffenerPitchNum = globalDVs[pitchVarName]["num"]
            stiffenerPitch = globalDVs[pitchVarName]["value"]
            stiffenerPitchMin = globalDVs[pitchVarName]["lowerBound"]
            stiffenerPitchMax = globalDVs[pitchVarName]["upperBound"]
    else:
        stiffenerPitchNum = -1
        stiffenerPitch = defaultStiffenerPitch
        stiffenerPitchMin = stiffenerPitchMax = defaultStiffenerPitch

    panelThicknessNum = currDVNum
    DVScales.append(defaultPanelThicknessScale)
    currDVNum += 1

    if usePlyFractionDVs:
        panelPlyFracNums = np.arange(currDVNum, currDVNum + numPlies).astype(np.intc)
        DVScales += [defaultPlyFractionScale] * numPlies
        currDVNum += numPlies
    else:
        panelPlyFracNums = -np.ones(numPlies).astype(np.intc)

    stiffenerHeightNum = currDVNum
    DVScales.append(defaultStiffenerHeightScale)
    currDVNum += 1

    stiffenerThicknessNum = currDVNum
    DVScales.append(defaultStiffenerThicknessScale)
    currDVNum += 1

    if usePlyFractionDVs:
        stiffenerPlyFracNums = np.arange(currDVNum, currDVNum + numPlies).astype(np.intc)
        DVScales += [defaultPlyFractionScale] * numPlies
        currDVNum += numPlies
    else:
        stiffenerPlyFracNums = -np.ones(numPlies).astype(np.intc)

    con = constitutive.BladeStiffenedShellConstitutive(
        panelPly=ply,
        stiffenerPly=ply,
        kcorr=kcorr,
        panelLength=TACS.dtype(panelLength),
        panelLengthNum=panelLengthNum,
        stiffenerPitch=TACS.dtype(stiffenerPitch),
        stiffenerPitchNum=stiffenerPitchNum,
        panelThick=TACS.dtype(defaultPanelThickness),
        panelThickNum=panelThicknessNum,
        panelPlyAngles=plyAngles.astype(TACS.dtype),
        panelPlyFracs=panelPlyFractions.astype(TACS.dtype),
        panelPlyFracNums=panelPlyFracNums,
        stiffenerHeight=TACS.dtype(defaultStiffenerHeight),
        stiffenerHeightNum=stiffenerHeightNum,
        stiffenerThick=TACS.dtype(defaultStiffenerThickness),
        stiffenerThickNum=stiffenerThicknessNum,
        stiffenerPlyAngles=plyAngles.astype(TACS.dtype),
        stiffenerPlyFracs=stiffenerPlyFractions.astype(TACS.dtype),
        stiffenerPlyFracNums=stiffenerPlyFracNums,
        flangeFraction=flangeFraction,
    )
    if usePlyFractionDVs:
        con.setPanelPlyFractionBounds(
            defaultPlyFractionMin * np.ones(numPlies), defaultPlyFractionMax * np.ones(numPlies)
        )
        con.setStiffenerPlyFractionBounds(
            defaultPlyFractionMin * np.ones(numPlies), defaultPlyFractionMax * np.ones(numPlies)
        )
    con.setStiffenerPitchBounds(stiffenerPitchMin, stiffenerPitchMax)
    con.setPanelThicknessBounds(panelThicknessMin, defaultPanelThicknessMax)
    con.setStiffenerHeightBounds(stiffenerHeightMin, defaultStiffenerHeightMax)
    con.setStiffenerThicknessBounds(stiffenerThicknessMin, defaultStiffenerThicknessMax)

    transform = elements.ShellRefAxisTransform(refAxis)

    if not nonlinear:
        if elemDescripts[0] == "CQUAD4":
            elem = elements.Quad4Shell(transform, con)
        elif elemDescripts[0] == "CQUAD9":
            elem = elements.Quad9Shell(transform, con)
        elif elemDescripts[0] == "CQUAD16":
            elem = elements.Quad16Shell(transform, con)
    else:
        if elemDescripts[0] == "CQUAD4":
            elem = elements.Quad4NonlinearShellModRot(transform, con)
        elif elemDescripts[0] == "CQUAD9":
            elem = elements.Quad9NonlinearShellModRot(transform, con)
        elif elemDescripts[0] == "CQUAD16":
            elem = elements.Quad16NonlinearShellModRot(transform, con)

    return elem, DVScales


def setup_tacs_assembler(fea_assembler, args):
    # Add global DVs for the skin and spar stiffener pitch
    if args.useStiffPitchDVs:
        if args.oldSizingRules:
            stiffenerPitchMin = 0.05
        else:
            stiffenerPitchMin = defaultStiffenerPitchMin
        for stiffenerGroup in ["u_skin", "l_skin", "le_spar", "te_spar"]:
            fea_assembler.addGlobalDV(
                f"{stiffenerGroup}_stiffenerPitch",
                defaultStiffenerPitch,
                lower=stiffenerPitchMin,
                upper=defaultStiffenerPitchMax,
                scale=defaultStiffenerPitchScale,
            )


def problem_setup(
    scenario_name,
    flightPoint,
    fea_assembler,
    problem,
    args,
    options=None,
    newtonOptions=None,
    continuationOptions=None,
):
    """
    Helper function to add fixed forces and eval functions
    to structural problems used in tacs builder
    """

    if options is not None:
        problem.setOptions(options)
    if continuationOptions is not None:
        problem.nonlinearSolver.setOptions(continuationOptions)
    if newtonOptions is not None:
        problem.nonlinearSolver.innerSolver.setOptions(newtonOptions)

    # Add TACS Functions
    problem.addFunction("mass", functions.StructuralMass)
    for massGroup in ["spar", "u_skin", "l_skin", "rib"]:
        compIDs = fea_assembler.selectCompIDs(include=massGroup.upper())
        problem.addFunction(
            f"{massGroup}_mass",
            functions.StructuralMass,
            compIDs=compIDs,
        )
    if flightPoint.failureGroups is not None:
        for group in flightPoint.failureGroups:
            compIDs = fea_assembler.selectCompIDs(include=group.upper())
            problem.addFunction(
                f"{group}_ksFailure",
                functions.KSFailure,
                compIDs=compIDs,
                safetyFactor=1.5,
                ksWeight=args.ksWeight,
                ftype=args.ksType,
            )
            problem.addFunction(
                f"{group}_maxFailure",
                functions.KSFailure,
                compIDs=compIDs,
                safetyFactor=1.5,
                ksWeight=1e20,
            )
            problem.addFunction(
                f"{group}_mass",
                functions.StructuralMass,
                compIDs=compIDs,
            )
    problem.addFunction("compliance", functions.Compliance)

    # Add gravity load
    g = np.zeros(3)
    g[verticalIndex] = -9.81
    problem.addInertialLoad(g * flightPoint.loadFactor)


def setupConstraints(scenario_name, fea_assembler, constraints, args):
    usePanelLengthDVs = args.usePanelLengthDVs
    usePlyFractionDVs = args.usePlyFractionDVs
    useStiffenerPitchDVs = args.useStiffPitchDVs
    thicknessAdjCon = args.thicknessAdjCon
    heightAdjCon = args.heightAdjCon
    stiffAspectMax = args.stiffAspectMax
    stiffAspectMin = args.stiffAspectMin

    compIDs = {}
    for group in ["SPAR", "U_SKIN", "L_SKIN", "RIB"]:
        compIDs[group] = fea_assembler.selectCompIDs(include=group.upper())

    DVNums = computeDVNums(
        0,
        usePanelLengthDVs=usePanelLengthDVs,
        usePlyFractionDVs=usePlyFractionDVs,
        useStiffenerPitchDVs=args.useStiffPitchDVs,
        numPlies=4 if args.useComposite else 1,
    )

    # Add adjacency constraints on panel thickness, stiffener thickness and stiffener height to everything but the ribs
    adjCon = fea_assembler.createAdjacencyConstraint("AdjCon")
    for group in ["SPAR", "U_SKIN", "L_SKIN"]:
        adjCon.addConstraint(
            conName=group + "_panelThicknessAdj",
            compIDs=compIDs[group],
            lower=-thicknessAdjCon,
            upper=thicknessAdjCon,
            dvIndex=DVNums["panelThickness"],
        )
        adjCon.addConstraint(
            conName=group + "_stiffenerThicknessAdj",
            compIDs=compIDs[group],
            lower=-thicknessAdjCon,
            upper=thicknessAdjCon,
            dvIndex=DVNums["stiffenerThickness"],
        )
        adjCon.addConstraint(
            conName=group + "_stiffenerHeightAdj",
            compIDs=compIDs[group],
            lower=-heightAdjCon,
            upper=heightAdjCon,
            dvIndex=DVNums["stiffenerHeight"],
        )
    constraints.append(adjCon)

    # Add constraints between the DV's on each panel
    dvCon = fea_assembler.createDVConstraint("DVCon")

    if args.oldSizingRules:
        # Limit the difference in thickness between the panel and stiffener
        # -thickDiffMax <= panelThickness - stiffenerThickness <= thickDiffMax
        dvCon.addConstraint(
            conName="thickDiffLimit",
            lower=-2.5,
            upper=2.5,
            dvIndices=[DVNums["panelThickness"], DVNums["stiffenerThickness"]],
            dvWeights=[1.0, -1.0],
        )
        # Limit the aspect ratio of the stiffener
        # stiffenerHeight - stiffAspectMax * stiffenerThickness <= 0
        dvCon.addConstraint(
            conName="stiffenerAspectMax",
            upper=0.0,
            dvIndices=[DVNums["stiffenerHeight"], DVNums["stiffenerThickness"]],
            dvWeights=[1.0, -stiffAspectMax],
        )
        # stiffenerHeight - stiffAspectMin * stiffenerThickness >= 0
        dvCon.addConstraint(
            conName="stiffenerAspectMin",
            lower=0.0,
            dvIndices=[DVNums["stiffenerHeight"], DVNums["stiffenerThickness"]],
            dvWeights=[1.0, -stiffAspectMin],
        )
        # Ensure there is space between the stiffeners
        # 2*flangeFraction - stiffenerPitch <= 0
        dvCon.addConstraint(
            conName="stiffSpacingMin",
            upper=0.0,
            dvIndices=[DVNums["stiffenerHeight"], DVNums["stiffenerPitch"]],
            dvWeights=[2.0, -1.0],
        )
    else:
        # Flange thickness should be at least 1.5x skin thickness to reduce stress concentrations at the flange-skin
        # interface, but no more than 15x skin thickness
        # dvCon.addConstraint(
        #     conName="flangeThicknessMin",
        #     lower=0.0,
        #     dvIndices=[DVNums["panelThickness"], DVNums["stiffenerThickness"]],
        #     dvWeights=[-1.5, 1.0],
        # )
        dvCon.addConstraint(
            conName="flangeThicknessMax",
            upper=0.0,
            dvIndices=[DVNums["panelThickness"], DVNums["stiffenerThickness"]],
            dvWeights=[-15.0, 1.0],
        )

        # Limit the aspect ratio of the stiffener
        # stiffenerHeight - stiffAspectMax * stiffenerThickness <= 0
        dvCon.addConstraint(
            conName="stiffenerAspectMax",
            upper=0.0,
            dvIndices=[DVNums["stiffenerHeight"], DVNums["stiffenerThickness"]],
            dvWeights=[1.0, -stiffAspectMax],
        )
        # stiffenerHeight - stiffAspectMin * stiffenerThickness >= 0
        dvCon.addConstraint(
            conName="stiffenerAspectMin",
            lower=0.0,
            dvIndices=[DVNums["stiffenerHeight"], DVNums["stiffenerThickness"]],
            dvWeights=[1.0, -stiffAspectMin],
        )
        # Spacing between stiffeners should be greater than stiffener flange width to avoid overlapping stiffeners
        # flangeFraction*stiffenerHeight - stiffenerPitch <= 0
        # NOTE: This constrain is actually not strictly right now necessary because the stiffener height upper bound and
        # the stiffener pitch lower bound are both 0.15m so the constraint is always satisfied. However, I'm keeping it
        # enabled for now in case the bounds change in the future.
        if useStiffenerPitchDVs:
            dvCon.addConstraint(
                conName="stiffSpacingMin",
                upper=0.0,
                dvIndices=[DVNums["stiffenerHeight"], DVNums["stiffenerPitch"]],
                dvWeights=[flangeFraction, -1.0],
            )
        else:
            dvCon.addConstraint(
                conName="stiffSpacingMin",
                upper=defaultStiffenerPitch,
                dvIndices=[DVNums["stiffenerHeight"]],
                dvWeights=[flangeFraction],
            )
        constraints.append(dvCon)

    if usePanelLengthDVs:
        panelLengthCon = fea_assembler.createPanelLengthConstraint("PanelLengthCon")
        panelLengthCon.addConstraint("PanelLength", dvIndex=DVNums["panelLength"])
        constraints.append(panelLengthCon)


def buildStructDVDictMap(assembler, args):
    """Build a dictionary that contains the design variable numbers for each component

    This function assumes that each component has 5 design variables:

    - panel length
    - stiffener pitch
    - panel thickness
    - stiffener height
    - stiffener thickness

    Parameters
    ----------
    assembler : pyTACS assembler
        pyTACS assembler object for the problem

    Returns
    -------
    dict[str, dict[str, int]]
        A dictionary whose keys are component names and whose value is another dictionary with the keys being the design
        variable names and the values being the design variable numbers
    """
    dvMap = {}
    for ii in range(assembler.nComp):
        compName = assembler.compDescripts[ii]
        element = assembler.meshLoader.getElementObject(ii, 0)
        dvNums = element.getDesignVarNums(0)
        dvInds = computeDVNums(
            0,
            usePanelLengthDVs=args.usePanelLengthDVs,
            usePlyFractionDVs=args.usePlyFractionDVs,
            useStiffenerPitchDVs=args.useStiffPitchDVs,
            numPlies=4 if args.useComposite else 1,
        )
        dvMap[compName] = {}
        for dvName, dvNum in dvInds.items():
            # Need to check if dvNum is an iterable
            if isinstance(dvNum, Iterable):
                dvMap[compName][dvName] = [dvNums[dvN] if dvN >= 0 else -1 for dvN in dvNum]
            else:
                dvMap[compName][dvName] = dvNums[dvNum] if dvNum >= 0 else -1

    return dvMap
