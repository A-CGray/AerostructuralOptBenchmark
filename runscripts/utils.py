# ==============================================================================
# Standard Python modules
# ==============================================================================
import re
import os
import sys

# ==============================================================================
# External Python modules
# ==============================================================================
import dill
import openmdao.api as om
from mpi4py import MPI
import reverse_argparse
from pyoptsparse import History
from scipy.sparse import coo_matrix
import numpy as np

# ==============================================================================
# Extension modules
# ==============================================================================
THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_FILE_DIR, "../AircraftSpecs"))
from STWFlightPoints import flightPointSets  # noqa: E402
from STWSpecs import aircraftSpecs  # noqa: E402

sys.path.append(os.path.join(THIS_FILE_DIR, "../geometry"))
from wingGeometry import wingGeometry  # noqa: E402


def getGeometryData():
    return wingGeometry


def getAircraftSpecs():
    return aircraftSpecs


def getFlightPointSet(name: str) -> list:
    return flightPointSets[name]


def getAeroMeshPath(level: int) -> str:
    return os.path.join(THIS_FILE_DIR, f"../aero/wing_vol_L{level}.cgns")


def getStructMeshPath(level: int, order: int) -> str:
    return os.path.join(THIS_FILE_DIR, f"../struct/wingbox-L{level}-Order{order}.bdf")


def getFFDPath(level: str):
    return os.path.join(THIS_FILE_DIR, f"../geometry/wing-ffd-advanced-{level}.xyz")


def saveRunCommand(parser, args, outputDir):
    """Save the command used to run this script in the output directory

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The parser used to parse the command line arguments
    args : argparse.Namespace
        The parsed command line arguments
    outputDir : str
        The directory to save the command in
    """
    if MPI.COMM_WORLD.rank == 0:
        print("\n" * 10)
        print("===============================================================================")
        print("To recreate this job, run the following command:")
        unparser = reverse_argparse.ReverseArgumentParser(parser, args)
        command = unparser.get_pretty_command_line_invocation()
        print(command)
        print("===============================================================================")
        print("\n" * 10)

        with open(os.path.join(outputDir, "runCommand.txt"), "w") as f:
            f.write(command)


def setValsFromFiles(files, prob):
    model = prob.model
    if prob.comm.rank == 0:
        print("\n===============================================================================")
    for fileName in files:
        if ".sql" in fileName:
            prevRecorder = om.CaseReader(fileName)
            driver_cases = prevRecorder.list_cases("driver", recurse=False, out_stream=None)
            last_case = prevRecorder.get_case(driver_cases[-1])
            old_design_vars = last_case.get_design_vars(scaled=False)
        elif ".pkl" in fileName:
            with open(fileName, "rb") as f:
                old_design_vars = dill.load(f)
        elif ".hst" in fileName:
            hist = History(fileName)
            data = hist.getValues()
            old_design_vars = {key: data[key][-1] for key in data}
        else:
            raise ValueError("Unrecognised file type, setValsFromFiles only works with .sql, .pkl or .hst files")
        for dv in model.get_design_vars():
            # See if the abs name is in the file
            try:
                prob.set_val(dv, old_design_vars[dv])
                if prob.comm.rank == 0:
                    print(f"Setting {dv} from {fileName}")
            except KeyError:
                # If it's not then try the promoted name
                promName = get_prom_name(prob.model, dv)
                try:
                    prob.set_val(dv, old_design_vars[promName])
                    if prob.comm.rank == 0:
                        print(f"Setting {dv} from {fileName}")
                except KeyError:
                    pass
        # Also try setting anything from the file that looks like a dv
        for dv in old_design_vars:
            if "dvs." in dv and dv not in model.get_design_vars():
                try:
                    prob.set_val(dv, old_design_vars[dv])
                    if prob.comm.rank == 0:
                        print(f"Setting {dv} from {fileName}")
                except KeyError:
                    pass

    if prob.comm.rank == 0:
        print("===============================================================================\n")


def writeOutputs(prob, outputDir, fileName="Outputs"):
    """Write the outputs of an OpenMDAO model to a pickle

    Any outputs with more than 10000 elements will not be written to avoid storing states of large solvers etc

    Parameters
    ----------
    prob : OpenMDAO Problem
        The problem to write the outputs from
    outputDir : _type_
        _description_
    """
    outputs = prob.model.list_outputs(return_format="dict", print_arrays=False, excludes=["*_vol_coords", "*_states"])
    if prob.model.comm.rank == 0:
        print(outputs)
    outputData = {}
    for output in outputs:
        try:
            data = prob.get_val(output)
            if not hasattr(data, "__len__") or len(data) < 10000:
                outputData[output] = data
        except TypeError:
            pass
    if prob.model.comm.rank == 0:
        outFileName = os.path.join(outputDir, os.path.basename(fileName) + ".pkl")
        with open(outFileName, "wb") as f:
            dill.dump(outputData, f, protocol=-1)


def whereAmIRunning():
    """Figure out which computer you are currently running on so you can set some output folder paths

    Returns
    -------
    str
        "albert" if running on my machine Albert, "greatlakes" if running on Great Lakes, "stampede" if running on Stampede2
    """
    NASNodePattern = "r[0-9]{1,3}i[0-9]n[0-9]{1,3}$"
    uname = os.uname()[1]
    if uname.lower() == "albert":
        return "albert"
    elif "arc-ts.umich.edu" in uname:
        return "greatlakes"
    elif "stampede2.tacc.utexas.edu" in uname:
        return "stampede"
    elif "pfe" in uname or len(re.findall(NASNodePattern, uname)) > 0:
        return "hecc"


def getOutputDir():
    machineName = whereAmIRunning()
    parentDirs = {
        "albert": "/home/ali/BigBoi/ali/AerostructuralOptBenchmark",
        "greatlakes": "/scratch/jrram_root/jrram1/alachris/AerostructuralOptBenchmark",
        "stampede": "/work2/07488/acgray/stampede2/AerostructuralOptBenchmark",
        "hecc": "/nobackup/achris10/AerostructuralOptBenchmark",
    }
    if machineName in parentDirs:
        return parentDirs[machineName]
    else:
        return "Output"


# ==============================================================================
# Function for translating OpenMDAO optimisation problem to a pyOptSparse problem
# ==============================================================================
def get_prom_name(model, abs_name):
    abs2prom = model._var_abs2prom
    if abs_name in abs2prom["input"]:
        return abs2prom["input"][abs_name]
    elif abs_name in abs2prom["output"]:
        return abs2prom["output"][abs_name]
    else:
        return abs_name


def convertSensDict(openmdaoSensDict):
    """
    Convert the OpenMDAO sensitivity dictionary into the format expected by pyOptSparse

    OpenMDAO stores the derivative of y wrt x in sens[(y,x)], whereas pyoptsparse expects sens[y][x]
    """
    sensDict = {}
    for key, val in openmdaoSensDict.items():
        of = key[0]
        wrt = key[1]
        if of not in sensDict:
            sensDict[of] = {}
        sensDict[of][wrt] = val
    return sensDict


def addConstraintFromOpenMDAO(con, optProb, omProb, wrt=None):
    name = get_prom_name(omProb.model, con["source"])

    # Get the scaling factor if there is one
    if con["scaler"] is None:
        scale = 1.0
    else:
        scale = con["scaler"]

    # Get bounds
    # The bounds stored in the con are already scaled so we need to un-scale them
    if con["equals"] is not None:
        lb = ub = con["equals"] / scale
    else:
        lb = con["lower"] / scale
        ub = con["upper"] / scale

    # get size
    size = con["global_size"] if con["distributed"] else con["size"]

    linear = con["linear"]
    if linear:
        # If this constraint is linear we need to compute it's jacobian and transform it from the form:
        # lb <= Ax - b <= ub
        # to the form:
        # lb + b <= Ax <= ub + b
        conVals = omProb.get_val(name)
        offsets = -conVals.copy()
        jac = omProb.compute_totals(of=name, return_format="dict", debug_print=True)
        jac = jac[name]

        x = {}
        for dv in wrt:
            x[dv] = omProb.get_val(dv)

        sparseJac = {}
        for dv, subJac in jac.items():
            dvPromName = get_prom_name(omProb.model, dv)
            if dvPromName in wrt:
                sparseMat = coo_matrix(subJac)
                if len(sparseMat.data) != 0:
                    # TODO: May need to account for DV scaling here
                    sparseJac[dvPromName] = {
                        "coo": [sparseMat.row, sparseMat.col, sparseMat.data],
                        "shape": sparseMat.shape,
                    }

                    # Figure out the offset
                    # conVals = Ax - b
                    # b = Ax - conVals
                    offsets += sparseMat.dot(x[dvPromName])
        jac = sparseJac
        ub += offsets
        lb += offsets
    else:
        jac = None

    # Add the constraint
    optProb.addConGroup(name, size, lower=lb, upper=ub, scale=scale, wrt=wrt, jac=jac, linear=linear)


def getTipDisplacement(prob, fpName):
    # ==============================================================================
    # Extract tip displacement
    # ==============================================================================

    FEAAssembler = prob.model.FEAAssembler
    comm = prob.comm

    chordIndex = wingGeometry["chordIndex"]
    verticalIndex = wingGeometry["verticalIndex"]

    components = ["SPAR.00", "SPAR.01", "RIB.22", "U_SKIN", "L_SKIN"]
    nodes = {}
    for comp in components:
        compIDs = FEAAssembler.selectCompIDs(include=comp)
        nodes[comp] = set(FEAAssembler.getGlobalNodeIDsForComps(compIDs, nastranOrdering=False))

    # The node at the front upper corner of the tip rib is the one node that is common to the upper skin, the front spar and the tip rib
    frontUpperNodeGlobalID = list(nodes["U_SKIN"].intersection(nodes["RIB.22"]).intersection(nodes["SPAR.00"]))[0]

    # Similarly, the node at the rear upper corner of the tip rib is the one node that is common to the upper skin, the rear spar and the tip rib
    rearUpperNodeGlobalID = list(nodes["U_SKIN"].intersection(nodes["RIB.22"]).intersection(nodes["SPAR.01"]))[0]

    # Now do the same for the lower skin
    frontLowerNodeGlobalID = list(nodes["L_SKIN"].intersection(nodes["RIB.22"]).intersection(nodes["SPAR.00"]))[0]
    rearLowerNodeGlobalID = list(nodes["L_SKIN"].intersection(nodes["RIB.22"]).intersection(nodes["SPAR.01"]))[0]

    frontUpperNodeLocalID = FEAAssembler.meshLoader.getLocalNodeIDsFromGlobal(
        frontUpperNodeGlobalID, nastranOrdering=False
    )[0]
    rearUpperNodeLocalID = FEAAssembler.meshLoader.getLocalNodeIDsFromGlobal(
        rearUpperNodeGlobalID, nastranOrdering=False
    )[0]

    # To compute the tip rotation we need the node coordinates
    frontUpperCoord = FEAAssembler.meshLoader.getBDFNodes(frontUpperNodeGlobalID, nastranOrdering=False)
    rearUpperCoord = FEAAssembler.meshLoader.getBDFNodes(rearUpperNodeGlobalID, nastranOrdering=False)

    # Now retrieve the displacements at these nodes and compute the overall tip displacement and rotation
    disp = prob.model.get_val(f"{fpName}.solver.u_struct", get_remote=False)
    frontUpperDisp = None
    rearUpperDisp = None
    if frontUpperNodeLocalID != -1:
        frontUpperDisp = disp[6 * frontUpperNodeLocalID : 6 * frontUpperNodeLocalID + 3]
    if rearUpperNodeLocalID != -1:
        rearUpperDisp = disp[6 * rearUpperNodeLocalID : 6 * rearUpperNodeLocalID + 3]

    # broadcast front and rear upper displacements to all procs
    hasFrontDisp = comm.allgather(frontUpperDisp is not None)
    hasRearDisp = comm.allgather(rearUpperDisp is not None)
    frontUpperDisp = comm.bcast(frontUpperDisp, root=np.argmax(hasFrontDisp))
    rearUpperDisp = comm.bcast(rearUpperDisp, root=np.argmax(hasRearDisp))

    # Compute the tip twist as the change in the angle of the line in the XZ plane between the front and rear upper nodes
    x1 = frontUpperCoord[chordIndex]
    z1 = frontUpperCoord[verticalIndex]
    dx1 = frontUpperDisp[chordIndex]
    dz1 = frontUpperDisp[verticalIndex]
    x2 = rearUpperCoord[chordIndex]
    z2 = rearUpperCoord[verticalIndex]
    dx2 = rearUpperDisp[chordIndex]
    dz2 = dz2

    tipZDisp = (dz1 + dz2) / 2

    tipTwist = np.rad2deg(np.arctan2((z2 + dz2) - (z1 + dz1), (x2 + dx2) - (x1 + dx1)) - np.arctan2(z2 - z1, x2 - x1))

    return tipZDisp, tipTwist
