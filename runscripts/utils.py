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
    outputs = prob.model.list_outputs(
        return_format="dict", print_arrays=False, excludes=["*_vol_coords", "*_states"]
    )
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
