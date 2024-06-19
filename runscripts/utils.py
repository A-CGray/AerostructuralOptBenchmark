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

# ==============================================================================
# Extension modules
# ==============================================================================
THIS_FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(THIS_FILE_DIR, "../AircraftSpecs"))
from STWFlightPoints import flightPointSets
from STWSpecs import aircraftSpecs

sys.path.append(os.path.join(THIS_FILE_DIR, "../geometry"))
from wingGeometry import wingGeometry


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


def buildStructDVDictMap(assembler):
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
        dvMap[compName] = {
            "panel length": dvNums[0],
            "stiffener pitch": dvNums[1],
            "panel thickness": dvNums[2],
            "stiffener height": dvNums[3],
            "stiffener thickness": dvNums[4],
        }

    return dvMap


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
            try:
                prob.set_val(dv, old_design_vars[dv])
                if prob.comm.rank == 0:
                    print(f"Setting {dv} from {fileName}")
            except KeyError:
                pass
        # Also try setting anything from dvs
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
    """  #
    outputs = prob.model.list_outputs(
        return_format="dict", print_arrays=False, excludes=["*adflow_vol_coords", "*adflow_states"]
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
