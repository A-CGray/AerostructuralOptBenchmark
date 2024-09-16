"""
==============================================================================
Command line arguments
==============================================================================
@File    :   CommonArgs.py
@Date    :   2023/04/27
@Author  :   Alasdair Christison Gray
@Description : This file creates a parser and defines all the command line
arguments that are common between the structural and aerostructural scripts
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import argparse

parser = argparse.ArgumentParser()

# --- General options ---
parser.add_argument("--output", type=str, default="debug", help="Output directory")
parser.add_argument("--noFiles", action="store_true", help="Flag to turn off the writing of solution files")
parser.add_argument("--noRecorder", action="store_true", help="Flag to turn off the the use of the OpenMDAO recorder")
parser.add_argument(
    "--initDVs",
    type=str,
    nargs="*",
    default=[],
    help="Files from which to set DVs used during initialisation, some MPhys components perform some initialisation the first time their compute method is called, these DVs will be set when that happens. If you have DVs you would like to use for your optimisation/analysis from but not use for initialisation, pass those to the `--postInitDVs` options. You can pass multiple paths in which case later files will overwrite DVs from earlier files",
)
parser.add_argument(
    "--postInitDVs",
    type=str,
    nargs="*",
    default=[],
    help="Similar to `initDVs` except that these DVs will be set after an initial call of `run_model` so that they're not used during initialisation. You can pass multiple paths in which case later files will overwrite DVs from earlier files",
)

# --- Solver options ---
parser.add_argument(
    "--tolFactor", type=float, default=1.0, help="Factor to scale the default convergence tolerances by"
)

# --- Optimiser Options ---
parser.add_argument(
    "--optimiser",
    type=str,
    default="snopt",
    choices=["snopt", "ipopt", "nlpqlp", "paroptsl1", "paroptfilter", "paroptmma"],
)
parser.add_argument("--optIter", type=int, default=1000)
parser.add_argument("--hessianUpdate", type=int, default=40)
parser.add_argument("--initPenalty", type=float, default=0.0)
parser.add_argument("--stepLimit", type=float, default=0.02)
parser.add_argument("--feasibility", type=float, default=1e-6)
parser.add_argument("--optimality", type=float, default=1e-6)
parser.add_argument("--timeLimit", type=float, default=None, help="Time limit for optimization, in s")
parser.add_argument(
    "--structScalingFactor",
    type=float,
    default=1.0,
    help="Factor by which to scale all struct DVs, on top of whatever is specified in the element callback",
)
parser.add_argument(
    "--geoScalingFactor",
    type=float,
    default=1.0,
    help="Factor by which to scale all geometric DVs, on top of whatever is specified in the element callback",
)

# --- Structure options ---
parser.add_argument("--nonlinear", action="store_true", help="Use nonlinear structural analysis")
parser.add_argument(
    "--addStructDVs",
    action="store_true",
    help="Whether to add structural design variables as OpenMDAO design variables",
)
parser.add_argument("--useComposite", action="store_true", help="Use composite material", default=True)
parser.add_argument(
    "--useStiffPitchDVs",
    action="store_true",
    help="Enable stiffener pitch design variables",
)
parser.add_argument(
    "--usePanelLengthDVs",
    action="store_true",
    help="Enable panel length design variables to keep panel lengths up to date with true geometry",
    default=True,
)
parser.add_argument("--usePlyFractionDVs", action="store_true", help="Enable composite ply fraction DVs")
parser.add_argument("--structLevel", type=int, default=3, choices=[1, 2, 3, 4])
parser.add_argument("--structOrder", type=int, default=2, choices=[2, 3, 4])
parser.add_argument("--oldSizingRules", action="store_true", help="Use old sizing rules")
parser.add_argument(
    "--thicknessAdjCon",
    type=float,
    default=2.5e-3,
    help="Thickness adjacency constraint limit",
)
parser.add_argument(
    "--heightAdjCon",
    type=float,
    default=1e-2,
    help="Stiffener height adjacency constraint limit",
)
parser.add_argument(
    "--stiffAspectMax",
    type=float,
    default=30.0,
    help="Maximum allowable stiffener aspect ratio (height/thickness)",
)
parser.add_argument(
    "--stiffAspectMin",
    type=float,
    default=5.0,
    help="Minimum allowable stiffener aspect ratio (height/thickness)",
)
parser.add_argument(
    "--ksWeight",
    type=float,
    default=100.0,
    help="Weight used for KS aggregation of failure constraints",
)
parser.add_argument(
    "--ksType",
    type=str,
    choices=["continuous", "discrete"],
    default="continuous",
    help="Type of KS aggregation to use for failure constraints",
)

# --- Geometric parameterisation options ---
parser.add_argument(
    "--addGeoDVs",
    action="store_true",
    help="Whether to add the geometric design variables as OpenMDAO design variables",
)
parser.add_argument("--ffdLevel", type=str, default="coarse", choices=["coarse", "med", "fine"])
parser.add_argument("--twist", action="store_true")
parser.add_argument("--shape", action="store_true")
parser.add_argument("--sweep", action="store_true")
parser.add_argument("--span", action="store_true")
parser.add_argument("--dihedral", action="store_true")
parser.add_argument("--taper", action="store_true")
