"""
==============================================================================
Structural runscript
==============================================================================
@File    :   mphys_struct.py
@Date    :   2023/03/31
@Author  :   Alasdair Christison Gray
@Description :
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import sys

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from mpi4py import MPI
import openmdao.api as om
from mphys import Multipoint
from mphys.scenario_structural import ScenarioStructural
from tacs.mphys import TacsBuilder
from tacs import TACS
import tacs
from pygeo.mphys import OM_DVGEOCOMP
import dill  # A better version of pickle

# ==============================================================================
# Extension modules
# ==============================================================================
from SETUP.setupDVGeo import setupDVGeo
import SETUP.setupTACS as setupTACS
from OptimiserOptions import getOptOptions
from CommonArgs import parser

from utils import (
    getOutputDir,
    setValsFromFiles,
    writeOutputs,
    saveRunCommand,
    getStructMeshPath,
    getFFDPath,
)

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../AircraftSpecs"))
from STWFlightPoints import flightPointSets  # noqa: E402

# set these for convenience
comm = MPI.COMM_WORLD
rank = comm.rank
isComplex = TACS.dtype == complex

# --- General options ---
parser.add_argument(
    "--task",
    type=str,
    default="analysis",
    choices=["check", "analysis", "derivCheck", "opt"],
    help="Task to run",
)
parser.add_argument("--flightPointSet", type=str, default="maneuverOnly", choices=list(flightPointSets.keys()))

# --- Optimiser Options ---
parser.add_argument(
    "--optType",
    type=str,
    default="minMass",
    choices=["minComp", "minMass"],
    help="Optimisation type: 'minComp' for compliance minimisation with a mass constraint, 'minMass' for mass minimisation with a failure constraint",
)

args = parser.parse_args()

if args.task == "opt" and not args.addStructDVs and not args.addGeoDVs:
    raise ValueError("You must specify at least one of --addStructDVs or --addGeoDVs to run an optimisation")

if args.task == "derivCheck":
    args.addGeoDVs = True
    args.twist = True
    args.addStructDVs = True


# --- Figure out where to put the results ---
outputDir = os.path.join(getOutputDir(), args.output)
resultsDir = os.path.join(outputDir, "Results")

# Create output directories
os.makedirs(outputDir, exist_ok=True)
os.makedirs(resultsDir, exist_ok=True)

# Print out the full list of command line arguments
saveRunCommand(parser, args, outputDir)

# Define location of input files
structMeshFile = getStructMeshPath(level=args.structLevel, order=args.structOrder)
ffdFile = getFFDPath(level=args.ffdLevel)

# Define flight points to use
flightPoints = flightPointSets[args.flightPointSet]
flightPointsDict = {fp.name: fp for fp in flightPoints}

# Read in panel lengths, they're stored in a csv file along with the struct meshes
panelLengthFileName = os.path.join(os.path.dirname(structMeshFile), "PanelLengths.csv")
with open(panelLengthFileName, "r") as panelLengthFile:
    panelLengths = {line.split(",")[0]: float(line.split(",")[1]) for line in panelLengthFile}


class Top(Multipoint):
    def setup(self):
        ################################################################################
        # TACS options
        ################################################################################
        problemOptions = {
            "outputDir": resultsDir,
            "useMonitor": not args.nonlinear,
            "monitorFrequency": 1,
            "L2convergence": 1e-20,
            "L2convergenceRel": 1e-20,
        }
        nonlinearOptions = {"writeNLIterSolutions": False, "printLevel": 1}
        newtonOptions = None
        continuationOptions = None
        if args.nonlinear:
            problemOptions.update(nonlinearOptions)
            newtonOptions = {
                "UseEW": True,
                "MaxLinIters": 10,
                "skipFirstNLineSearch": 1,
                "ForceFirstIter": True,
                "maxIter": 20,
            }
            continuationOptions = {
                "RelTol": 1e-7,
                "UsePredictor": True,
                "NumPredictorStates": 4,
                "maxIter": 10,
            }

        def setup_assembler(assembler):
            return setupTACS.setup_tacs_assembler(assembler, args)

        def setup_tacs_problem(scenario_name, fea_assembler, problem):
            flightPoint = flightPointsDict[scenario_name]
            setupTACS.problem_setup(
                scenario_name,
                flightPoint,
                fea_assembler,
                problem,
                args=args,
                options=problemOptions,
                newtonOptions=newtonOptions,
                continuationOptions=continuationOptions,
            )

            # Apply a uniform pressure over the lower skin, 30 kPa is almost exactly the right value to get a vertical
            # load equivalent to 2.5g
            lSkinCompIDs = fea_assembler.selectCompIDs("L_SKIN")
            problem.addPressureToComponents(lSkinCompIDs, -flightPoint.loadFactor * 30e3 / 2.5)

        def element_callback(dvNum, compID, compDescript, elemDescripts, specialDVs, **kwargs):
            return setupTACS.element_callback(
                dvNum,
                compID,
                compDescript,
                elemDescripts,
                specialDVs,
                args,
                nonlinear=args.nonlinear,
                useComposite=args.useComposite,
                usePanelLengthDVs=args.usePanelLengthDVs,
                usePlyFractionDVs=args.usePlyFractionDVs,
                useStiffenerPitchDVs=args.useStiffPitchDVs,
                panelLengths=panelLengths,
                **kwargs,
            )

        def constraint_callback(scenario_name, fea_assembler, constraints):
            if scenario_name == flightPoints[0].name:
                return setupTACS.setupConstraints(scenario_name, fea_assembler, constraints, args)
            return None

        for ii, flightPoint in enumerate(flightPoints):
            scenarioName = flightPoint.name

            struct_builder = TacsBuilder(
                mesh_file=structMeshFile,
                assembler_setup=setup_assembler,
                element_callback=element_callback,
                constraint_setup=constraint_callback,
                problem_setup=setup_tacs_problem,
                coupled=False,
                write_solution=not args.noFiles,
            )
            struct_builder.initialize(self.comm)
            structDVMap = setupTACS.buildStructDVDictMap(struct_builder.get_fea_assembler(), args)
            if rank == 0:
                with open(os.path.join(outputDir, "structDVMap.pkl"), "wb") as structDVMapFile:
                    dill.dump(structDVMap, structDVMapFile)

            ################################################################################
            # MPHYS setup
            ################################################################################
            if ii == 0:
                # ivc to keep the top level DVs
                dvSys = self.add_subsystem("dvs", om.IndepVarComp(), promotes=["*"])
                # add the structural DVs
                init_dvs = struct_builder.get_initial_dvs()
                dvSys.add_output("dv_struct", init_dvs)
                if args.addStructDVs:
                    lb, ub = struct_builder.get_dv_bounds()
                    structDVScaling = np.array(struct_builder.fea_assembler.scaleList)
                    self.add_design_var(
                        "dv_struct", lower=lb, upper=ub, scaler=structDVScaling * args.structScalingFactor
                    )

                self.add_subsystem("mesh", struct_builder.get_mesh_coordinate_subsystem())

                # Create the geometry component, we dont need a builder because we do it here.
                geometrySys = self.add_subsystem(
                    "geometry",
                    OM_DVGEOCOMP(file=ffdFile, type="ffd", options={"isComplex": isComplex}),
                )
                # Tell the geometry component that there will be a set of coordinates for the structural discipline
                geometrySys.nom_add_discipline_coords("struct")
                # Connect the original structural mesh coordinates as an input to the geometry component
                self.connect("mesh.x_struct0", "geometry.x_struct_in")

            # this is the method that needs to be called for every point in this mp_group
            self.mphys_add_scenario(scenarioName, ScenarioStructural(struct_builder=struct_builder))
            self.mphys_connect_scenario_coordinate_source(
                "geometry", scenarioName, "struct"
            )  # This is equivalent to `self.connect("geometry.x_struct0", f"{scenarioName}.x_struct0")`

            self.connect("dv_struct", f"{scenarioName}.dv_struct")

        # For the compliance minimisation problem, we need to add a component to sum the compliance from each point
        if args.optType.lower() == "mincomp":
            complianceFuncs = [f"compliance_{ii}" for ii in range(len(flightPoints))]
            totalCompFunc = om.AddSubtractComp(input_names=complianceFuncs, output_name="totalCompliance")
            self.add_subsystem("totalCompliance", totalCompFunc, promotes=["totalCompliance"])
            for ii, fp in enumerate(flightPoints):
                self.connect(f"{fp.name}.compliance", f"totalCompliance.compliance_{ii}")

    def configure(self):
        # Setup the geometric DVs
        setupDVGeo(
            args,
            self,
            self.geometry,
            self.dvs,
            dvScaleFactor=args.geoScalingFactor,
            geoCompName="geometry",
            addGeoDVs=args.addGeoDVs,
        )

        # Add TACS constraints
        firstScenario = self.__getattribute__(flightPoints[0].name)
        if hasattr(firstScenario.struct_post, "constraints"):
            for system in firstScenario.struct_post.constraints.system_iter():
                constraint = system.constr
                constraintFuncNames = constraint.getConstraintKeys()
                bounds = {}
                constraint.getConstraintBounds(bounds)
                for conName in constraintFuncNames:
                    if self.comm.rank == 0:
                        print("Adding constraint: ", conName)
                    name = f"{flightPoints[0].name}.{system.name}.{conName}"
                    if isinstance(constraint, tacs.constraints.panel_length.PanelLengthConstraint):
                        self.add_constraint(name, equals=0.0, scaler=1.0)
                    else:
                        lb = bounds[f"{system.name}_{conName}"][0]
                        ub = bounds[f"{system.name}_{conName}"][1]
                        if all(lb == ub):
                            self.add_constraint(name, equals=lb, linear=True, scaler=1e2)
                        else:
                            self.add_constraint(name, lower=lb, upper=ub, linear=True, scaler=1e2)


################################################################################
# OpenMDAO setup
################################################################################
prob = om.Problem()
prob.model = Top()
model = prob.model
if args.optType.lower() == "mincomp":
    model.add_objective("totalCompliance", scaler=1e-6)
    model.add_constraint(f"{flightPoints[0].name}.mass", upper=2e3, scaler=1e-3)
elif args.optType.lower() == "minmass":
    model.add_objective(f"{flightPoints[0].name}.mass", scaler=1e-3)
    for fp in flightPoints:
        for group in fp.failureGroups:
            model.add_constraint(f"{fp.name}.{group}_ksFailure", upper=1.0, scaler=1.0)
else:
    raise ValueError(f"Unknown optType: {args.optType}")

prob.setup(force_alloc_complex=isComplex)

# ==============================================================================
# Potentially set initial DVs from a previous run
# ==============================================================================
if len(args.initDVs) != 0:
    setValsFromFiles(args.initDVs, prob)

om.n2(prob, show_browser=False, outfile=os.path.join(outputDir, "Struct-N2-Pre-Run.html"))
if args.task == "check":
    exit(0)

# ==============================================================================
# Analysis / derivative check
# ==============================================================================
if args.task in ["analysis", "derivCheck"]:
    prob.run_model()
    prob.model.list_outputs()
    if args.task == "derivCheck":
        np.set_printoptions(precision=16, linewidth=200)
        geoDVNames = model.geometry.DVGeo.getVarNames()
        wrt = geoDVNames  # + ["dv_struct"]
        of = []
        for fp in flightPoints:
            of += [f"{fp.name}.{func}" for func in ["mass", "compliance"]]
            of += [f"{fp.name}.{group}_ksFailure" for group in fp.failureGroups]
        origDVs = {}
        for variable in wrt:
            origDVs[variable] = prob.get_val(variable)
        with open(os.path.join(outputDir, f"derivCheck-{comm.rank:02d}.pkl"), "wb") as pickleFile:
            with open(os.path.join(outputDir, f"derivCheck-{comm.rank:02d}.txt"), "w") as textFile:
                totalsCheckData = prob.check_totals(
                    of=of,
                    wrt=wrt,
                    method="cs" if isComplex else "fd",
                    form="central",
                    step=1e-200 if isComplex else 1e-3,
                    step_calc="abs",
                    out_stream=textFile,
                    compact_print=True,
                    rel_err_tol=1e-8 if isComplex else 1e-2,
                    abs_err_tol=1e6,
                )
                for variable in wrt:
                    prob.set_val(variable, origDVs[variable])
                prob.run_model()
                totalsCheckData.update(
                    prob.check_totals(
                        of=of,
                        wrt=["dv_struct"],
                        method="cs" if isComplex else "fd",
                        step=1e-200 if isComplex else 1e-8,
                        step_calc="abs",
                        out_stream=textFile,
                        compact_print=True,
                        rel_err_tol=1e-8 if isComplex else 1e-2,
                        abs_err_tol=1e6,
                        directional=True,
                    )
                )

                dill.dump(totalsCheckData, pickleFile, protocol=-1)

elif args.task == "opt":
    optimiserMap = {
        "paroptsl1": "ParOpt",
        "paroptfilter": "ParOpt",
        "paroptmma": "ParOpt",
        "slsqp": "SLSQP",
        "nlpqlp": "NLPQLP",
        "snopt": "SNOPT",
        "ipopt": "IPOPT",
    }
    prob.driver = om.pyOptSparseDriver(
        optimizer=optimiserMap[args.optimiser],
        title="TACS Structural Optimization",
        print_opt_prob=True,
    )
    prob.driver.options["hist_file"] = os.path.join(outputDir, "structOpt.hst")
    optimiserOptions = getOptOptions(
        args.optimiser,
        outputDir,
        args.optIter,
        args.hessianUpdate,
        args.initPenalty,
        args.stepLimit,
        args.feasibility,
        args.optimality,
    )
    prob.driver.opt_settings.update(optimiserOptions)
    if args.optimiser == "snopt" and args.timeLimit is not None:
        prob.driver.opt_settings["Time limit"] = args.timeLimit
    prob.driver.options["invalid_desvar_behavior"] = "ignore"
    prob.driver.options["debug_print"] = ["desvars", "objs", "nl_cons"]

    # ==============================================================================
    # Setup case recorder
    # ==============================================================================
    if not args.noRecorder:
        recorder = om.SqliteRecorder(os.path.join(outputDir, "StructOpt.sql"), record_viewer_data=False)
        prob.driver.add_recorder(recorder)
        prob.driver.recording_options["record_desvars"] = True
        prob.driver.recording_options["record_responses"] = True
        prob.driver.recording_options["includes"] = [f"{fp.name}.struct_post.eval_funcs.*" for fp in flightPoints] + [
            f"{fp.name}.struct_post.mass_funcs.*" for fp in flightPoints
        ]

    # ==============================================================================
    # Run Optimisation
    # ==============================================================================
    prob.run_driver()

# --- Write out the DVs and outputs that aren't too long (e.g not the ADflow state vector) in unscaled form to a pickle file ---
writeOutputs(prob, outputDir)
om.n2(prob, show_browser=False, outfile=os.path.join(outputDir, "Struct-N2-Post-Run.html"))
