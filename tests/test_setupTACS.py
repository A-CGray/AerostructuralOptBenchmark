# ==============================================================================
# Standard Python modules
# ==============================================================================
import copy
import os
import string
import tempfile
import unittest
from argparse import Namespace
from types import SimpleNamespace

# ==============================================================================
# External Python modules
# ==============================================================================
import dill
from hypothesis import given
from hypothesis import strategies as st
from mpi4py import MPI

# ==============================================================================
# Extension modules
# ==============================================================================
from AircraftSpecs.STWFlightPoints import flightPointSets
from CommonArgs import parser
from SETUP.setupTACS import (
    NONLINEAR_PROBLEM_OPTIONS,
    PANEL_LENGTHS_FILE_NAME,
    STRUCT_DV_MAP_FILE_NAME,
    buildStructDVDictMap,
    createTacsBuilder,
    getNonlinearSolverOptions,
    readPanelLengths,
    writeStructDVMap,
)
from utils import getStructMeshPath

COMPONENT_NAME_ALPHABET = string.ascii_letters + string.digits + "/._"
TEST_STRUCT_LEVEL = 3
TEST_FLIGHT_POINT_SET = "maneuverOnly"
TEST_RES_REF = 123.0


def createBuilderInputs(argList):
    """Parse the given command line arguments and return them with the test mesh file and flight points"""
    args = parser.parse_args(argList)
    structMeshFile = getStructMeshPath(level=TEST_STRUCT_LEVEL, order=args.structOrder)
    flightPoints = flightPointSets[TEST_FLIGHT_POINT_SET]
    return args, structMeshFile, flightPoints


class TestReadPanelLengths(unittest.TestCase):
    @given(
        st.dictionaries(
            st.text(COMPONENT_NAME_ALPHABET, min_size=1),
            st.floats(allow_nan=False, allow_infinity=False),
        )
    )
    def test_round_trips_written_lengths(self, lengths):
        """
        Given a panel lengths file written next to a mesh file,
        when readPanelLengths is called with the mesh file path,
        then it returns the component names and lengths that were written.
        """
        with tempfile.TemporaryDirectory() as tmpDir:
            with open(os.path.join(tmpDir, PANEL_LENGTHS_FILE_NAME), "w") as panelLengthFile:
                for name, value in lengths.items():
                    panelLengthFile.write(f"{name},{value!r}\n")
            self.assertEqual(readPanelLengths(os.path.join(tmpDir, "mesh.bdf")), lengths)


class TestGetNonlinearSolverOptions(unittest.TestCase):
    def test_linear_analysis_gets_no_options(self):
        """Assert that a linear analysis gets no Newton or continuation solver options."""
        self.assertEqual(getNonlinearSolverOptions(Namespace(nonlinear=False)), (None, None))

    def test_nonlinear_options_match_the_scripts(self):
        """
        Given a nonlinear analysis,
        when getNonlinearSolverOptions is called,
        then the Newton and continuation options are the ones the multipoint runscript used before they were shared,
        which structRun.py also used except for its lower continuation iteration limit.
        """
        newtonOptions, continuationOptions = getNonlinearSolverOptions(Namespace(nonlinear=True))
        self.assertEqual(
            newtonOptions,
            {
                "UseEW": True,
                "MaxLinIters": 10,
                "skipFirstNLineSearch": 1,
                "ForceFirstIter": True,
                "maxIter": 20,
            },
        )
        self.assertEqual(
            continuationOptions,
            {
                "RelTol": 1e-7,
                "UsePredictor": True,
                "NumPredictorStates": 4,
                "maxIter": 20,
            },
        )

    def test_returns_new_dicts_each_call(self):
        """Assert that modifying the option dicts returned by one call doesn't change those returned by the next."""
        args = Namespace(nonlinear=True)
        firstNewtonOptions, firstContinuationOptions = getNonlinearSolverOptions(args)
        expectedNewtonOptions = copy.deepcopy(firstNewtonOptions)
        expectedContinuationOptions = copy.deepcopy(firstContinuationOptions)
        firstNewtonOptions["maxIter"] = -1
        firstContinuationOptions["maxIter"] = -1
        self.assertEqual(getNonlinearSolverOptions(args), (expectedNewtonOptions, expectedContinuationOptions))


class TestCreateTacsBuilder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpDirObject = tempfile.TemporaryDirectory()
        cls.tmpDir = cls.tmpDirObject.name
        cls.args, structMeshFile, cls.flightPoints = createBuilderInputs([])
        cls.extraSetupCalls = []

        def recordExtraProblemSetup(scenarioName, feaAssembler, problem):
            cls.extraSetupCalls.append((scenarioName, feaAssembler, problem, list(problem.getFunctionKeys())))

        problemOptions = {"outputDir": cls.tmpDir, "useMonitor": True, "monitorFrequency": 1}
        cls.builder = createTacsBuilder(
            cls.args,
            structMeshFile,
            cls.flightPoints,
            problemOptions,
            extraProblemSetup=recordExtraProblemSetup,
            write_solution=False,
            res_ref=TEST_RES_REF,
        )
        cls.builder.initialize(MPI.COMM_SELF)
        cls.assembler = cls.builder.get_fea_assembler()

        # Call the problem setup the way MPhys does, for a flight point other than the first
        cls.scenarioName = cls.flightPoints[1].name
        cls.problem = cls.assembler.createStaticProblem(cls.scenarioName)
        cls.builder.problem_setup(cls.scenarioName, cls.assembler, cls.problem)

    @classmethod
    def tearDownClass(cls):
        cls.tmpDirObject.cleanup()

    def test_problem_setup_applies_problem_options(self):
        """Assert that the builder's problem setup applies the problem options it was given."""
        self.assertEqual(self.problem.getOption("outputDir"), self.tmpDir)

    def test_extra_problem_setup_runs_after_standard_setup(self):
        """
        Given a builder created with an extra problem setup function,
        when the builder sets up a problem,
        then the extra function is called once with the scenario name, assembler and problem, after the standard
        setup has added the structural functions.
        """
        self.assertEqual(len(self.extraSetupCalls), 1)
        scenarioName, feaAssembler, problem, functionKeys = self.extraSetupCalls[0]
        self.assertEqual(scenarioName, self.scenarioName)
        self.assertIs(feaAssembler, self.assembler)
        self.assertIs(problem, self.problem)
        self.assertIn("mass", functionKeys)

    def test_constraints_only_for_first_flight_point(self):
        """Assert that the builder adds TACS constraints for the first flight point only."""
        firstConstraints = []
        self.builder.constraint_setup(self.flightPoints[0].name, self.assembler, firstConstraints)
        self.assertGreater(len(firstConstraints), 0)

        secondConstraints = []
        self.builder.constraint_setup(self.flightPoints[1].name, self.assembler, secondConstraints)
        self.assertEqual(len(secondConstraints), 0)

    def test_builder_kwargs_are_passed_to_tacs_builder(self):
        """Assert that extra keyword arguments to createTacsBuilder are passed on to the TacsBuilder."""
        self.assertIs(self.builder.write_solution, False)
        self.assertEqual(self.builder.res_ref, TEST_RES_REF)


class TestCreateTacsBuilderNonlinear(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpDirObject = tempfile.TemporaryDirectory()
        cls.args, structMeshFile, flightPoints = createBuilderInputs(["--nonlinear"])
        cls.problemOptions = {"outputDir": cls.tmpDirObject.name, "useMonitor": False, "monitorFrequency": 1}
        cls.problemOptionsBefore = copy.deepcopy(cls.problemOptions)
        builder = createTacsBuilder(cls.args, structMeshFile, flightPoints, cls.problemOptions)
        builder.initialize(MPI.COMM_SELF)
        assembler = builder.get_fea_assembler()

        scenarioName = flightPoints[0].name
        cls.problem = assembler.createStaticProblem(scenarioName)
        builder.problem_setup(scenarioName, assembler, cls.problem)

    @classmethod
    def tearDownClass(cls):
        cls.tmpDirObject.cleanup()

    def test_nonlinear_problem_gets_shared_solver_options(self):
        """
        Given a builder for a nonlinear analysis,
        when it sets up a problem,
        then the continuation and Newton solvers get the shared iteration limits and the problem gets the shared
        nonlinear problem options.
        """
        newtonOptions, continuationOptions = getNonlinearSolverOptions(self.args)
        self.assertEqual(self.problem.nonlinearSolver.getOption("maxIter"), continuationOptions["maxIter"])
        self.assertEqual(self.problem.nonlinearSolver.innerSolver.getOption("maxIter"), newtonOptions["maxIter"])
        self.assertEqual(self.problem.getOption("printLevel"), NONLINEAR_PROBLEM_OPTIONS["printLevel"])

    def test_problem_options_dict_is_not_modified(self):
        """Assert that creating a nonlinear builder doesn't add the nonlinear options to the caller's dict."""
        self.assertEqual(self.problemOptions, self.problemOptionsBefore)


class TestWriteStructDVMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.args, structMeshFile, flightPoints = createBuilderInputs([])
        builder = createTacsBuilder(cls.args, structMeshFile, flightPoints, {})
        builder.initialize(MPI.COMM_SELF)
        cls.assembler = builder.get_fea_assembler()

    def test_root_rank_writes_the_dv_map(self):
        """Assert that on rank 0 writeStructDVMap writes the assembler's DV map to the output directory."""
        with tempfile.TemporaryDirectory() as tmpDir:
            writeStructDVMap(self.assembler, self.args, tmpDir, SimpleNamespace(rank=0))
            with open(os.path.join(tmpDir, STRUCT_DV_MAP_FILE_NAME), "rb") as structDVMapFile:
                self.assertEqual(dill.load(structDVMapFile), buildStructDVDictMap(self.assembler, self.args))

    def test_other_ranks_write_nothing(self):
        """Assert that on ranks other than 0 writeStructDVMap writes no file."""
        with tempfile.TemporaryDirectory() as tmpDir:
            writeStructDVMap(self.assembler, self.args, tmpDir, SimpleNamespace(rank=1))
            self.assertEqual(os.listdir(tmpDir), [])
