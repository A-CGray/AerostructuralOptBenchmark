# ==============================================================================
# Standard Python modules
# ==============================================================================
import contextlib
import functools
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import openmdao.api as om
import pytest
from hypothesis import assume, given, settings
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import composite, data, floats, integers, lists, one_of, sampled_from, text
from pyoptsparse import Optimization
from pygeo.pyBlock import pyBlock

# ==============================================================================
# Extension modules
# ==============================================================================
from utils import (
    BANNER_LINE,
    NUM_SPATIAL_DIMS,
    TRANSFER_TO_OPTIONS,
    addConstraintFromOpenMDAO,
    checkInputsHaveSources,
    checkPerformanceOutputsDeclared,
    computeProcsPerMember,
    convertInfiniteBoundsToNone,
    gatherMergedDict,
    gatherUnion,
    getDesignVarSources,
    getMeldBodyTags,
    getPromotedInputNames,
    getRelevantInputs,
    getSource,
    printBanner,
    setLocalVals,
    solveBoundedSecant,
    writeMultiBlockFFD,
)

MAX_TEST_MEMBERS = 50
MAX_TEST_EXTRA_PROCS = 1000
NAME_ALPHABET = "abc"
MISSING_NAME_ALPHABET = NAME_ALPHABET + "d"
RUNSCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runscripts")
NUM_MPI_TEST_PROCS = 3
MPI_TEST_TIMEOUT = 120  # seconds
# Sizes of the FFD blocks and files drawn for the writeMultiBlockFFD tests. pyBlock needs at least 2 nodes in each
# direction
MIN_TEST_BLOCK_NODES = 2
MAX_TEST_BLOCK_NODES = 4
MAX_TEST_BLOCKS_PER_FILE = 3
MAX_TEST_BLOCK_FILES = 3
MAX_TEST_BLOCK_ORIGIN = 100.0
MIN_TEST_NODE_SPACING = 0.1
MAX_TEST_NODE_SPACING = 10.0
# Largest shift of a node from its place in a regular lattice, as a fraction of the node spacing
MAX_TEST_NODE_SHIFT = 0.25
# Wing family name and rib-spar intersection node IDs used by the getMeldBodyTags tests
TEST_WING_FAMILY = "wing"
TEST_RIB_SPAR_NODE_IDS = [1, 2]
# The structural tags of the single MELD body for each --transferTo option: -1 is every structural node except the
# RBE multiplier nodes. It is written out rather than imported as ALL_STRUCT_NODES_TAG so the test pins the value TACS
# expects. The skin+ribends rib-spar intersection nodes are only added once the structural model exists
EXPECTED_MELD_STRUCT_TAGS = {
    "all": -1,
    "skin": ["SKIN"],
    "skin+ribends": ["SKIN"],
    "skin+spar": ["SKIN", "SPAR"],
}
# Bounds, starting points and functions drawn for the solveBoundedSecant tests, of the order of the polar's tail
# rotations in degrees and its cmy values
MAX_TEST_BOUND_MAGNITUDE = 20.0
MIN_TEST_BOUND_WIDTH = 1.0
MAX_TEST_BOUND_WIDTH = 30.0
# The step as a fraction of the distance between the bounds. solveBoundedSecant only promises to evaluate between the
# bounds when the step is at most half that distance, and the largest fraction drawn leaves a margin for rounding
MIN_TEST_STEP_FRACTION = 0.01
MAX_TEST_STEP_FRACTION = 0.4
MIN_TEST_SLOPE = 1e-2
MAX_TEST_SLOPE = 10.0
# Largest cubic term of the monotonic test functions, as a multiple of the linear term one bound width from the root:
# with 1, the slope changes by at most a factor of 4 between the bounds when the root is between them
MAX_TEST_CUBIC_FACTOR = 1.0
# Smallest and largest distance from the nearer bound to a root outside the bounds, as a fraction of the bound width
MIN_TEST_ROOT_GAP = 1e-3
MAX_TEST_ROOT_GAP = 1.0
MAX_TEST_SINE_AMPLITUDE = 1.0
MAX_TEST_SINE_FREQUENCY = 5.0
MAX_TEST_EVALS = 20
# The tolerance, evaluation limit, step and bounds adflowPolar.py trims cmy with, used by the example tests
TEST_F_TOL = 1e-6
TEST_MAX_EVALS = 10
EXAMPLE_STEP = 0.5
EXAMPLE_LOWER = -15.0
EXAMPLE_UPPER = 5.0
# A starting point less than a step below the upper bound
EXAMPLE_X0_NEAR_UPPER = 4.75
# A line through zero with its root above the upper bound
EXAMPLE_ROOT_ABOVE_UPPER = 8.0
EXAMPLE_SLOPE = 1.0
# An exponential whose root is just below the upper bound of [0, 3], where the first secant update from 0 and 0.5
# overshoots the upper bound
EXAMPLE_EXP_LOWER = 0.0
EXAMPLE_EXP_UPPER = 3.0
EXAMPLE_EXP_ROOT = 2.9

# Run by test_same_names_on_every_rank_under_mpi: prints each rank's promoted input names, for a model whose parallel
# group puts some inputs on only some ranks
PROMOTED_INPUT_NAMES_MPI_SCRIPT = """
import json
import openmdao.api as om
from mpi4py import MPI
from utils import getPromotedInputNames

prob = om.Problem()
prob.model.add_subsystem("comp1", om.ExecComp("b = 2 * x"), promotes=["*"])
par = prob.model.add_subsystem("par", om.ParallelGroup(), promotes=["*"])
par.add_subsystem("comp2", om.ExecComp("c = 3 * y"), promotes=["*"])
par.add_subsystem("comp3", om.ExecComp("d = 3 * z"), promotes=["*"])
prob.setup()
allNames = MPI.COMM_WORLD.gather(getPromotedInputNames(prob), root=0)
if MPI.COMM_WORLD.rank == 0:
    print(json.dumps(allNames))
"""


class FakeComm:
    """Imitates an mpi4py communicator as seen from one rank, returning pre-set data for the other ranks.

    The data this rank sends replaces its own pre-set entry, and the last data sent is kept in `sentData`.
    """

    def __init__(self, rank: int, allRankData: list):
        self.rank = rank
        self.allRankData = allRankData
        self.sentData = None

    def _collect(self, data) -> list:
        self.sentData = data
        collected = list(self.allRankData)
        collected[self.rank] = data
        return collected

    def allgather(self, data):
        return self._collect(data)

    def gather(self, data, root=0):
        collected = self._collect(data)
        if self.rank == root:
            return collected
        return None


class TestComputeProcsPerMember(unittest.TestCase):
    @given(numMembers=integers(1, MAX_TEST_MEMBERS), extraProcs=integers(0, MAX_TEST_EXTRA_PROCS))
    def test_split_uses_every_proc_evenly(self, numMembers, extraProcs):
        """
        Given at least as many processors as members,
        when the processors are split between the members,
        then every member gets a count, the counts add up to the number of processors, and no two counts differ by
        more than one.
        """
        numProcs = numMembers + extraProcs
        procsPerMember = computeProcsPerMember(numProcs, numMembers)
        self.assertEqual(len(procsPerMember), numMembers)
        self.assertEqual(sum(procsPerMember), numProcs)
        self.assertLessEqual(max(procsPerMember) - min(procsPerMember), 1)

    @given(numMembers=integers(1, MAX_TEST_MEMBERS), extraProcs=integers(0, MAX_TEST_EXTRA_PROCS))
    def test_extra_procs_go_to_last_members(self, numMembers, extraProcs):
        """Assert that processors left over from an even split go to the last members, so the counts never decrease."""
        procsPerMember = computeProcsPerMember(numMembers + extraProcs, numMembers)
        self.assertEqual(procsPerMember, sorted(procsPerMember))

    def test_fewer_procs_than_members_raises(self):
        """Assert that a ValueError is raised when there are fewer processors than members."""
        with self.assertRaises(ValueError):
            computeProcsPerMember(3, 4)

    def test_no_members_raises(self):
        """Assert that a ValueError is raised when there are no members to split the processors between."""
        with self.assertRaises(ValueError):
            computeProcsPerMember(4, 0)


class TestGatherMergedDict(unittest.TestCase):
    def test_dicts_from_all_ranks_merged_on_every_rank(self):
        """Assert that every rank, not just the root, gets one dict holding the entries from every rank's dict."""
        allRankData = [{"a": 1}, {"b": 2}, {"c": 3}]
        for rank in range(len(allRankData)):
            comm = FakeComm(rank, allRankData)
            self.assertEqual(gatherMergedDict(comm, allRankData[rank]), {"a": 1, "b": 2, "c": 3})

    def test_later_rank_wins_for_duplicate_keys(self):
        """
        Given two ranks that both have a value for the same key,
        when their dicts are merged,
        then the merged dict holds the value from the higher rank.
        """
        allRankData = [{"a": 1, "b": 1}, {"a": 2}]
        comm = FakeComm(0, allRankData)
        self.assertEqual(gatherMergedDict(comm, allRankData[0]), {"a": 2, "b": 1})

    def test_without_broadcast_only_root_gets_result(self):
        """
        Given dicts on two ranks,
        when they are merged without broadcasting,
        then rank 0 gets the merged dict and rank 1 gets an empty dict.
        """
        allRankData = [{"a": 1}, {"b": 2}]
        rootComm = FakeComm(0, allRankData)
        otherComm = FakeComm(1, allRankData)
        self.assertEqual(gatherMergedDict(rootComm, allRankData[0], doBroadcast=False), {"a": 1, "b": 2})
        self.assertEqual(gatherMergedDict(otherComm, allRankData[1], doBroadcast=False), {})


class TestGatherUnion(unittest.TestCase):
    @given(lists(lists(text()), min_size=1))
    def test_result_is_sorted_union_of_all_ranks(self, allRankItems):
        """Assert that the gathered result is the sorted list of every distinct item found on any rank."""
        expected = sorted({item for items in allRankItems for item in items})
        comm = FakeComm(0, allRankItems)
        self.assertEqual(gatherUnion(comm, allRankItems[0]), expected)


class TestSetLocalVals(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        ivc = self.prob.model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
        ivc.add_output("x", np.zeros(2))
        self.prob.setup()

    def test_existing_values_set_and_missing_names_skipped(self):
        """
        Given a problem with a variable "x" but no variable "notInModel",
        when values for both are set,
        then "x" takes its new value and "notInModel" is skipped without an error.
        """
        setLocalVals(self.prob, {"x": np.array([1.0, 2.0]), "notInModel": 3.0})
        np.testing.assert_array_equal(self.prob.get_val("x"), [1.0, 2.0])

    def test_other_errors_still_raised(self):
        """Assert that an error other than a missing name, such as a value of the wrong shape, is still raised."""
        with self.assertRaises(Exception) as context:
            setLocalVals(self.prob, {"x": np.ones(3)})
        self.assertNotIsInstance(context.exception, KeyError)


def capturePrintedBanner(sections: dict) -> str:
    """Return everything printBanner prints for the given sections."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        printBanner(sections)
    return printed.getvalue()


class TestPrintBanner(unittest.TestCase):
    def test_single_section_matches_existing_format(self):
        """
        Given one section with two items,
        when the banner is printed,
        then the output matches the runscript's original banner format character for character.
        """
        expected = "\n" + BANNER_LINE + "\nGrad funcs:\n  - a\n  - b\n" + BANNER_LINE + "\n\n"
        self.assertEqual(capturePrintedBanner({"Grad funcs": ["a", "b"]}), expected)

    def test_sections_separated_by_one_empty_line(self):
        """Assert that several sections are printed in order inside one banner, with one empty line between them."""
        expected = (
            "\n"
            + BANNER_LINE
            + "\nDesign variables:\n  - a\n\nConstraints:\n  - b\n\nObjectives:\n  - c\n"
            + BANNER_LINE
            + "\n\n"
        )
        sections = {"Design variables": ["a"], "Constraints": ["b"], "Objectives": ["c"]}
        self.assertEqual(capturePrintedBanner(sections), expected)

    def test_section_without_items_prints_only_title(self):
        """Assert that a section with no items prints its title and no item lines."""
        expected = "\n" + BANNER_LINE + "\nDisp funcs:\n" + BANNER_LINE + "\n\n"
        self.assertEqual(capturePrintedBanner({"Disp funcs": []}), expected)

    def test_banner_line_matches_existing_banners(self):
        """Assert that the banner line is identical to the line the runscript's banners printed before."""
        self.assertEqual(BANNER_LINE, "===============================================================================")


def buildRelevanceTestProblem() -> om.Problem:
    """
    Build a small model where b depends on x and y, c depends on z, and d depends on an unconnected promoted input p.

    No design variables or responses are declared, so any relevance information must come from explicit seeds.
    """
    prob = om.Problem()
    ivc = prob.model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
    ivc.add_output("x", 1.0)
    ivc.add_output("y", 1.0)
    ivc.add_output("z", 1.0)
    prob.model.add_subsystem("compA", om.ExecComp("a = 2*x"), promotes=["*"])
    prob.model.add_subsystem("compB", om.ExecComp("b = a + y"), promotes=["*"])
    prob.model.add_subsystem("compC", om.ExecComp("c = 3*z"), promotes=["*"])
    prob.model.add_subsystem("compD", om.ExecComp("d = p**2"), promotes=["*"])
    prob.setup()
    prob.final_setup()
    return prob


class TestGetRelevantInputs(unittest.TestCase):
    def setUp(self):
        self.prob = buildRelevanceTestProblem()
        self.model = self.prob.model

    def getSources(self, names: list[str]) -> list[str]:
        return [getSource(self.model, name) for name in names]

    def test_output_depends_on_upstream_candidates_only(self):
        """
        Given an output that depends on some of the candidates,
        when its relevant inputs are found,
        then exactly the candidates upstream of it are returned.
        """
        relevantNames = getRelevantInputs(self.model, getSource(self.model, "b"), self.getSources(["x", "y", "z"]))
        self.assertEqual(sorted(relevantNames), sorted(self.getSources(["x", "y"])))

    def test_unrelated_candidates_excluded(self):
        """Assert that an output that depends on none of the candidates gets an empty list."""
        relevantNames = getRelevantInputs(self.model, getSource(self.model, "c"), self.getSources(["x", "y"]))
        self.assertEqual(relevantNames, [])

    def test_empty_candidates_returns_empty_list(self):
        """
        Given no candidates,
        when the relevant inputs of an output are found,
        then an empty list is returned rather than every variable in the model.
        """
        relevantNames = getRelevantInputs(self.model, getSource(self.model, "b"), [])
        self.assertEqual(relevantNames, [])

    def test_auto_ivc_input_source_found(self):
        """Assert that an input fed by OpenMDAO's automatic IVC is found as relevant through its `_auto_ivc` source."""
        pSource = getSource(self.model, "p")
        relevantNames = getRelevantInputs(self.model, getSource(self.model, "d"), [pSource, getSource(self.model, "x")])
        self.assertEqual(relevantNames, [pSource])

    def test_candidate_equal_to_output_is_relevant(self):
        """Assert that a candidate that is the output itself, like a constraint on a design variable, is relevant."""
        xSource = getSource(self.model, "x")
        self.assertEqual(getRelevantInputs(self.model, xSource, [xSource]), [xSource])

    def test_duplicate_candidate_returned_once(self):
        """Assert that a relevant candidate listed more than once appears only once in the result."""
        xSource = getSource(self.model, "x")
        self.assertEqual(getRelevantInputs(self.model, getSource(self.model, "b"), [xSource, xSource]), [xSource])


class TestGetDesignVarSources(unittest.TestCase):
    def test_returns_source_of_each_design_variable(self):
        """
        Given design variables on an IVC output and on an unconnected promoted input,
        when their sources are found,
        then the source of each design variable is returned, including the automatic IVC source of the input.
        """
        prob = om.Problem()
        ivc = prob.model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
        ivc.add_output("x", 1.0)
        prob.model.add_subsystem("comp", om.ExecComp("y = x + p"), promotes=["*"])
        prob.model.add_design_var("x")
        prob.model.add_design_var("p")
        prob.setup()
        prob.final_setup()
        expected = [getSource(prob.model, "x"), getSource(prob.model, "p")]
        self.assertEqual(sorted(getDesignVarSources(prob.model)), sorted(expected))


class TestGetPromotedInputNames(unittest.TestCase):
    def test_returns_each_unconnected_input_once_in_sorted_order(self):
        """
        Given a model where two components share one promoted input and one has another input of its own,
        when the promoted input names are listed,
        then each input that takes its value from outside the model appears once, in sorted order.
        """
        prob = om.Problem()
        prob.model.add_subsystem("comp1", om.ExecComp("b = 2 * x"), promotes=["*"])
        prob.model.add_subsystem("comp2", om.ExecComp("c = x + w"), promotes=["*"])
        prob.setup()
        self.assertEqual(getPromotedInputNames(prob), ["w", "x"])

    def test_excludes_inputs_fed_by_another_component(self):
        """Assert that an input connected to another component's output is not listed as a model input."""
        prob = om.Problem()
        prob.model.add_subsystem("comp1", om.ExecComp("y = 2 * x"), promotes=["*"])
        prob.model.add_subsystem("comp2", om.ExecComp("z = y + w"), promotes=["*"])
        prob.setup()
        self.assertEqual(getPromotedInputNames(prob), ["w", "x"])

    def test_reports_an_unpromoted_unconnected_input_by_its_path(self):
        """
        Given a component whose input is neither promoted nor connected,
        when the promoted input names are listed,
        then the input is still reported, by its path, so a missing source for it can be caught.
        """
        prob = om.Problem()
        prob.model.add_subsystem("comp", om.ExecComp("b = 2 * a"))
        prob.setup()
        self.assertEqual(getPromotedInputNames(prob), ["comp.a"])

    @unittest.skipIf(shutil.which("mpiexec") is None, "mpiexec is not available")
    def test_same_names_on_every_rank_under_mpi(self):
        """
        Given a model run on several MPI ranks, with a parallel group that puts some inputs on only some ranks,
        when every rank lists the promoted input names,
        then every rank gets the full list, in the same order.
        """
        env = dict(os.environ, PYTHONPATH=os.pathsep.join([RUNSCRIPTS_DIR, os.environ.get("PYTHONPATH", "")]))
        result = subprocess.run(
            ["mpiexec", "-n", str(NUM_MPI_TEST_PROCS), sys.executable, "-c", PROMOTED_INPUT_NAMES_MPI_SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=MPI_TEST_TIMEOUT,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        allRankNames = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(allRankNames, [["x", "y", "z"]] * NUM_MPI_TEST_PROCS)


allRankOutputs = lists(lists(text(NAME_ALPHABET, min_size=1)), min_size=1)


class TestCheckInputsHaveSources(unittest.TestCase):
    @given(allRankOutputs, data())
    def test_inputs_produced_on_any_rank_pass(self, allRankOutputNames, drawData):
        """
        Given output names spread over several ranks,
        when every one of those names is checked as an input from any one rank,
        then no error is raised, because each input is an output on some rank.
        """
        rank = drawData.draw(integers(0, len(allRankOutputNames) - 1))
        inputNames = [name for rankNames in allRankOutputNames for name in rankNames]
        self.assertIsNone(
            checkInputsHaveSources(inputNames, allRankOutputNames[rank], FakeComm(rank, allRankOutputNames))
        )

    @given(allRankOutputs, data())
    def test_missing_input_raises_naming_it(self, allRankOutputNames, drawData):
        """
        Given an input name that is not an output on any rank,
        when the inputs are checked,
        then a ValueError naming the missing input is raised.
        """
        rank = drawData.draw(integers(0, len(allRankOutputNames) - 1))
        missingName = drawData.draw(text(MISSING_NAME_ALPHABET, min_size=1))
        presentNames = [name for rankNames in allRankOutputNames for name in rankNames]
        assume(missingName not in presentNames)
        inputNames = presentNames[:1] + [missingName]
        with self.assertRaises(ValueError) as context:
            checkInputsHaveSources(inputNames, allRankOutputNames[rank], FakeComm(rank, allRankOutputNames))
        self.assertIn(repr(missingName), str(context.exception))

    @given(allRankOutputs, data())
    def test_sends_only_local_outputs_that_are_inputs(self, allRankOutputNames, drawData):
        """
        Given inputs that are all outputs on some rank,
        when the inputs are checked,
        then this rank sends only those of its outputs that are inputs, not all of them.
        """
        allNames = sorted({name for rankNames in allRankOutputNames for name in rankNames})
        assume(len(allNames) > 0)
        rank = drawData.draw(integers(0, len(allRankOutputNames) - 1))
        inputNames = drawData.draw(lists(sampled_from(allNames)))
        comm = FakeComm(rank, allRankOutputNames)
        checkInputsHaveSources(inputNames, allRankOutputNames[rank], comm)
        self.assertEqual(set(comm.sentData), set(inputNames) & set(allRankOutputNames[rank]))

    @given(allRankOutputs)
    def test_empty_inputs_pass(self, allRankOutputNames):
        """Assert that checking no inputs raises no error, whatever outputs the ranks have."""
        self.assertIsNone(checkInputsHaveSources([], allRankOutputNames[0], FakeComm(0, allRankOutputNames)))


TEST_ALIAS_PREFIX = "_testAlias"
DECLARABLE_OUTPUT_NAMES = ["a", "b", "c"]


def buildDeclaredOutputsTestProblem(declaredNames: list[str]) -> om.Problem:
    """Build a toy problem with design variable `x`, outputs `a`, `b` and `c`, and aliased constraints on `declaredNames`."""
    prob = om.Problem()
    ivc = prob.model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
    ivc.add_output("x", 1.0)
    prob.model.add_subsystem("comp", om.ExecComp(["a = 2*x", "b = 3*x", "c = 4*x"]), promotes=["*"])
    prob.model.add_design_var("x")
    for name in declaredNames:
        prob.model.add_constraint(name, alias=f"{TEST_ALIAS_PREFIX}{name}")
    prob.setup()
    prob.final_setup()
    return prob


def singleRankComm() -> FakeComm:
    """Return a fake communicator with one rank."""
    return FakeComm(0, [[]])


class TestCheckPerformanceOutputsDeclared(unittest.TestCase):
    @given(lists(sampled_from(DECLARABLE_OUTPUT_NAMES), unique=True))
    def test_raises_exactly_when_an_output_is_undeclared(self, declaredNames):
        """
        Given a model on which some of its performance outputs are declared as aliased constraints,
        when the declarations are checked,
        then a ValueError naming every undeclared output is raised if any is undeclared, and nothing is raised otherwise.
        """
        prob = buildDeclaredOutputsTestProblem(declaredNames)
        undeclaredNames = sorted(set(DECLARABLE_OUTPUT_NAMES) - set(declaredNames))
        outputNames = DECLARABLE_OUTPUT_NAMES + ["x"]
        if len(undeclaredNames) == 0:
            checkPerformanceOutputsDeclared(
                prob.model, DECLARABLE_OUTPUT_NAMES, outputNames, TEST_ALIAS_PREFIX, singleRankComm()
            )
        else:
            with self.assertRaises(ValueError) as context:
                checkPerformanceOutputsDeclared(
                    prob.model, DECLARABLE_OUTPUT_NAMES, outputNames, TEST_ALIAS_PREFIX, singleRankComm()
                )
            self.assertIn(repr(undeclaredNames), str(context.exception))

    def test_design_variables_need_no_declaration(self):
        """Assert that a performance output that is a design variable of the model doesn't need an aliased constraint."""
        prob = buildDeclaredOutputsTestProblem(DECLARABLE_OUTPUT_NAMES)
        outputNames = DECLARABLE_OUTPUT_NAMES + ["x"]
        checkPerformanceOutputsDeclared(
            prob.model, ["x"] + DECLARABLE_OUTPUT_NAMES, outputNames, TEST_ALIAS_PREFIX, singleRankComm()
        )

    def test_names_this_model_does_not_output_are_ignored(self):
        """Assert that performance outputs created by other flight point models don't need a declaration on this one."""
        prob = buildDeclaredOutputsTestProblem(DECLARABLE_OUTPUT_NAMES)
        outputNames = DECLARABLE_OUTPUT_NAMES + ["x"]
        checkPerformanceOutputsDeclared(prob.model, ["elsewhere"], outputNames, TEST_ALIAS_PREFIX, singleRankComm())

    def test_undeclared_output_on_another_rank_raises_here(self):
        """
        Given a rank on which every performance output is declared, and another rank with an undeclared one,
        when the declarations are checked,
        then a ValueError naming the other rank's undeclared output is raised on this rank too.
        """
        prob = buildDeclaredOutputsTestProblem(DECLARABLE_OUTPUT_NAMES)
        outputNames = DECLARABLE_OUTPUT_NAMES + ["x"]
        otherRankUndeclaredNames = ["elsewhere"]
        comm = FakeComm(0, [[], otherRankUndeclaredNames])
        with self.assertRaises(ValueError) as context:
            checkPerformanceOutputsDeclared(prob.model, DECLARABLE_OUTPUT_NAMES, outputNames, TEST_ALIAS_PREFIX, comm)
        self.assertIn(repr(otherRankUndeclaredNames), str(context.exception))

    def test_constraints_with_other_aliases_do_not_count(self):
        """
        Given an output that is constrained, but not with the performance output alias prefix,
        when the declarations are checked,
        then a ValueError naming it is raised.
        """
        prob = om.Problem()
        ivc = prob.model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
        ivc.add_output("x", 1.0)
        prob.model.add_subsystem("comp", om.ExecComp("a = 2*x"), promotes=["*"])
        prob.model.add_design_var("x")
        prob.model.add_constraint("a", upper=1.0)
        prob.setup()
        prob.final_setup()
        with self.assertRaises(ValueError) as context:
            checkPerformanceOutputsDeclared(prob.model, ["a"], ["a", "x"], TEST_ALIAS_PREFIX, singleRankComm())
        self.assertIn(repr(["a"]), str(context.exception))


class TestConvertInfiniteBoundsToNone(unittest.TestCase):
    def test_no_bound_stays_none(self):
        """Assert that a bound of None is returned as None."""
        self.assertIsNone(convertInfiniteBoundsToNone(None))

    @given(floats(allow_nan=False))
    def test_scalar_bound(self, bound):
        """Assert that an infinite scalar bound becomes None and a finite one is returned unchanged."""
        converted = convertInfiniteBoundsToNone(bound)
        if np.isinf(bound):
            self.assertIsNone(converted)
        else:
            self.assertEqual(converted, bound)

    @given(lists(floats(allow_nan=False), min_size=1))
    def test_array_bound_entries(self, bound):
        """
        Given an array bound with any mix of finite and infinite entries,
        when it is converted,
        then each infinite entry becomes None and each finite entry keeps its value.
        """
        converted = convertInfiniteBoundsToNone(np.array(bound))
        self.assertEqual(len(converted), len(bound))
        for convertedValue, value in zip(converted, bound, strict=True):
            if np.isinf(value):
                self.assertIsNone(convertedValue)
            else:
                self.assertEqual(convertedValue, value)


class TestAddConstraintFromOpenMDAO(unittest.TestCase):
    def test_infinite_bound_entries_are_unbounded_in_pyoptsparse(self):
        """
        Given an OpenMDAO constraint whose upper bound array has infinite entries and no lower bound,
        when it is added to a pyOptSparse problem,
        then the infinite upper bound entries and all the lower bound entries are None, so pyOptSparse leaves them
        unbounded, while the finite upper bound entry is kept.
        """
        prob = om.Problem()
        ivc = prob.model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
        ivc.add_output("x", np.ones(3))
        prob.model.add_design_var("x")
        prob.model.add_constraint("x", upper=np.array([1.0, np.inf, np.inf]))
        prob.setup()
        prob.final_setup()

        optProb = Optimization("test", lambda x: x)
        optProb.addVarGroup("x", 3)
        addConstraintFromOpenMDAO(prob.model.get_constraints()["x"], optProb, prob)

        con = optProb.constraints["x"]
        self.assertEqual(list(con.upper), [1.0, None, None])
        self.assertEqual(list(con.lower), [None, None, None])


@composite
def ffdBlocks(draw):
    """Draw the (nx, ny, nz, 3) node coordinates of an FFD block: a regular lattice with each node shifted a little

    pyBlock can't build its topology for blocks with coincident nodes, so the coordinates are not fully arbitrary.
    """
    blockSize = [draw(integers(MIN_TEST_BLOCK_NODES, MAX_TEST_BLOCK_NODES)) for _ in range(NUM_SPATIAL_DIMS)]
    origin = draw(arrays(float, NUM_SPATIAL_DIMS, elements=floats(-MAX_TEST_BLOCK_ORIGIN, MAX_TEST_BLOCK_ORIGIN)))
    spacing = draw(arrays(float, NUM_SPATIAL_DIMS, elements=floats(MIN_TEST_NODE_SPACING, MAX_TEST_NODE_SPACING)))
    nodeShifts = draw(
        arrays(float, (*blockSize, NUM_SPATIAL_DIMS), elements=floats(-MAX_TEST_NODE_SHIFT, MAX_TEST_NODE_SHIFT))
    )
    nodeIndices = np.stack(np.meshgrid(*(np.arange(numNodes) for numNodes in blockSize), indexing="ij"), axis=-1)
    return origin + (nodeIndices + nodeShifts) * spacing


def writePlot3DFile(blocks: list, fileName: str) -> None:
    """Write blocks of (nx, ny, nz, 3) node coordinates to an ASCII plot3d file, keeping every digit"""
    with open(fileName, "w") as f:
        f.write(f"{len(blocks)}\n")
        for block in blocks:
            f.write(" ".join(str(numNodes) for numNodes in block.shape[:NUM_SPATIAL_DIMS]) + "\n")
        for block in blocks:
            for dim in range(NUM_SPATIAL_DIMS):
                f.write(" ".join(repr(float(coord)) for coord in block[..., dim].flatten(order="F")) + "\n")


class TestWriteMultiBlockFFD(unittest.TestCase):
    # Building a pyBlock takes longer than Hypothesis's default deadline when the machine is busy
    @settings(deadline=None)
    @given(
        fileBlocks=lists(
            lists(ffdBlocks(), min_size=1, max_size=MAX_TEST_BLOCKS_PER_FILE), min_size=1, max_size=MAX_TEST_BLOCK_FILES
        )
    )
    def test_round_trip_through_pyblock_keeps_blocks_in_order(self, fileBlocks):
        """
        Given several plot3d files, each holding one or more FFD blocks,
        when writeMultiBlockFFD combines them and pyBlock reads the combined file,
        then pyBlock holds exactly the input blocks, in the order of the files and of the blocks within each file.
        """
        with tempfile.TemporaryDirectory() as tmpDir:
            blockFiles = []
            for ii, blocks in enumerate(fileBlocks):
                blockFiles.append(os.path.join(tmpDir, f"blocks{ii}.xyz"))
                writePlot3DFile(blocks, blockFiles[-1])
            combinedFile = os.path.join(tmpDir, "combined.xyz")
            writeMultiBlockFFD(blockFiles, combinedFile)
            combinedFFD = pyBlock("plot3d", fileName=combinedFile, FFD=True)

        expectedBlocks = [block for blocks in fileBlocks for block in blocks]
        self.assertEqual(combinedFFD.nVol, len(expectedBlocks))
        for vol, expectedBlock in zip(combinedFFD.vols, expectedBlocks, strict=True):
            np.testing.assert_array_equal(vol.coef, expectedBlock)

    def test_file_with_missing_values_raises(self):
        """Assert that writeMultiBlockFFD raises a ValueError naming a file that holds fewer values than its header
        describes.
        """
        block = np.zeros((2, 2, 2, NUM_SPATIAL_DIMS))
        with tempfile.TemporaryDirectory() as tmpDir:
            truncatedFile = os.path.join(tmpDir, "truncated.xyz")
            writePlot3DFile([block], truncatedFile)
            with open(truncatedFile) as f:
                values = f.read().split()
            with open(truncatedFile, "w") as f:
                f.write(" ".join(values[:-1]) + "\n")
            with self.assertRaisesRegex(ValueError, "truncated.xyz"):
                writeMultiBlockFFD([truncatedFile], os.path.join(tmpDir, "combined.xyz"))

    def test_file_with_incomplete_header_raises(self):
        """Assert that writeMultiBlockFFD raises a ValueError naming a file that is empty or ends before giving the
        sizes of every block its header counts.
        """
        incompleteHeaders = ["", "1\n", "2\n2 2 2\n"]
        with tempfile.TemporaryDirectory() as tmpDir:
            for header in incompleteHeaders:
                with self.subTest(header=header):
                    headerFile = os.path.join(tmpDir, "incompleteHeader.xyz")
                    with open(headerFile, "w") as f:
                        f.write(header)
                    with self.assertRaisesRegex(ValueError, "incompleteHeader.xyz"):
                        writeMultiBlockFFD([headerFile], os.path.join(tmpDir, "combined.xyz"))


class TestGetMeldBodyTags:
    @pytest.mark.parametrize("includeTail", [False, True])
    @pytest.mark.parametrize("transferTo", TRANSFER_TO_OPTIONS)
    @given(wingFamily=text(min_size=1))
    def test_only_the_wing_is_tagged(self, transferTo, includeTail, wingFamily):
        """
        Given any --transferTo option, with or without the tail, and any wing family name,
        when getMeldBodyTags builds the MELD body tags,
        then `all` without the tail gives None, so MELD couples every node as before, and every other case gives one
        body whose only aero tag is the wing family and whose struct tags are the option's components, or -1 (every
        structural node) for `all` with the tail.
        """
        bodyTags = getMeldBodyTags(transferTo, wingFamily, includeTail)
        if transferTo == "all" and not includeTail:
            assert bodyTags is None
        else:
            assert bodyTags == [{"aero": [wingFamily], "struct": EXPECTED_MELD_STRUCT_TAGS[transferTo]}]

    def test_struct_tags_can_be_extended_in_place(self):
        """
        Given the body tags getMeldBodyTags returns for skin+ribends,
        when node IDs are appended to their struct tags in place, as the runscript does with the rib-spar nodes,
        then a later call still returns the skin-only tags, because every call builds new lists.
        """
        bodyTags = getMeldBodyTags("skin+ribends", TEST_WING_FAMILY, False)
        bodyTags[0]["struct"] += TEST_RIB_SPAR_NODE_IDS
        assert getMeldBodyTags("skin+ribends", TEST_WING_FAMILY, False) == [
            {"aero": [TEST_WING_FAMILY], "struct": ["SKIN"]}
        ]

    def test_unknown_option_raises(self):
        """Assert that getMeldBodyTags raises a ValueError naming a --transferTo option it doesn't know."""
        with pytest.raises(ValueError, match="'ribs'"):
            getMeldBodyTags("ribs", TEST_WING_FAMILY, True)


def evaluateMonotonicCubic(x: float, root: float, slope: float, cubicCoeff: float) -> float:
    """Evaluate a cubic whose only real root is at root, which is monotonic when slope and cubicCoeff have the same
    sign
    """
    return slope * (x - root) + cubicCoeff * (x - root) ** 3


def evaluateSine(x: float, amplitude: float, frequency: float, phase: float, offset: float) -> float:
    """Evaluate a sine wave, which has many roots or none"""
    return offset + amplitude * np.sin(frequency * x + phase)


def evaluateExponential(x: float, root: float) -> float:
    """Evaluate an exponential whose only root is at root, which is steep above it and flat below it"""
    return np.exp(x) - np.exp(root)


def evaluateConstant(x: float, value: float) -> float:
    """Evaluate a flat function"""
    return value


def recordCalls(func) -> tuple:
    """Wrap a function so that each point it is evaluated at, and the value there, is recorded in order"""
    calls = []

    def recordedFunc(x: float) -> float:
        fx = func(x)
        calls.append((x, fx))
        return fx

    return recordedFunc, calls


@composite
def secantStarts(draw) -> tuple[float, float, float, float]:
    """Draw the lower and upper bounds, a starting point between them or on one of them, and a step"""
    lower = draw(floats(-MAX_TEST_BOUND_MAGNITUDE, 0.0))
    upper = lower + draw(floats(MIN_TEST_BOUND_WIDTH, MAX_TEST_BOUND_WIDTH))
    x0 = draw(floats(lower, upper))
    width = upper - lower
    step = draw(floats(MIN_TEST_STEP_FRACTION * width, MAX_TEST_STEP_FRACTION * width))
    return lower, upper, x0, step


@composite
def monotonicCubics(draw, lower: float, upper: float, root: float):
    """Draw a monotonic cubic with its root at root, whose slope changes by at most a factor of 4 between the bounds
    when the root is between them
    """
    slope = draw(sampled_from([-1.0, 1.0])) * draw(floats(MIN_TEST_SLOPE, MAX_TEST_SLOPE))
    cubicCoeff = np.sign(slope) * draw(floats(0.0, MAX_TEST_CUBIC_FACTOR)) * abs(slope) / (upper - lower) ** 2
    return functools.partial(evaluateMonotonicCubic, root=root, slope=slope, cubicCoeff=cubicCoeff)


@composite
def rootInsideProblems(draw) -> tuple:
    """Draw bounds, a starting point, a step and a monotonic cubic with its root between the bounds or on one"""
    lower, upper, x0, step = draw(secantStarts())
    root = draw(floats(lower, upper))
    return lower, upper, x0, step, draw(monotonicCubics(lower, upper, root))


@composite
def rootOutsideProblems(draw) -> tuple:
    """Draw bounds, a starting point, a step and a monotonic cubic with its root below the lower bound or above the
    upper one, far enough away that the cubic is larger than TEST_F_TOL everywhere between the bounds
    """
    lower, upper, x0, step = draw(secantStarts())
    gap = draw(floats(MIN_TEST_ROOT_GAP, MAX_TEST_ROOT_GAP)) * (upper - lower)
    root = draw(sampled_from([lower - gap, upper + gap]))
    return lower, upper, x0, step, draw(monotonicCubics(lower, upper, root))


@composite
def flatProblems(draw) -> tuple:
    """Draw bounds, a starting point, a step and a constant function too far from zero to have a root"""
    lower, upper, x0, step = draw(secantStarts())
    value = draw(sampled_from([-1.0, 1.0])) * draw(floats(2 * TEST_F_TOL, MAX_TEST_SLOPE))
    return lower, upper, x0, step, functools.partial(evaluateConstant, value=value)


@composite
def anyProblems(draw) -> tuple:
    """Draw bounds, a starting point, a step, an evaluation limit and a monotonic cubic with its root anywhere near
    the bounds, a sine wave or a constant
    """
    lower, upper, x0, step = draw(secantStarts())
    width = upper - lower
    root = draw(floats(lower - width, upper + width))
    sine = functools.partial(
        evaluateSine,
        amplitude=draw(floats(0.0, MAX_TEST_SINE_AMPLITUDE)),
        frequency=draw(floats(0.0, MAX_TEST_SINE_FREQUENCY)),
        phase=draw(floats(0.0, 2 * np.pi)),
        offset=draw(floats(-MAX_TEST_SINE_AMPLITUDE, MAX_TEST_SINE_AMPLITUDE)),
    )
    constant = functools.partial(evaluateConstant, value=draw(floats(-MAX_TEST_SLOPE, MAX_TEST_SLOPE)))
    func = draw(one_of(monotonicCubics(lower, upper, root), sampled_from([sine, constant])))
    return lower, upper, x0, step, func, draw(integers(1, MAX_TEST_EVALS))


class TestSolveBoundedSecant(unittest.TestCase):
    @given(rootInsideProblems())
    def test_root_inside_bounds_is_found(self, problem):
        """
        Given a monotonic cubic with its root between the bounds or on one of them, and a starting point between them
        or on one of them,
        when solveBoundedSecant searches with the polar's tolerance and evaluation limit,
        then it converges at a point between the bounds or on one of them where the cubic is within the tolerance of
        zero.
        """
        lower, upper, x0, step, func = problem
        x, fx, isConverged, _ = solveBoundedSecant(func, x0, step, lower, upper, TEST_F_TOL, TEST_MAX_EVALS)
        self.assertTrue(isConverged)
        self.assertLessEqual(abs(fx), TEST_F_TOL)
        self.assertTrue(lower <= x <= upper)

    @given(anyProblems())
    def test_iterates_stay_between_bounds(self, problem):
        """
        Given a monotonic cubic with its root inside or outside the bounds, a sine wave or a constant, and a starting
        point between the bounds or on one of them,
        when solveBoundedSecant searches for a root,
        then every point it evaluates the function at is between the bounds or on one of them.
        """
        lower, upper, x0, step, func, maxEvals = problem
        recordedFunc, calls = recordCalls(func)
        solveBoundedSecant(recordedFunc, x0, step, lower, upper, TEST_F_TOL, maxEvals)
        for x, _ in calls:
            self.assertTrue(lower <= x <= upper, f"{x} is outside [{lower}, {upper}]")

    @given(rootOutsideProblems())
    def test_root_outside_bounds_is_not_converged(self, problem):
        """Assert that solveBoundedSecant reports no convergence for a monotonic cubic whose root is outside the
        bounds.
        """
        lower, upper, x0, step, func = problem
        _, _, isConverged, _ = solveBoundedSecant(func, x0, step, lower, upper, TEST_F_TOL, TEST_MAX_EVALS)
        self.assertFalse(isConverged)

    @given(flatProblems())
    def test_flat_function_is_not_converged(self, problem):
        """Assert that solveBoundedSecant reports no convergence for a nonzero constant, and stops after the two
        starting points because their values are equal.
        """
        lower, upper, x0, step, func = problem
        _, _, isConverged, numEvals = solveBoundedSecant(func, x0, step, lower, upper, TEST_F_TOL, TEST_MAX_EVALS)
        self.assertFalse(isConverged)
        self.assertEqual(numEvals, 2)

    @given(anyProblems())
    def test_max_evals_never_exceeded(self, problem):
        """
        Given any of the test functions and an evaluation limit,
        when solveBoundedSecant searches for a root,
        then it evaluates the function at most the limit's number of times and reports how many times it did.
        """
        lower, upper, x0, step, func, maxEvals = problem
        recordedFunc, calls = recordCalls(func)
        _, _, _, numEvals = solveBoundedSecant(recordedFunc, x0, step, lower, upper, TEST_F_TOL, maxEvals)
        self.assertLessEqual(len(calls), maxEvals)
        self.assertEqual(numEvals, len(calls))

    @given(anyProblems())
    def test_returns_last_evaluated_point(self, problem):
        """
        Given any of the test functions and an evaluation limit,
        when solveBoundedSecant searches for a root,
        then it returns the last point it evaluated and the value there, and reports convergence exactly when that
        value is within the tolerance of zero.
        """
        lower, upper, x0, step, func, maxEvals = problem
        recordedFunc, calls = recordCalls(func)
        x, fx, isConverged, _ = solveBoundedSecant(recordedFunc, x0, step, lower, upper, TEST_F_TOL, maxEvals)
        self.assertEqual((x, fx), calls[-1])
        self.assertEqual(isConverged, abs(fx) <= TEST_F_TOL)

    def test_second_point_is_below_first_when_above_upper_bound(self):
        """Assert that solveBoundedSecant takes its second point a step below the first when a step above it would be
        above the upper bound.
        """
        line = functools.partial(evaluateMonotonicCubic, root=0.0, slope=EXAMPLE_SLOPE, cubicCoeff=0.0)
        recordedFunc, calls = recordCalls(line)
        solveBoundedSecant(
            recordedFunc, EXAMPLE_X0_NEAR_UPPER, EXAMPLE_STEP, EXAMPLE_LOWER, EXAMPLE_UPPER, TEST_F_TOL, TEST_MAX_EVALS
        )
        self.assertEqual(calls[1][0], EXAMPLE_X0_NEAR_UPPER - EXAMPLE_STEP)

    def test_stops_on_landing_on_same_bound_twice(self):
        """
        Given a line whose root is above the upper bound, and a start at zero,
        when solveBoundedSecant's first secant update is clipped to the upper bound and the next one lands there again,
        then it stops without converging and without evaluating the upper bound a second time.
        """
        line = functools.partial(
            evaluateMonotonicCubic, root=EXAMPLE_ROOT_ABOVE_UPPER, slope=EXAMPLE_SLOPE, cubicCoeff=0.0
        )
        recordedFunc, calls = recordCalls(line)
        x, _, isConverged, numEvals = solveBoundedSecant(
            recordedFunc, 0.0, EXAMPLE_STEP, EXAMPLE_LOWER, EXAMPLE_UPPER, TEST_F_TOL, TEST_MAX_EVALS
        )
        self.assertEqual([callX for callX, _ in calls], [0.0, EXAMPLE_STEP, EXAMPLE_UPPER])
        self.assertFalse(isConverged)
        self.assertEqual(x, EXAMPLE_UPPER)
        self.assertEqual(numEvals, len(calls))

    def test_recovers_after_update_clipped_to_bound(self):
        """
        Given an exponential whose root is just below the upper bound, and a start at the lower bound,
        when solveBoundedSecant's first secant update overshoots the upper bound and is clipped to it,
        then the later updates come back between the bounds and converge there.
        """
        recordedFunc, calls = recordCalls(functools.partial(evaluateExponential, root=EXAMPLE_EXP_ROOT))
        x, _, isConverged, _ = solveBoundedSecant(
            recordedFunc,
            EXAMPLE_EXP_LOWER,
            EXAMPLE_STEP,
            EXAMPLE_EXP_LOWER,
            EXAMPLE_EXP_UPPER,
            TEST_F_TOL,
            TEST_MAX_EVALS,
        )
        self.assertEqual(calls[2][0], EXAMPLE_EXP_UPPER)
        self.assertTrue(isConverged)
        self.assertTrue(EXAMPLE_EXP_LOWER < x < EXAMPLE_EXP_UPPER)


if __name__ == "__main__":
    unittest.main()
