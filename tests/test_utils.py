# ==============================================================================
# Standard Python modules
# ==============================================================================
import contextlib
import io
import json
import os
import shutil
import subprocess
import sys
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import openmdao.api as om
from hypothesis import assume, given
from hypothesis.strategies import data, integers, lists, sampled_from, text

# ==============================================================================
# Extension modules
# ==============================================================================
from utils import (
    BANNER_LINE,
    allowResponsesAfterSetup,
    checkInputsHaveSources,
    computeProcsPerMember,
    gatherMergedDict,
    gatherUnion,
    getDesignVarSources,
    getPromotedInputNames,
    getRelevantInputs,
    getSource,
    printBanner,
    setLocalVals,
)

MAX_TEST_MEMBERS = 50
MAX_TEST_EXTRA_PROCS = 1000
NAME_ALPHABET = "abc"
MISSING_NAME_ALPHABET = NAME_ALPHABET + "d"
RUNSCRIPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runscripts")
NUM_MPI_TEST_PROCS = 3
MPI_TEST_TIMEOUT = 120  # seconds

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


class TestAllowResponsesAfterSetup(unittest.TestCase):
    def setUp(self):
        self.prob = om.Problem()
        ivc = self.prob.model.add_subsystem("ivc", om.IndepVarComp(), promotes=["*"])
        ivc.add_output("x", 1.0)
        self.prob.model.add_subsystem("comp", om.ExecComp("y = 2*x"), promotes=["*"])
        self.prob.model.add_design_var("x")
        self.prob.setup()

    def test_constraint_added_inside_is_included_by_final_setup(self):
        """
        Given a problem that has already been set up,
        when a constraint is added inside the context manager and final_setup is then called,
        then the problem's constraints include it.
        """
        with allowResponsesAfterSetup(self.prob):
            self.prob.model.add_constraint("y", upper=1.0)
        self.prob.final_setup()
        self.assertIn("y", self.prob.model.get_constraints())

    def test_static_mode_restored_after_error(self):
        """Assert that the problem's static mode flag is restored even when an error is raised inside the context."""
        originalStaticMode = self.prob.model._problem_meta["static_mode"]
        with self.assertRaises(RuntimeError):
            with allowResponsesAfterSetup(self.prob):
                raise RuntimeError("Failure while adding a response")
        self.assertEqual(self.prob.model._problem_meta["static_mode"], originalStaticMode)

    def test_after_final_setup_raises(self):
        """
        Given a problem on which final_setup has already been called,
        when the context manager is entered,
        then an error is raised, because responses added now would never be included.
        """
        self.prob.final_setup()
        with self.assertRaises(RuntimeError):
            with allowResponsesAfterSetup(self.prob):
                self.prob.model.add_constraint("y", upper=1.0)

    def test_before_setup_raises(self):
        """Assert that entering the context manager on a problem that hasn't been set up raises an error."""
        prob = om.Problem()
        with self.assertRaises(RuntimeError):
            with allowResponsesAfterSetup(prob):
                pass


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


if __name__ == "__main__":
    unittest.main()
