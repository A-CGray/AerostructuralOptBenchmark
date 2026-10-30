# ==============================================================================
# Standard Python modules
# ==============================================================================
import contextlib
import io
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import openmdao.api as om
from hypothesis import given
from hypothesis.strategies import integers, lists, text

# ==============================================================================
# Extension modules
# ==============================================================================
from utils import BANNER_LINE, computeProcsPerMember, gatherMergedDict, gatherUnion, printBanner, setLocalVals

MAX_TEST_MEMBERS = 50
MAX_TEST_EXTRA_PROCS = 1000


class FakeComm:
    """Imitates an mpi4py communicator as seen from one rank, returning pre-set data for every rank."""

    def __init__(self, rank: int, allRankData: list):
        self.rank = rank
        self.allRankData = allRankData

    def allgather(self, data):
        return list(self.allRankData)

    def gather(self, data, root=0):
        if self.rank == root:
            return list(self.allRankData)
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


if __name__ == "__main__":
    unittest.main()
