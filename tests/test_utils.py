# ==============================================================================
# Standard Python modules
# ==============================================================================
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
from hypothesis import given
from hypothesis.strategies import integers

# ==============================================================================
# Extension modules
# ==============================================================================
from utils import computeProcsPerMember

MAX_TEST_MEMBERS = 50
MAX_TEST_EXTRA_PROCS = 1000


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


if __name__ == "__main__":
    unittest.main()
