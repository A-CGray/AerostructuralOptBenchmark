# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import tempfile
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import floats

# ==============================================================================
# Extension modules
# ==============================================================================
from compareRunOutputs import (
    DEFAULT_ATOL,
    DEFAULT_RTOL,
    compareRunDirs,
    compareValues,
    normaliseStdoutLines,
    parseListBlocks,
)

RUN_DIR = "/some/output/dir/refactor/baseline/check"


class TestParseListBlocks(unittest.TestCase):
    def test_grad_funcs_block_parsed_as_set(self):
        """Assert that the items listed under a "Grad funcs:" title are returned as a set keyed by the title."""
        text = "Grad funcs:\n  - b\n  - a\n"
        self.assertEqual(parseListBlocks(text), {"Grad funcs": {"a", "b"}})

    def test_title_without_items_gives_empty_set(self):
        """
        Given a performance constraint that depends on no performance inputs and no design variables,
        when its printed blocks are parsed,
        then both titles are present with empty sets rather than missing.
        """
        text = "Performance constraint c depends on performance inputs:\nand thus on design variables:\n"
        self.assertEqual(
            parseListBlocks(text),
            {
                "Performance constraint c depends on performance inputs": set(),
                "Performance constraint c design variables": set(),
            },
        )

    def test_design_variables_keyed_by_preceding_performance_constraint(self):
        """
        Given two performance-constraint blocks, each followed by its own "and thus on design variables:" list,
        when the text is parsed,
        then each design-variable list is keyed by the name of the performance constraint printed before it.
        """
        text = (
            "\n\nPerformance constraint con1 depends on performance inputs:\n"
            "- input1\n"
            "and thus on design variables:\n"
            "- dv1\n"
            "\n\nPerformance constraint con2 depends on performance inputs:\n"
            "- input2\n"
            "and thus on design variables:\n"
            "- dv2\n"
            "- dv3\n"
        )
        self.assertEqual(
            parseListBlocks(text),
            {
                "Performance constraint con1 depends on performance inputs": {"input1"},
                "Performance constraint con1 design variables": {"dv1"},
                "Performance constraint con2 depends on performance inputs": {"input2"},
                "Performance constraint con2 design variables": {"dv2", "dv3"},
            },
        )

    def test_repeated_title_items_are_unioned(self):
        """Assert that when the same title appears twice in one text, its items from both occurrences are combined."""
        text = "Constraints:\n  - a\n  - b\nsomething else\nConstraints:\n  - b\n  - c\n"
        self.assertEqual(parseListBlocks(text), {"Constraints": {"a", "b", "c"}})

    def test_unrecognised_colon_lines_are_ignored(self):
        """Assert that a line ending in a colon that is not a known list title does not produce a block."""
        text = "Gradients of x:\n  dx/dy = 1\n"
        self.assertEqual(parseListBlocks(text), {})


class TestNormaliseStdoutLines(unittest.TestCase):
    def test_run_dir_replaced(self):
        """Assert that every occurrence of the run directory in a line is replaced by a placeholder."""
        text = f"Writing {RUN_DIR}/a.dat and {RUN_DIR}/b.dat\n"
        counts = normaliseStdoutLines(text, RUN_DIR)
        self.assertEqual(counts["Writing <RUN_DIR>/a.dat and <RUN_DIR>/b.dat"], 1)
        self.assertEqual(sum(counts.values()), 1)

    def test_volatile_lines_dropped(self):
        """
        Given lines starting with a time or elapsed-time report and a line containing a memory address,
        when the stdout text is normalised,
        then only the remaining stable line is counted.
        """
        text = "Time to solve: 1.2 s\nelapsed 3.4\n<object at 0x7f3a2b1c9d00>\nstable line\n"
        self.assertEqual(normaliseStdoutLines(text, RUN_DIR), normaliseStdoutLines("stable line\n", RUN_DIR))

    def test_echoed_output_argument_dropped(self):
        """Assert that the echoed `--output` argument line of a run command, which names the run's label, is dropped."""
        text = "    --output refactor/baseline/structCheck \\\nstable line\n"
        self.assertEqual(normaliseStdoutLines(text, RUN_DIR), normaliseStdoutLines("stable line\n", RUN_DIR))

    def test_line_order_ignored(self):
        """Assert that two texts containing the same lines in a different order normalise to equal results."""
        self.assertEqual(normaliseStdoutLines("a\nb", RUN_DIR), normaliseStdoutLines("b\na", RUN_DIR))


class TestCompareValues(unittest.TestCase):
    @given(arrays(np.float64, array_shapes(), elements=floats(allow_nan=True, allow_infinity=False)))
    def test_value_equals_itself(self, values):
        """Assert that any float array, including ones containing NaN, compares equal to a copy of itself."""
        self.assertEqual(compareValues({"x": values}, {"x": values.copy()}, DEFAULT_RTOL, DEFAULT_ATOL), [])

    def test_difference_beyond_tolerance_reported_with_path(self):
        """Assert that a value differing by more than the tolerance gives one message naming the value's key."""
        messages = compareValues({"x": np.array([1.0])}, {"x": np.array([1.1])}, DEFAULT_RTOL, DEFAULT_ATOL)
        self.assertEqual(len(messages), 1)
        self.assertIn("x", messages[0])

    def test_difference_within_tolerance_ignored(self):
        """Assert that a value differing by less than the relative tolerance gives no messages."""
        ref = np.array([1.0, 2.0])
        new = ref * (1.0 + 0.1 * DEFAULT_RTOL)
        self.assertEqual(compareValues({"x": ref}, {"x": new}, DEFAULT_RTOL, DEFAULT_ATOL), [])

    def test_missing_and_extra_keys_both_reported(self):
        """
        Given a reference dict with a key the new dict lacks, and a new dict with a key the reference lacks,
        when they are compared,
        then each of the two keys is named in its own message.
        """
        messages = compareValues(
            {"common": 1.0, "onlyRef": 1.0}, {"common": 1.0, "onlyNew": 1.0}, DEFAULT_RTOL, DEFAULT_ATOL
        )
        self.assertEqual(len(messages), 2)
        self.assertTrue(any("onlyRef" in message for message in messages))
        self.assertTrue(any("onlyNew" in message for message in messages))

    def test_nested_metadata_dict_compared(self):
        """
        Given OpenMDAO-style metadata dicts holding a value under "val",
        when the values differ,
        then one message is produced and it names the dotted path to the value.
        """
        ref = {"y": {"val": np.array([1.0]), "prom_name": "y"}}
        new = {"y": {"val": np.array([2.0]), "prom_name": "y"}}
        messages = compareValues(ref, new, DEFAULT_RTOL, DEFAULT_ATOL)
        self.assertEqual(len(messages), 1)
        self.assertIn("y.val", messages[0])

    def test_shape_mismatch_reported_not_raised(self):
        """Assert that comparing arrays with different shapes gives a message rather than raising an exception."""
        messages = compareValues({"x": np.zeros(3)}, {"x": np.zeros((3, 2))}, DEFAULT_RTOL, DEFAULT_ATOL)
        self.assertEqual(len(messages), 1)
        self.assertIn("x", messages[0])

    def test_non_numeric_values_compared_with_equality(self):
        """Assert that non-numeric values such as strings give a message only when they are unequal."""
        self.assertEqual(compareValues({"u": "m"}, {"u": "m"}, DEFAULT_RTOL, DEFAULT_ATOL), [])
        messages = compareValues({"u": "m"}, {"u": "ft"}, DEFAULT_RTOL, DEFAULT_ATOL)
        self.assertEqual(len(messages), 1)
        self.assertIn("u", messages[0])


class TestCompareRunDirs(unittest.TestCase):
    def test_missing_directory_reported(self):
        """
        Given a reference or new run directory that does not exist,
        when the two directories are compared,
        then a difference is reported instead of the comparison passing with nothing to compare.
        """
        with tempfile.TemporaryDirectory() as existingDir:
            missingDir = os.path.join(existingDir, "missing")
            self.assertEqual(len(compareRunDirs(missingDir, existingDir, DEFAULT_RTOL, DEFAULT_ATOL, True)), 1)
            self.assertEqual(len(compareRunDirs(existingDir, missingDir, DEFAULT_RTOL, DEFAULT_ATOL, True)), 1)


if __name__ == "__main__":
    unittest.main()
