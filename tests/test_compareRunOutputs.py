# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import tempfile
import unittest

# ==============================================================================
# External Python modules
# ==============================================================================
import dill
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import array_shapes, arrays
from hypothesis.strategies import floats
from pyoptsparse import OPT, Optimization

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
HISTORY_FILE_NAME = "AeroStructOpt.hst"
TOY_START_VALUE = 3.0
TOY_BOUND = 10.0
SMALL_DIFFERENCE = 1e-6
SENS_ATOL = 1e-5


def writeToyHistory(historyFile: str, funcOffset: float = 0.0, sensOffset: float = 0.0) -> None:
    """Write a pyOptSparse history file from one SLSQP iteration on f = x^2, offsetting the function and derivative."""

    def computeFuncs(x):
        return {"f": x["x"][0] ** 2 + funcOffset}, False

    def computeSens(x, funcs):
        return {"f": {"x": [2 * x["x"][0] + sensOffset]}}, False

    optProb = Optimization("toy", computeFuncs)
    optProb.addVarGroup("x", 1, lower=-TOY_BOUND, upper=TOY_BOUND, value=TOY_START_VALUE)
    optProb.addObj("f")
    optimiser = OPT("SLSQP", options={"MAXIT": 1, "IPRINT": -1})
    optimiser(optProb, sens=computeSens, storeHistory=historyFile)


def writeOutputsPickle(runDir: str, outputs: dict) -> None:
    """Write an Outputs.pkl file into a run directory."""
    with open(os.path.join(runDir, "Outputs.pkl"), "wb") as file:
        dill.dump(outputs, file)


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
        then two messages are produced and each of the two keys is named in one of them.
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

    def test_bool_difference_reported_not_raised(self):
        """Assert that differing boolean values, scalar or array, give a message rather than raising an exception."""
        self.assertEqual(len(compareValues({"b": True}, {"b": False}, DEFAULT_RTOL, DEFAULT_ATOL)), 1)
        messages = compareValues(
            {"b": np.array([True, False])}, {"b": np.array([True, True])}, DEFAULT_RTOL, DEFAULT_ATOL
        )
        self.assertEqual(len(messages), 1)

    def test_identical_lists_of_arrays_equal(self):
        """
        Given lists and tuples holding arrays of different lengths, as pyOptSparse stores sparse sensitivities,
        when each is compared with an identical copy,
        then no messages are produced.
        """
        values = {
            "csr": [np.array([0, 1, 3]), np.array([0, 1, 2]), np.array([1.0, 2.0, 3.0, np.nan])],
            "pair": (np.zeros(2), "a"),
            "nested": [{"a": np.zeros(3)}],
        }
        copiedValues = {
            "csr": [array.copy() for array in values["csr"]],
            "pair": (np.zeros(2), "a"),
            "nested": [{"a": np.zeros(3)}],
        }
        self.assertEqual(compareValues(values, copiedValues, DEFAULT_RTOL, DEFAULT_ATOL), [])

    def test_lists_of_arrays_differences_reported_with_index(self):
        """Assert that a difference inside a list of arrays gives one message naming the list's key and the index."""
        ref = {"csr": [np.array([0, 1]), np.array([1.0, 2.0, 3.0])]}
        new = {"csr": [np.array([0, 1]), np.array([1.0, 2.0, 4.0])]}
        messages = compareValues(ref, new, DEFAULT_RTOL, DEFAULT_ATOL)
        self.assertEqual(len(messages), 1)
        self.assertIn("csr.1", messages[0])

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

    def test_reference_with_nothing_to_compare_reported(self):
        """
        Given a reference run directory containing no stdout, Outputs.pkl or history files,
        when it is compared with another such directory,
        then a difference is reported instead of the comparison passing with nothing compared.
        """
        with tempfile.TemporaryDirectory() as refDir, tempfile.TemporaryDirectory() as newDir:
            self.assertEqual(len(compareRunDirs(refDir, newDir, DEFAULT_RTOL, DEFAULT_ATOL, True)), 1)

    def test_file_only_in_new_reported(self):
        """
        Given a new run directory with a stdout file that the reference run directory lacks,
        when the two directories are compared,
        then the extra file is reported by its relative path.
        """
        with tempfile.TemporaryDirectory() as refDir, tempfile.TemporaryDirectory() as newDir:
            for runDir in [refDir, newDir]:
                os.makedirs(os.path.join(runDir, "fp1"))
                with open(os.path.join(runDir, "fp1", "stdout.out"), "w") as file:
                    file.write("Grad funcs:\n  - a\n")
            os.makedirs(os.path.join(newDir, "fp2"))
            with open(os.path.join(newDir, "fp2", "stdout.out"), "w") as file:
                file.write("Grad funcs:\n  - a\n")
            messages = compareRunDirs(refDir, newDir, DEFAULT_RTOL, DEFAULT_ATOL, True)
            self.assertEqual(len(messages), 1)
            self.assertIn(os.path.join("fp2", "stdout.out"), messages[0])


class TestCompareRunDirsSensAtol(unittest.TestCase):
    def test_sens_difference_within_sens_atol_ignored(self):
        """
        Given history files whose first derivatives differ by more than atol but less than sensAtol,
        when the run directories are compared with that sensAtol,
        then no difference is reported.
        """
        with tempfile.TemporaryDirectory() as refDir, tempfile.TemporaryDirectory() as newDir:
            writeToyHistory(os.path.join(refDir, HISTORY_FILE_NAME))
            writeToyHistory(os.path.join(newDir, HISTORY_FILE_NAME), sensOffset=SMALL_DIFFERENCE)
            messages = compareRunDirs(refDir, newDir, DEFAULT_RTOL, DEFAULT_ATOL, False, sensAtol=SENS_ATOL)
            self.assertEqual(messages, [])

    def test_sens_difference_reported_without_sens_atol(self):
        """Assert that derivatives are compared with atol when no sensAtol is given."""
        with tempfile.TemporaryDirectory() as refDir, tempfile.TemporaryDirectory() as newDir:
            writeToyHistory(os.path.join(refDir, HISTORY_FILE_NAME))
            writeToyHistory(os.path.join(newDir, HISTORY_FILE_NAME), sensOffset=SMALL_DIFFERENCE)
            messages = compareRunDirs(refDir, newDir, DEFAULT_RTOL, DEFAULT_ATOL, False)
            self.assertEqual(len(messages), 1)
            self.assertIn("funcsSens", messages[0])

    def test_sens_atol_does_not_loosen_function_values(self):
        """
        Given history files whose first function values differ by more than atol but less than sensAtol,
        when the run directories are compared with that sensAtol,
        then the function value difference is still reported.
        """
        with tempfile.TemporaryDirectory() as refDir, tempfile.TemporaryDirectory() as newDir:
            writeToyHistory(os.path.join(refDir, HISTORY_FILE_NAME))
            writeToyHistory(os.path.join(newDir, HISTORY_FILE_NAME), funcOffset=SMALL_DIFFERENCE)
            messages = compareRunDirs(refDir, newDir, DEFAULT_RTOL, DEFAULT_ATOL, False, sensAtol=SENS_ATOL)
            self.assertEqual(len(messages), 1)
            self.assertIn(".funcs.f", messages[0])

    def test_sens_atol_does_not_loosen_outputs(self):
        """
        Given Outputs.pkl files whose values differ by more than atol but less than sensAtol,
        when the run directories are compared with that sensAtol,
        then the output difference is still reported.
        """
        with tempfile.TemporaryDirectory() as refDir, tempfile.TemporaryDirectory() as newDir:
            writeOutputsPickle(refDir, {"thickness": np.array([TOY_START_VALUE])})
            writeOutputsPickle(newDir, {"thickness": np.array([TOY_START_VALUE + SMALL_DIFFERENCE])})
            messages = compareRunDirs(refDir, newDir, DEFAULT_RTOL, DEFAULT_ATOL, False, sensAtol=SENS_ATOL)
            self.assertEqual(len(messages), 1)
            self.assertIn("thickness", messages[0])


if __name__ == "__main__":
    unittest.main()
