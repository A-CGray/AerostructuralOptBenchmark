"""
==============================================================================
Compare the outputs of two runscript runs
==============================================================================
@File    :   compareRunOutputs.py
@Date    :   2026/10/29
@Author  :   Alasdair Christison Gray
@Description : Compare two run directories to check that a refactor of the runscripts changed no behaviour. The
printed lists in each stdout.out are compared as sets, the remaining stdout lines are compared after normalisation,
and Outputs.pkl files and the first pyOptSparse history entries are compared to a tolerance.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import collections
import re

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np

DEFAULT_RTOL = 1e-10
DEFAULT_ATOL = 1e-12

PERFORMANCE_CONSTRAINT_TITLE = re.compile(r"Performance constraint (\S+) depends on performance inputs:")
PERFORMANCE_CONSTRAINT_DVS_TITLE = re.compile(r"and thus on design variables:")

# Titles of the lists the runscript prints, each followed by lines of the form "- item"
LIST_TITLE_PATTERNS = [
    re.compile(r"Grad funcs:"),
    re.compile(r"Disp funcs:"),
    re.compile(r"Design variables:"),
    re.compile(r"Constraints:"),
    re.compile(r"Objectives:"),
    PERFORMANCE_CONSTRAINT_TITLE,
    PERFORMANCE_CONSTRAINT_DVS_TITLE,
]
LIST_ITEM_PATTERN = re.compile(r"^\s*- (.+)$")

# stdout lines that change from run to run without the behaviour changing
VOLATILE_LINE_PATTERNS = [
    re.compile(r"(?i)^time"),
    re.compile(r"(?i)^elapsed"),
    re.compile(r"0x[0-9a-fA-F]{6,}"),
]

RUN_DIR_PLACEHOLDER = "<RUN_DIR>"


def parseListBlocks(text: str) -> dict[str, set[str]]:
    """Parse the lists printed by the runscript into sets of items keyed by their titles

    The design variables a performance constraint depends on are keyed by
    "Performance constraint <name> design variables", using the most recent performance constraint title.

    Parameters
    ----------
    text : str
        Contents of a stdout file

    Returns
    -------
    dict[str, set[str]]
        Items of each list, a title that appears more than once has its items unioned
    """
    blocks = {}
    currentKey = None
    performanceConstraintName = None
    for line in text.splitlines():
        strippedLine = line.strip()
        titlePattern = None
        titleMatch = None
        for pattern in LIST_TITLE_PATTERNS:
            titleMatch = pattern.fullmatch(strippedLine)
            if titleMatch is not None:
                titlePattern = pattern
                break

        if titleMatch is not None:
            if titlePattern is PERFORMANCE_CONSTRAINT_TITLE:
                performanceConstraintName = titleMatch.group(1)
            if titlePattern is PERFORMANCE_CONSTRAINT_DVS_TITLE:
                currentKey = f"Performance constraint {performanceConstraintName} design variables"
            else:
                currentKey = strippedLine.removesuffix(":")
            blocks.setdefault(currentKey, set())
            continue

        itemMatch = LIST_ITEM_PATTERN.match(line)
        if currentKey is not None and itemMatch is not None:
            blocks[currentKey].add(itemMatch.group(1))
        else:
            currentKey = None
    return blocks


def normaliseStdoutLines(text: str, runDir: str) -> collections.Counter:
    """Count the lines of a stdout file after removing everything that changes between identical runs

    Parameters
    ----------
    text : str
        Contents of a stdout file
    runDir : str
        Run directory, replaced by a placeholder wherever it appears

    Returns
    -------
    collections.Counter
        Number of times each normalised line appears
    """
    counts = collections.Counter()
    for line in text.splitlines():
        normalisedLine = line.replace(runDir, RUN_DIR_PLACEHOLDER).rstrip()
        isVolatile = any(pattern.search(normalisedLine) is not None for pattern in VOLATILE_LINE_PATTERNS)
        if not isVolatile:
            counts[normalisedLine] += 1
    return counts


def _getNumericArray(value):
    """Return value as a numeric numpy array, or None if it isn't numeric"""
    try:
        array = np.asarray(value)
    except (ValueError, TypeError):
        return None
    if np.issubdtype(array.dtype, np.number) or np.issubdtype(array.dtype, np.bool_):
        return array
    return None


def _areEqual(ref, new) -> bool:
    """Compare two non-numeric values with ==, reducing elementwise results to a single bool"""
    try:
        return bool(np.all(ref == new))
    except (ValueError, TypeError):
        return False


def compareValues(ref, new, rtol: float, atol: float, path: str = "") -> list[str]:
    """Compare two (possibly nested) values and describe every difference

    Dicts are compared key by key, numeric values to the given tolerance (with NaN equal to NaN) and anything else
    with ==.

    Parameters
    ----------
    ref : Any
        Reference value
    new : Any
        Value to compare against the reference
    rtol : float
        Relative tolerance for numeric values
    atol : float
        Absolute tolerance for numeric values
    path : str, optional
        Dotted path to these values, used to prefix each message

    Returns
    -------
    list[str]
        One message per difference, each starting with the dotted path to the value, empty if there are none
    """
    pathName = path if len(path) > 0 else "<root>"

    if isinstance(ref, dict) and isinstance(new, dict):
        messages = []
        for key in ref:
            childPath = f"{path}.{key}" if len(path) > 0 else str(key)
            if key in new:
                messages += compareValues(ref[key], new[key], rtol, atol, childPath)
            else:
                messages.append(f"{childPath}: missing from new")
        for key in new:
            if key not in ref:
                childPath = f"{path}.{key}" if len(path) > 0 else str(key)
                messages.append(f"{childPath}: only in new")
        return messages

    if isinstance(ref, dict) or isinstance(new, dict):
        return [f"{pathName}: type differs, ref is {type(ref).__name__}, new is {type(new).__name__}"]

    refArray = _getNumericArray(ref)
    newArray = _getNumericArray(new)
    if refArray is not None and newArray is not None:
        if refArray.shape != newArray.shape:
            return [f"{pathName}: shape differs, ref {refArray.shape}, new {newArray.shape}"]
        if not np.allclose(newArray, refArray, rtol=rtol, atol=atol, equal_nan=True):
            maxAbsDiff = np.nanmax(np.abs(newArray - refArray)) if refArray.size > 0 else 0.0
            return [f"{pathName}: values differ, max abs difference {maxAbsDiff:.3e}"]
        return []

    if not _areEqual(ref, new):
        return [f"{pathName}: ref {ref!r} != new {new!r}"]
    return []
