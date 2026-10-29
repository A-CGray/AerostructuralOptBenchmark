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
import argparse
import collections
import os
import re
import sys

# ==============================================================================
# External Python modules
# ==============================================================================
import dill
import numpy as np
from pyoptsparse import History

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

STDOUT_FILE_NAME = "stdout.out"
OUTPUTS_FILE_NAME = "Outputs.pkl"
HISTORY_FILE_EXTENSION = ".hst"


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


def readFirstHistoryEntries(historyFile: str) -> tuple[dict, dict]:
    """Read the first function values and the first function sensitivities stored in a pyOptSparse history file

    Parameters
    ----------
    historyFile : str
        Path to the history file

    Returns
    -------
    tuple[dict, dict]
        The ``funcs`` of the first call counter that has them and the ``funcsSens`` of the first call counter that
        has them, each an empty dict if no call counter has them
    """
    hist = History(historyFile, flag="r")
    firstFuncs = None
    firstFuncsSens = None
    for callCounter in sorted(hist.getCallCounters(), key=int):
        entry = hist.read(callCounter)
        if entry is None:
            continue
        if firstFuncs is None and "funcs" in entry:
            firstFuncs = entry["funcs"]
        if firstFuncsSens is None and "funcsSens" in entry:
            firstFuncsSens = entry["funcsSens"]
        if firstFuncs is not None and firstFuncsSens is not None:
            break
    hist.close()
    return (firstFuncs if firstFuncs is not None else {}, firstFuncsSens if firstFuncsSens is not None else {})


def _compareStdoutFiles(
    refFile: str, newFile: str, refDir: str, newDir: str, relPath: str, doCompareStdoutLines: bool
) -> list[str]:
    """Compare the printed lists, and optionally all normalised lines, of two stdout files"""
    with open(refFile, errors="replace") as file:
        refText = file.read()
    with open(newFile, errors="replace") as file:
        newText = file.read()

    messages = []
    refBlocks = parseListBlocks(refText)
    newBlocks = parseListBlocks(newText)
    for key in refBlocks:
        if key not in newBlocks:
            messages.append(f"{relPath}: list '{key}' missing from new")
            continue
        for item in sorted(refBlocks[key] - newBlocks[key]):
            messages.append(f"{relPath}: list '{key}' item only in ref: {item}")
        for item in sorted(newBlocks[key] - refBlocks[key]):
            messages.append(f"{relPath}: list '{key}' item only in new: {item}")
    for key in newBlocks:
        if key not in refBlocks:
            messages.append(f"{relPath}: list '{key}' only in new")

    if doCompareStdoutLines:
        refCounts = normaliseStdoutLines(refText, refDir)
        newCounts = normaliseStdoutLines(newText, newDir)
        for line in sorted(set(refCounts) | set(newCounts)):
            if refCounts[line] != newCounts[line]:
                messages.append(
                    f"{relPath}: line appears {refCounts[line]} times in ref, {newCounts[line]} times in new: {line}"
                )
    return messages


def compareRunDirs(refDir: str, newDir: str, rtol: float, atol: float, doCompareStdoutLines: bool) -> list[str]:
    """Compare every stdout, Outputs.pkl and history file in a reference run directory with the same file in a new one

    Parameters
    ----------
    refDir : str
        Reference run directory
    newDir : str
        New run directory
    rtol : float
        Relative tolerance for numeric values
    atol : float
        Absolute tolerance for numeric values
    doCompareStdoutLines : bool
        Whether to compare all normalised stdout lines as well as the printed lists

    Returns
    -------
    list[str]
        One message per difference, each prefixed with the relative path of the file, empty if there are none
    """
    refDir = os.path.abspath(refDir)
    newDir = os.path.abspath(newDir)
    messages = []
    for runDir in [refDir, newDir]:
        if not os.path.isdir(runDir):
            messages.append(f"{runDir}: directory does not exist")
    if len(messages) > 0:
        return messages

    for dirPath, _, fileNames in sorted(os.walk(refDir)):
        for fileName in sorted(fileNames):
            isStdout = fileName == STDOUT_FILE_NAME
            isOutputs = fileName == OUTPUTS_FILE_NAME
            isHistory = fileName.endswith(HISTORY_FILE_EXTENSION)
            if not (isStdout or isOutputs or isHistory):
                continue

            refFile = os.path.join(dirPath, fileName)
            relPath = os.path.relpath(refFile, refDir)
            newFile = os.path.join(newDir, relPath)
            if not os.path.isfile(newFile):
                messages.append(f"{relPath}: missing from new")
                continue

            if isStdout:
                messages += _compareStdoutFiles(refFile, newFile, refDir, newDir, relPath, doCompareStdoutLines)
            elif isOutputs:
                with open(refFile, "rb") as file:
                    refOutputs = dill.load(file)
                with open(newFile, "rb") as file:
                    newOutputs = dill.load(file)
                messages += compareValues(refOutputs, newOutputs, rtol, atol, relPath)
            else:
                refFuncs, refFuncsSens = readFirstHistoryEntries(refFile)
                newFuncs, newFuncsSens = readFirstHistoryEntries(newFile)
                messages += compareValues(refFuncs, newFuncs, rtol, atol, f"{relPath}.funcs")
                messages += compareValues(refFuncsSens, newFuncsSens, rtol, atol, f"{relPath}.funcsSens")
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare two run directories to check a refactor changed nothing")
    parser.add_argument("refDir", help="Reference run directory")
    parser.add_argument("newDir", help="New run directory")
    parser.add_argument("--rtol", type=float, default=DEFAULT_RTOL, help="Relative tolerance for numeric values")
    parser.add_argument("--atol", type=float, default=DEFAULT_ATOL, help="Absolute tolerance for numeric values")
    parser.add_argument(
        "--compareStdoutLines",
        action="store_true",
        help="Compare all normalised stdout lines, not just the printed lists",
    )
    args = parser.parse_args()

    messages = compareRunDirs(args.refDir, args.newDir, args.rtol, args.atol, args.compareStdoutLines)
    for message in messages:
        print(message)
    if len(messages) == 0:
        print("No differences")
        return 0
    print(f"{len(messages)} differences")
    return 1


if __name__ == "__main__":
    sys.exit(main())
