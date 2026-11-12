# ==============================================================================
# Standard Python modules
# ==============================================================================
import os
import sys

# Put the repo root and runscripts/ at the front of the path so tests can import the runscript modules directly
TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TESTS_DIR)
sys.path.insert(0, os.path.join(REPO_ROOT, "runscripts"))
sys.path.insert(0, REPO_ROOT)
