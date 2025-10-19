"""Pytest configuration ensuring local compatibility shims are available."""
import os
import sys

ROOT = os.path.dirname(__file__)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Import the lightweight modules so they are registered in ``sys.modules``
import numpy  # noqa: F401  # pylint: disable=unused-import
import pandas  # noqa: F401  # pylint: disable=unused-import

