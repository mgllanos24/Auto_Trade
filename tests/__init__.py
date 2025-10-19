"""Test helpers to ensure compatibility shims are available when third-party
libraries are not installed in the execution environment."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - import guard
    import pandas  # type: ignore # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - executed in CI only
    from . import _pandas_stub as pandas_stub

    sys.modules["pandas"] = pandas_stub

try:  # pragma: no cover - import guard
    import requests  # type: ignore # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - executed in CI only
    from . import _requests_stub as requests_stub

    sys.modules["requests"] = requests_stub
