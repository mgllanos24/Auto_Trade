"""Test environment bootstrap helpers.

Ensures the lightweight compatibility shims ship with the project are always
importable.  Some execution environments manipulate ``sys.path`` which can
prevent the modules from being discovered automatically – we import them
manually here and register them in :data:`sys.modules`.
"""
from __future__ import annotations

import importlib.machinery
import os
import sys
import types

ROOT = os.path.dirname(__file__)


def _load_module(name: str, relative_path: str) -> None:
    module_path = os.path.join(ROOT, relative_path)
    if name in sys.modules:
        return
    loader = importlib.machinery.SourceFileLoader(name, module_path)
    module = types.ModuleType(loader.name)
    loader.exec_module(module)
    sys.modules[name] = module


_load_module("numpy", os.path.join("numpy", "__init__.py"))
_load_module("pandas", "pandas.py")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

