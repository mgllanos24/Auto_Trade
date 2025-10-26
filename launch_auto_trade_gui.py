"""Double-clickable launcher for the Auto Trade desktop GUI.

This helper wraps the existing ``Trading_gui.py`` module so it can be
run directly from the file manager.  Simply double click this file to
start the GUI.  Any exceptions are surfaced in the console window so you
can troubleshoot missing dependencies or configuration.
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
import traceback


def main() -> int:
    """Launch the Tkinter trading GUI in the current Python environment."""
    project_root = Path(__file__).resolve().parent
    target_script = project_root / "Trading_gui.py"

    if not target_script.exists():
        print("Unable to locate Trading_gui.py next to this launcher.")
        print("Please keep launch_auto_trade_gui.py in the project root.")
        return 1

    os.chdir(project_root)

    try:
        runpy.run_path(str(target_script), run_name="__main__")
    except FileNotFoundError as exc:
        print("Python could not start the GUI because a required file was missing:")
        print(exc)
        return 1
    except Exception:  # pragma: no cover - defensive CLI wrapper
        print("An unexpected error occurred while launching the Auto Trade GUI.")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit_code = main()
    if sys.platform.startswith("win") and sys.stdin.isatty():
        # When double-clicked on Windows the console would close immediately.
        # Giving the user a chance to read any error messages makes debugging easier.
        if exit_code != 0:
            input("Press Enter to close this window...")
    sys.exit(exit_code)
