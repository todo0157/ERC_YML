"""
Runner for PM feature extraction script.

Why this exists:
- On Windows PowerShell, paths containing Korean characters and '&' may get mangled
  depending on the console code page / encoding, causing Python to fail opening
  the target script.
- This runner builds the path using Python (Unicode-safe) and executes the script
  via runpy.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    # NOTE: kept for backward-compat. If your script is already renamed to an ASCII-only path,
    # you can run it directly (e.g. `python "ERC\\code\\PM_dataprocessing feature extraction_v1.0.py"`).
    target = repo_root / "ERC" / "code" / "PM_dataprocessing feature extraction_v1.0.py"
    if not target.exists():
        raise FileNotFoundError(f"Target script not found: {target}")

    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()


