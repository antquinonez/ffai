"""Validate a notebook by running all code cells via exec().

Usage:
    python nb_validate.py <notebook.ipynb> [--cwd <dir>]

Exits 0 on success, 1 on any cell failure. The --cwd flag changes directory
before execution to test CWD-dependent imports.
"""

import argparse
import os
import sys
from pathlib import Path

import nbformat


def validate_notebook(nb_path: str, cwd: str | None = None) -> None:
    resolved = Path(nb_path).resolve()

    if cwd:
        os.chdir(cwd)

    nb = nbformat.read(resolved, as_version=4)
    exec_globals: dict = {"__name__": "__main__"}

    for i, cell in enumerate(nb.cells):
        if cell.cell_type != "code":
            continue
        try:
            exec(cell.source, exec_globals)
        except Exception as e:
            print(f"Cell {i} FAILED: {e}")
            sys.exit(1)

    cwd_info = f" (CWD = {cwd})" if cwd else ""
    print(f"All code cells executed successfully{cwd_info}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate notebook cells via exec()")
    parser.add_argument("notebook", help="Path to .ipynb file")
    parser.add_argument("--cwd", help="Change to this directory before executing")
    args = parser.parse_args()
    validate_notebook(args.notebook, args.cwd)


if __name__ == "__main__":
    main()
