"""Execute a notebook via nbconvert --execute, embedding outputs in the file.

Usage:
    python nb_execute.py <notebook.ipynb> [--timeout 120]

Uses an empty JUPYTER_CONFIG_DIR to bypass broken global configs.
Overwrites the input file with the executed version (outputs embedded).
"""

import argparse
import os
import subprocess
import sys
import tempfile


def execute_notebook(nb_path: str, timeout: int = 120) -> None:
    empty_config = tempfile.mkdtemp(prefix="empty_jupyter_config_")

    cmd = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.timeout={timeout}",
        nb_path,
        "--output",
        os.path.basename(nb_path),
    ]

    env = {**os.environ, "JUPYTER_CONFIG_DIR": empty_config}

    result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)

    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        print(result.stdout)
        sys.exit(1)

    print(f"Executed: {nb_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Execute notebook via nbconvert with output embedding"
    )
    parser.add_argument("notebook", help="Path to .ipynb file")
    parser.add_argument(
        "--timeout", type=int, default=120, help="Execution timeout in seconds"
    )
    args = parser.parse_args()
    execute_notebook(args.notebook, args.timeout)


if __name__ == "__main__":
    main()
