"""Minimal generator script template for creating notebooks.

Copy and adapt this script. Run it to generate the .ipynb, then use
nb_execute.py to populate outputs:

    python _nb_my_notebook.py
    python nb_execute.py ../examples/my_notebook/my_notebook.ipynb

The script lives in scripts/ (or wherever generator scripts go).
Adjust the output path, cell content, and imports to match your project.
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = []


def code(s):
    nb.cells.append(nbf.v4.new_code_cell(s))


def md(s):
    nb.cells.append(nbf.v4.new_markdown_cell(s))


md("""\
# Notebook Title

Description of what this notebook demonstrates.

<div class="page-break"></div>

---
""")

code("""\
import sys
from pathlib import Path

_cwd = Path().resolve()
_project_root = _cwd
for _p in [_cwd, *list(_cwd.parents)]:
    if (_p / 'pyproject.toml').is_file():
        _project_root = _p
        break

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

# Add your imports here
print("Setup complete")
""")

md("""\
<div class="page-break"></div>

---

## Step 1: Title

Description of this step.
""")

code("""\
# Your code here
print("Hello from notebook!")
""")

md("""\
## Summary

- Point 1
- Point 2
""")

with open("examples/my_notebook/my_notebook.ipynb", "w") as f:
    nbf.write(nb, f)

print("Created my_notebook.ipynb")
