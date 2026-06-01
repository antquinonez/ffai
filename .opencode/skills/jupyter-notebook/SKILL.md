---
name: jupyter-notebook
description: >
  Best practices for creating, editing, and validating Jupyter notebooks with
  AI agents. Covers nbformat usage, write-run-describe workflow, AI-readable
  output, visualization libraries, PDF export, and notebook validation. Use
  when creating or modifying .ipynb files, generating notebook-based reports,
  or debugging notebook execution failures.
license: MIT
---

# AI Agent Guide: Creating Reliable Jupyter Notebooks

Best practices for AI coding agents that generate or edit Jupyter notebooks.
These rules prevent the most common defects: stale markdown, unverifiable
output, broken imports, and figures that disappear in PDF export.

## Table of Contents

- [Tool: nbformat with Helper Functions](#tool-nbformat-with-helper-functions)
- [Generator Scripts](#generator-scripts)
- [Write-Run-Describe Rule](#write-run-describe-rule)
- [Full-Notebook Review After Edits](#full-notebook-review-after-edits)
- [AI-Readable Output for Every Code Cell](#ai-readable-output-for-every-code-cell)
- [Visualizations and AI-Readable Data](#visualizations-and-ai-readable-data)
- [Visualization Libraries](#visualization-libraries)
- [Commenting Standards](#commenting-standards)
- [Structuring Notebooks](#structuring-notebooks)
- [Data Sanity Checks Before Analysis](#data-sanity-checks-before-analysis)
- [Import Robustness](#import-robustness)
- [Linting Notebooks](#linting-notebooks)
- [Exporting to HTML and PDF](#exporting-to-html-and-pdf)
- [PDF Formatting](#pdf-formatting)
- [Validation](#validation)

## Tool: nbformat with Helper Functions

Use the `nbformat` library to create/edit notebooks. Direct JSON editing is
fragile due to escaping issues.

### Use helper functions with triple-quoted strings

The recommended pattern uses **named helper functions** and **triple-quoted
strings** for cell content. This avoids quoting hell when cells contain
f-strings, YAML, JSON, or nested quotes:

```python
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = []

def code(s):
    nb.cells.append(nbf.v4.new_code_cell(s))

def md(s):
    nb.cells.append(nbf.v4.new_markdown_cell(s))

md("""\
# Title

Description of the notebook.
""")

code("""\
import pandas as pd

df = pd.read_csv('data.csv')
print(df.head(10).to_string(index=False))
""")

with open('notebook.ipynb', 'w') as f:
    nbf.write(nb, f)
```

### Why not implicit string concatenation?

The naive approach concatenates strings:

```python
# FRAGILE -- breaks with f-strings, quotes, YAML, JSON
nb.cells.append(nbf.v4.new_code_cell(
    "import pandas as pd\n"
    "df = pd.read_csv('data.csv')\n"
    "print(df.head(10).to_string(index=False))"
))
```

This fails when cell content contains:
- **f-strings**: `'print(f"value: {x}")'` — the `{x}` is ambiguous
- **Nested quotes**: `'print("it's a test")'` — quote escaping errors
- **YAML blocks**: Multi-line YAML in concatenated strings requires `\\n`
  at every line break, making the cell content unreadable
- **Braces**: `'result = {"key": "value"}'` — `{` and `}` in single-quoted
  Python strings conflict with f-string syntax in the surrounding code

Triple-quoted strings avoid all of these: the cell source is a single
multi-line string with no concatenation, no `\\n`, and no quote conflicts.

### Why `def` instead of `lambda`?

Use `def` for the helpers, not `lambda`:

```python
# Bad: ruff flags E731 (do not assign a lambda expression)
code = lambda s: nb.cells.append(nbf.v4.new_code_cell(s))
md = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s))

# Good: named functions pass linting
def code(s):
    nb.cells.append(nbf.v4.new_code_cell(s))

def md(s):
    nb.cells.append(nbf.v4.new_markdown_cell(s))
```

### Embedding YAML, JSON, or other triple-quoted content in cells

When a notebook cell itself contains triple-quoted strings (e.g., embedding
YAML in a Python code cell), escape the inner triple quotes with `\"\"\"`:

```python
code("""\
result = run_sync(ffai.execute_workflow(\"\"\"
workflow:
  name: hello
  prompts:
    - name: greet
      prompt: "What is 2+2?"
\"\"\"))

print(result.results['greet'].response)
""")
```

The outer `"""...\n"""` is the cell source. Inside it, `\"\"\"` becomes `"""`
in the generated notebook cell, so the cell contains valid Python with its own
triple-quoted string.

### The backslash-trailing-newline trick

End triple-quoted strings with `"""\` (backslash before closing `"""`) to
suppress the leading newline:

```python
code("""\
import pandas as pd
df = pd.read_csv('data.csv')
""")
```

Without the `\`, the cell source starts with a blank line.

Ensure `nbformat` is available: `pip install nbformat`

## Generator Scripts

For notebooks that call real APIs, use a **generator script** — a `.py` file
that defines the notebook structure with `nbformat`, then a separate execution
step that populates the outputs. This separates "define structure" from
"populate outputs".

### Why generator scripts?

- **Reproducibility**: Re-run the generator to refresh outputs after code changes
- **Lintability**: Generator scripts are `.py` files — ruff and pyright check them
- **No manual JSON**: The `.ipynb` is generated, never hand-edited
- **Clean diffs**: Output changes show in git without touching the structure

### Pattern

```
scripts/
  _nb_my_notebook.py       # generator script
examples/
  my_notebook/
    my_notebook.ipynb       # generated output
```

The generator script:

```python
# scripts/_nb_my_notebook.py
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = []

def code(s):
    nb.cells.append(nbf.v4.new_code_cell(s))

def md(s):
    nb.cells.append(nbf.v4.new_markdown_cell(s))

md("# My Notebook\n")

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
""")

# ... more cells ...

with open("examples/my_notebook/my_notebook.ipynb", "w") as f:
    nbf.write(nb, f)

print("Created my_notebook.ipynb")
```

### Workflow: generate, then execute

```bash
# Step 1: Generate the notebook (fast, no API calls)
python scripts/_nb_my_notebook.py

# Step 2: Execute to populate outputs (makes API calls)
JUPYTER_CONFIG_DIR=/tmp/empty_jupyter_config \
    jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=120 \
    examples/my_notebook/my_notebook.ipynb \
    --output my_notebook.ipynb
```

The `JUPYTER_CONFIG_DIR` bypass prevents broken global Jupyter configs from
interfering (see [Broken global Jupyter config](#broken-global-jupyter-config)).

### Naming convention

- Generator scripts: `scripts/_nb_<name>.py` (underscore prefix signals "tool")
- Output notebooks: `examples/<name>/<name>.ipynb`

## Write-Run-Describe Rule

The single most common defect in generated notebooks is **text that does not
match code output**. Markdown cells describe results the code never produced,
or quote numbers that differ from what actually prints. This happens when text
and code are written together without verification.

To prevent this, follow this procedure **without exception**:

### Step 1: Write and run the code cell first

Write the code cell, write it to the `.ipynb` file, then **execute it** using
the validation script (see [Validation](#validation)). Capture the actual
output. Do NOT write the markdown description yet.

### Step 2: Write the markdown cell based on actual output

Only after seeing the real output, write the markdown cell that describes what
happened. Use the actual numbers, actual names, actual counts. If the output
says "Communities found: 63", the markdown must say 63 -- not "approximately
60" or "over 50" unless you re-run and confirm those characterizations.

### Step 3: Verify text-code agreement

After writing markdown for a section, re-read both the code output and the
markdown cell. Check every concrete claim:

- Every number in markdown appears in the output
- Every entity name in markdown appears in the output
- Every "finds", "reveals", "produces" claim has corresponding output evidence

This applies to **every section**, not just the final one. Do not batch all
markdown cells to the end. Write code -> run -> write markdown -> verify, then
move to the next section.

### What this prevents

| Defect | Cause | Prevention |
|--------|-------|------------|
| Markdown says "8 clusters" but code outputs 12 | Text written before code ran | Run code first, copy actual count |
| Markdown describes "the largest group" but names the wrong entity | Agent guessed at output | Read actual output before writing |
| Markdown says "metric X reveals Y as top-ranked" but Y is rank 5 | Agent assumed largest = highest | Check actual ranking |
| Code cell silently fails, markdown describes successful result | Cell wasn't executed | Validation script catches the failure |
| Visualization shows wrong trend, markdown describes the intended one | Neither was verified against data | Data table rule (see Visualizations) |

## Full-Notebook Review After Edits

Editing a single code cell can break downstream cells or invalidate markdown
descriptions in other sections. After any code change to an existing notebook:

1. **Re-run the entire notebook** using the validation script. Do not only run
   the changed cell.
2. **Re-read the full notebook top to bottom**. Check every markdown cell for
   consistency with the code output that precedes it. A change in section 2 can
   invalidate a claim in section 6.
3. **Fix cascading inconsistencies**. If a code change produces different
   output, update every markdown cell that references the old output. Do not
   leave stale claims for "later".

The review is not optional. The notebook is a single document with shared
mutable state across cells -- a change anywhere can affect anything downstream.

## AI-Readable Output for Every Code Cell

The AI agent cannot see rendered charts, HTML tables, or interactive widgets.
A matplotlib PNG, a Plotly figure, a Jupyter rich display -- these are all
opaque. The agent can only verify what `print()` emits to stdout.

This means every code cell that produces results must include an explicit
`print()` call with the data. A bare expression like `df` or `result` at the
end of a cell produces nothing the agent can read.

### Scalar results: formatted `print()`

```python
print(f"Components: {result.component_count}")
print(f"Modularity: {result.modularity:.3f}")
for group in result.groups[:3]:
    print(f"  Group {group.id}: {group.size} members")
```

### Tabular results: pandas DataFrame

```python
import pandas as pd

df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Carol'],
    'score': [0.92, 0.87, 0.81],
    'rank': [1, 2, 3],
})
print(df.to_string(index=False))
```

This prints:

```
 name  score  rank
Alice   0.92     1
  Bob   0.87     2
Carol   0.81     3
```

Why this format:

- **Aligned columns** are unambiguous -- the agent can parse every row
- **`index=False`** removes the meaningless integer index that adds clutter
- **`to_string()`** is pure text, works in both the validation script and
  interactive sessions

For anything tabular with more than 3 rows, use the DataFrame pattern. For a
few key-value pairs, formatted `print()` is acceptable.

## Visualizations and AI-Readable Data

The AI agent cannot see rendered charts. A matplotlib PNG, a Plotly HTML
widget, a seaborn heatmap -- these are opaque pixel data or interactive DOM
that the agent cannot inspect. The agent can only verify the **numbers that
went into** the visualization.

### Rule 1: Print the data table, then plot

Every visualization cell must include two things in this order:

1. A `print()` call (or DataFrame print) showing the exact data being plotted
2. The plotting code that consumes the same data

This gives the agent empirical evidence of what the chart shows before the
opaque rendering happens.

```python
import pandas as pd
import matplotlib.pyplot as plt

# Build the data
rows = []
for item in items:
    rows.append({'name': item.name, 'score': item.score})
df = pd.DataFrame(rows)

# Print the data table (agent-verifiable)
print(df.to_string(index=False))

# Plot the same data
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(df['name'], df['score'])
ax.set_xlabel('Score')
ax.set_title('Top Items by Score')
plt.tight_layout()
plt.savefig('scores.png', dpi=150)
plt.show()
```

### Rule 2: Do not describe visualization content without the data table

If a markdown cell says "the bar chart shows Alice as the highest-scoring
entity", there must be a preceding `print()` output showing "Alice 0.92" or
similar. The agent writes the markdown description based on the printed data
table, not by imagining what the chart looks like.

### Rule 3: Save static exports for self-containment

For interactive libraries (Plotly, Altair), also save a static export so the
notebook remains self-contained:

```python
fig.write_html('chart.html')       # Plotly interactive
fig.write_image('chart.png')       # Plotly static (requires kaleido)
fig.save('chart.png')              # Altair static
```

### Rule 4: Saved figure files must be noted in markdown

If the code saves a figure to disk, the markdown cell after it should note the
file path so the user knows where to find it:

```markdown
The chart above is saved to `figures/scores.png`.
```

## Visualization Libraries

### Preferred stack

- **plotly**: Bar charts, scatter plots, line charts, heatmaps. Interactive by
  default (hover, zoom, pan). Saves to HTML for self-contained notebooks.
- **seaborn**: Statistical visualizations -- distribution plots, pair plots,
  correlation heatmaps, box/violin plots. Produces publication-quality
  statistical graphics with less code than raw matplotlib.
- **matplotlib**: Fallback only. Use for simple quick-and-dirty plots or when
  other libraries are unavailable.

```bash
pip install plotly seaborn kaleido
```

### When visualizations are required

Every notebook section that presents a ranked list, comparison, or distribution
of more than 5 items must include a visualization alongside the printed data
table. Text tables alone do not communicate patterns effectively.

Required visualization types by data shape:

| Data shape | Visualization | Library |
|-----------|---------------|---------|
| Ranked list with scores | Horizontal bar chart | plotly `px.bar(orientation='h')` |
| Two continuous variables | Scatter plot | plotly `px.scatter` |
| Distribution of a single variable | Histogram or KDE | seaborn `histplot` / `kdeplot` |
| Correlation between variables | Correlation heatmap | seaborn `heatmap` |
| Comparison across categories | Grouped bar or box plot | plotly or seaborn |
| Time series or ordered sequence | Line chart with markers | plotly `px.line(markers=True)` |

### Visualization density rule

A notebook with 7+ analytical sections should have at minimum 3 visualizations.
Data patterns that take paragraphs to describe in text can be seen instantly in
a chart.

### Plotly in headless validation

The `exec()` validation script runs headless. Plotly figures need a renderer
override:

```python
import plotly.io as pio
pio.renderers.default = 'png'

fig.show()  # renders to PNG in headless, interactive in Jupyter
```

## Commenting Standards

Notebook code cells may use comments more freely than production code. Each
code cell should include enough comments to explain:

1. **What the cell does** (1-2 lines at the top)
2. **Non-obvious arguments** (parameter values that aren't self-explanatory)
3. **Why this approach** (design rationale when the choice isn't obvious)

Example:

```python
# Compute degree centrality for all nodes, then filter to active institutions
# degree = count of edges incident on a node (higher = more connected)
degree = graph.centrality(method='degree')
active = {k: v for k, v in degree.items() if k in institution_labels}
```

## Structuring Notebooks

- **One concept per cell**: Each code cell demonstrates a single idea
- **Markdown headers**: Use `## Section N: Title` for section breaks
- **Executable**: Every code cell must run independently in sequence
- **No `print()` as documentation**: Use markdown cells for explanation, code
  cells for execution
- **Remove `if __name__ == "__main__":` guards**: Notebooks execute cells
  directly
- Include all necessary imports in the first code cell or in the cell where
  they are first used
- Include helper functions in a dedicated cell before the cells that call them

### Helper function validation

When a notebook defines non-trivial helper functions (functions that filter
datasets, transform data structures, run simulations), validate the helper cell
**in isolation** before writing the analysis cells that depend on it.

**Validation procedure**:

1. Write the helper function cell.
2. Write a minimal test cell immediately after it that calls the helper with
   known inputs and asserts the result. This test cell is temporary.
3. Run both cells. If the helper has a bug, it surfaces here -- not 10 cells
   later.
4. Once the helper passes, remove the temporary test cell and proceed with
   the analysis cells.

This only applies to helpers with real logic (data filtering, simulation,
graph mutation). Trivial helpers (formatting, string manipulation) don't need
isolated validation.

### Notebook-specific pitfalls

- **Shared mutable state**: All code cells share the same execution namespace.
  An object created in cell 3 is mutated by cell 7. If cell 7 runs without
  cell 3, it fails. This is by design -- do not add defensive
  re-initialization.
- **Non-deterministic outputs**: Algorithms like label propagation or random
  sampling produce different results across runs. In markdown cells, describe
  results as ranges or note non-determinism explicitly. Do not hardcode exact
  counts from a single run as permanent facts.
- **Print output is cell output**: In the validation script, `print()` goes to
  stdout which the script captures. Use `print()` for values you want to appear
  as output. Use markdown cells for narrative.
- **`asyncio.run()` fails inside Jupyter**: Jupyter notebooks run inside an
  async event loop. Calling `asyncio.run()` raises
  `RuntimeError: asyncio.run() cannot be called from a running event loop`.
  **Before adding your own `run_sync` helper, check if the project already
  provides one** (e.g. search for `def run_sync` in the codebase). If it does,
  import and use it. If not, use this pattern:

  ```python
  import asyncio
  import concurrent.futures

  def run_sync(coro):
      try:
          asyncio.get_running_loop()
      except RuntimeError:
          return asyncio.run(coro)
      with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
          return pool.submit(asyncio.run, coro).result()
  ```

  This delegates to `asyncio.run()` when no loop is running (plain Python
  scripts, `exec()` validation), and runs the coroutine in a background thread
  when a loop is already running (Jupyter, IPython). Never use `asyncio.run()`
  directly in notebook code.

## Data Sanity Checks Before Analysis

Before any analytical section, print basic data diagnostics so the agent can
verify the data loaded correctly and catch silent problems early. This is not
about choosing analytical methods — it is about preventing the code from
running successfully on corrupted or misinterpreted data.

### Required diagnostics after loading data

After reading a dataset into a DataFrame, always print:

```python
print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nDtypes:\n{df.dtypes.to_string()}")
print(f"\nNull counts:\n{df.isnull().sum().to_string()}")
print(f"\nFirst 5 rows:\n{df.head().to_string(index=False)}")
```

### What this catches

| Problem | Symptom | Caught by |
|---------|---------|-----------|
| Numeric columns loaded as strings | `"1,234"` instead of `1234` | `dtypes` check |
| Mixed types in a column | Silent coercion to `object` | `dtypes` check |
| Unexpected null counts | Analysis silently drops rows | `isnull().sum()` |
| Wrong encoding or delimiter | Garbage column names, single-column DataFrame | `head()` + `shape` |
| Empty DataFrame from bad query | Downstream code runs on nothing | `shape` check |

### When to add more checks

- **Categorical columns**: Print `df['col'].value_counts()` to verify expected
  categories and catch typos or encoding artifacts.
- **Numeric ranges**: Print `df.describe().to_string()` to catch impossible
  values (negative ages, percentages above 100, etc.).
- **ID uniqueness**: Print `df['id'].nunique()` vs `len(df)` to catch
  duplicate rows before joining.

These checks are notebook mechanics — they prevent the agent from building
analysis on a broken foundation. Choosing what to do about the findings
(clean, impute, exclude) is the user's decision.

## Import Robustness

Jupyter's working directory is unpredictable: it may be the project root, the
notebook's own directory, or somewhere else entirely. The `exec()` validation
script runs from a fixed directory and does not enforce Python's package import
rules, so dotted imports that pass validation can fail in a real Jupyter
kernel.

### Rule 1: Use project root discovery, not CWD assumptions

Do not assume `Path.cwd()` is any particular directory. Walk up from CWD to
find the project root using a marker file or directory you know exists:

```python
import sys
from pathlib import Path

_cwd = Path().resolve()
_project_root = _cwd
for _p in [_cwd, *list(_cwd.parents)]:
    if (_p / 'pyproject.toml').is_file():   # or setup.cfg, .git, etc.
        _project_root = _p
        break

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
```

Choose a marker that uniquely identifies your project root. Common options:
`pyproject.toml`, `setup.py`, `setup.cfg`, `.git` directory, `requirements.txt`.

### Rule 2: Import sibling modules directly

When a notebook lives alongside a `.py` file (e.g., `pipeline.py`,
`utils.py`), add that directory to `sys.path` and import the module directly.
Do not use deep dotted package imports because ancestor directories may lack
`__init__.py`.

```python
# Bad: requires __init__.py in every ancestor directory
from examples.analysis.data_processing.pipeline import Pipeline

# Good: add the directory directly, import top-level module name
_notebook_dir = str(Path().resolve().parent)  # or wherever the module lives
if _notebook_dir not in sys.path:
    sys.path.insert(0, _notebook_dir)

from pipeline import Pipeline
```

### Rule 3: Multi-CWD validation

The standard `exec()` validation always runs from a fixed directory, which
masks CWD-dependent failures. After the standard validation passes, also
validate with CWD set to the notebook's directory:

```bash
python -c "
import nbformat, sys, os
from pathlib import Path

os.chdir('path/to/notebook/directory')

nb = nbformat.read(Path('notebook.ipynb'), as_version=4)
exec_globals = {'__name__': '__main__'}
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        try:
            exec(cell.source, exec_globals)
        except Exception as e:
            print(f'Cell {i} FAILED: {e}')
            sys.exit(1)
print('All code cells executed successfully (CWD = notebook dir)')
"
```

### What this prevents

| Defect | Cause | Prevention |
|--------|-------|------------|
| `ModuleNotFoundError` for deep package path | Missing `__init__.py` in ancestor dirs | Import sibling modules directly via `sys.path` |
| Import works in validation but fails in Jupyter | Validation CWD differs from Jupyter CWD | Walk up to find project root |
| Notebook works on one machine but not another | Hardcoded relative path assumed specific CWD | Project root discovery from CWD |

## Linting Notebooks

`ruff check` lints `.ipynb` files natively. Every notebook cell must satisfy
the same lint rules as `.py` files. Common issues:

| Ruff rule | Typical cause in notebooks |
|-----------|---------------------------|
| `F401` unused import | Imported a symbol "for later" but never used it |
| `I001` unsorted imports | `# noqa: E402` imports not grouped with other `noqa` imports |
| `F541` f-string without placeholders | `f"literal string"` with no `{}` variables |
| `E731` lambda assignment | Generator script uses `lambda` instead of `def` |

### Workflow

```bash
# Lint notebooks alongside source code
ruff check examples/ ffai/ tests/
```

Fix any errors before committing. The same `ruff` configuration from
`pyproject.toml` applies to notebook cells.

### Generator scripts are lintable too

Generator scripts (`.py` files) are fully lintable. The `def` helper pattern
(see [Tool: nbformat with Helper Functions](#tool-nbformat-with-helper-functions))
exists partly because `ruff` flags `E731` for lambda assignments.

## Exporting to HTML and PDF

### Prerequisites

```bash
# HTML export (works out of the box)
pip install nbconvert

# PDF export via webpdf (requires Playwright + Chromium)
pip install "nbconvert[webpdf]"
playwright install chromium
```

On Linux, Playwright bundles its own Chromium -- no system packages needed.

### Matplotlib figures must be explicitly embedded

The `matplotlib.use('Agg')` backend suppresses display output. `plt.show()`
under Agg does not produce embeddable cell output. To include matplotlib/seaborn
figures in PDF exports, replace `plt.show()` with a bytes-based display pattern:

```python
import io
_img_buf = io.BytesIO()
fig.savefig(_img_buf, format='png', dpi=150, bbox_inches='tight')
_img_buf.seek(0)
from IPython.display import Image as _Img, display as _disp
_disp(_Img(data=_img_buf.getvalue()))
```

Plotly figures with `pio.renderers.default = 'png'` embed automatically via
`fig.show()` -- no extra steps needed.

### Export commands

```bash
# HTML export (code hidden)
jupyter nbconvert --to html --no-input notebook.ipynb

# PDF export (code hidden, executed)
jupyter nbconvert --to webpdf --no-input --execute \
    --ExecutePreprocessor.timeout=180 notebook.ipynb
```

### Artifact organization

Notebooks that generate figure files should save them to a `figures/` subfolder,
not alongside the notebook. Add a `.gitignore`:

```
figures/
*.html
*.pdf
.ipynb_checkpoints/
```

Create the directory in the setup cell:

```python
Path('figures').mkdir(exist_ok=True)
```

Then save all figures there:

```python
fig.write_image('figures/chart.png', scale=2)       # plotly
plt.savefig('figures/chart.png', dpi=150)            # matplotlib
```

## PDF Formatting

The `exec()` validation produces plain text. But PDF export renders through
Chromium with full CSS support. Use CSS for PDF quality, context-aware
`display()` for both contexts.

### CSS style block

Add a `<style>` block in the first markdown cell to control PDF rendering:

```html
<style>
:root {
  --jp-content-font-size1: 11px;
  --jp-code-font-size: 10px;
  --jp-ui-font-size1: 10px;
}
body { font-size: 11px; line-height: 1.4; }
h1 { font-size: 20px !important; }
h2 { font-size: 16px !important; margin-top: 1.2em !important; }
h3 { font-size: 13px !important; }
table { font-size: 9px !important; table-layout: auto !important; }
th, td { font-size: 9px !important; padding: 2px 4px !important; }
.dataframe { width: 100%; }
.jp-Cell { page-break-inside: avoid; }
.page-break { page-break-before: always; }
.jp-OutputArea-output img { max-width: 100%; height: auto; }
pre { font-size: 9px !important; line-height: 1.3 !important; }
@page { margin: 0.6in 0.5in; }
</style>
```

Adjust font sizes and margins to taste.

### Page breaks between sections

Insert page break dividers between major sections:

```html
<div class="page-break"></div>

---

## Section Title
```

### HTML tables via context-aware `display(df.style)`

For PDF output, `display(df.style.format(...))` produces styled HTML tables
with borders and alignment -- much better than `print(df.to_string())`. But in
`exec()` validation, `IPython.display.display` is importable but has no
frontend, so it prints `<Styler object at 0x...>` instead of table content.

Use a **context-aware `display()`** in the setup cell that detects whether it's
running inside a real IPython kernel:

```python
import pandas.io.formats.style as _pd_style
try:
    from IPython import get_ipython
    _in_kernel = get_ipython() is not None
    from IPython.display import display as _ip_display
except (ImportError, NameError):
    _in_kernel = False
    _ip_display = None

def display(x):
    if isinstance(x, _pd_style.Styler) and not _in_kernel:
        print(x.to_string())
    elif _ip_display is not None:
        _ip_display(x)
    else:
        print(x.to_string() if hasattr(x, 'to_string') else x)
```

This works in all three contexts:

- **Real Jupyter kernel** (`_in_kernel=True`): delegates to
  `IPython.display.display` for styled HTML rendering
- **`exec()` validation** (`_in_kernel=False`, IPython importable): prints
  `Styler.to_string()` as plain text
- **No IPython** (`_ip_display=None`): falls back to `print()` for everything

Then use `display()` with styled DataFrames:

```python
display(df.style.format({'assets': '{:.1f}', 'score': '{:.6f}'}).set_caption('Title'))
```

The pattern: **use `print()` for scalar metrics and `display(df.style)` for
tabular data**. Do NOT add a separate `print(df.to_string())` alongside
`display(df.style)` -- the context-aware `display()` handles both contexts
without duplication.

### Broken global Jupyter config

`~/.jupyter/jupyter_nbconvert_config.json` may reference missing packages and
break the export. Bypass with an empty config directory:

```bash
mkdir -p /tmp/empty_jupyter_config
JUPYTER_CONFIG_DIR=/tmp/empty_jupyter_config jupyter nbconvert --to webpdf \
    --no-input --execute --ExecutePreprocessor.timeout=180 notebook.ipynb
```

Always use this bypass for `nbconvert` commands in scripts and CI.

### Wide table handling

Tables with 6+ columns overflow PDF page width. Solutions:

- Split into two tables (summary columns + detail columns)
- Drop columns that are redundant with the visualization
- Use the chart for per-source detail, table for aggregate metrics

## Validation

After creating or modifying notebooks, validate they execute correctly using
**both** methods below. A notebook that passes `exec()` validation can still
fail under `nbconvert --execute` because the two contexts have different
semantics.

### Companion scripts

The skill directory includes reusable validation and execution scripts:

| Script | Purpose |
|--------|---------|
| `nb_validate.py` | `exec()` validation with optional `--cwd` flag |
| `nb_execute.py` | `nbconvert --execute` with config bypass |
| `nb_template.py` | Minimal generator script to copy and adapt |

Usage:

```bash
# Quick validation (fast, no kernel)
python nb_validate.py path/to/notebook.ipynb

# Validate with CWD set to notebook's directory
python nb_validate.py path/to/notebook.ipynb --cwd path/to/

# Full execution with output embedding (slow, real kernel)
python nb_execute.py path/to/notebook.ipynb --timeout 120
```

### Method 1: `exec()` validation (fast, no kernel)

Runs code cells in a plain Python `exec()` context. No event loop, no IPython
kernel. Best for quick iteration during development.

```bash
python -c "
import nbformat, sys
from pathlib import Path
nb_path = Path('path/to/notebook.ipynb')
nb = nbformat.read(nb_path, as_version=4)
exec_globals = {'__name__': '__main__'}
for i, cell in enumerate(nb.cells):
    if cell.cell_type == 'code':
        try:
            exec(cell.source, exec_globals)
        except Exception as e:
            print(f'Cell {i} FAILED: {e}')
            sys.exit(1)
print('All code cells executed successfully')
"
```

Also validate with CWD set to the notebook's directory (see Import
Robustness Rule 3).

### Method 2: `nbconvert --execute` (slow, real kernel)

Runs cells inside a real IPython kernel via nbconvert. This is the **only**
way to verify notebooks that use async code (Jupyter kernels have an active
event loop; `exec()` does not).

```bash
JUPYTER_CONFIG_DIR=/tmp/empty_jupyter_config \
    jupyter nbconvert --to notebook --execute \
    --ExecutePreprocessor.timeout=120 \
    path/to/notebook.ipynb \
    --output notebook.ipynb
```

This also writes the executed outputs back into the `.ipynb` file, making it
self-contained with embedded results.

### Key differences between the two methods

| Aspect | `exec()` | `nbconvert --execute` |
|--------|----------|-----------------------|
| Event loop | None | Active IPython kernel |
| `asyncio.run()` | Works | Raises `RuntimeError` |
| `run_sync()` pattern | Works | Works (uses thread fallback) |
| IPython display | No frontend | Full frontend |
| Speed | Fast | Slow (kernel startup) |
| Output embedding | No | Yes (writes to file) |
| Config sensitivity | None | Affected by `~/.jupyter/` config |

**Critical**: Notebook cells share state. Syntax errors in one cell may not
appear until a dependent cell executes. Always run the full notebook after
edits.
