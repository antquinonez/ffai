import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = []

def code(s):
    nb.cells.append(nbf.v4.new_code_cell(s))

def md(s):
    nb.cells.append(nbf.v4.new_markdown_cell(s))

md("""\
# YAML Workflow Basics

This notebook demonstrates FFAI's YAML workflow system — defining multi-step
LLM pipelines in YAML files and executing them with a single call.

Topics covered:

1. **Loading and validating** a YAML workflow
2. **Executing** a simple single-step workflow
3. **Sequential chains** with `{{name.response}}` interpolation
4. **Variable substitution** with `{variable}` placeholders
5. **Conditional execution** using the condition DSL
6. **File-based workflows** with `execute_workflow_file()`

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

from ffai.Clients.AsyncFFLiteLLMClient import AsyncFFLiteLLMClient  # noqa: E402
from ffai.FFAI import FFAI  # noqa: E402
from ffai.rag._async import run_sync  # noqa: E402
from ffai.workflow import load_workflow  # noqa: E402

async_client = AsyncFFLiteLLMClient(
    model_string="mistral/mistral-small-2503",
    temperature=0,
    max_tokens=100,
    system_instructions="Be concise. Answer in one short sentence.",
)

ffai = FFAI(async_client)

print(f"FFAI initialized with async client: {async_client.model}")
""")

md("""\
<div class="page-break"></div>

---

## Step 1: Load and Validate a Workflow

Workflows are defined in YAML. You can validate them before execution to catch
errors like duplicate names, bad history references, or missing tool definitions.
""")

code("""\
yaml_text = \"\"\"
workflow:
  name: research_pipeline
  description: "A simple research pipeline"

  prompts:
    - name: topic
      prompt: "Name an interesting scientific discovery. Be brief."
    - name: explain
      prompt: "Explain {{topic.response}} in simple terms. Two sentences max."
      history: [topic]
    - name: opinion
      prompt: "What is the practical impact of {{topic.response}}? One sentence."
      history: [topic]
\"\"\"

spec = load_workflow(yaml_text)

print(f"Workflow: {spec.name}")
print(f"Description: {spec.description}")
print(f"Steps: {len(spec.prompts)}")
for step in spec.prompts:
    deps = step.history or []
    print(f"  {step.name}: {len(deps)} dependencies")
""")

code("""\
errors, warnings = ffai.validate_workflow(yaml_text)
print(f"Validation errors: {len(errors)}")
print(f"Validation warnings: {len(warnings)}")
if warnings:
    for w in warnings:
        print(f"  Warning: {w}")
if errors:
    for e in errors:
        print(f"  Error: {e}")
""")

md("""\
No errors — this workflow is valid and ready to execute.

<div class="page-break"></div>

---

## Step 2: Execute a Single-Step Workflow

The simplest workflow has one prompt. Let's start there.
""")

code("""\
result = run_sync(ffai.execute_workflow(\"\"\"
workflow:
  name: hello
  prompts:
    - name: greet
      prompt: "What is 2+2? Answer with just the number."
\"\"\"))

greet = result.results['greet']
print(f"Workflow: {result.spec_name}")
print(f"Success: {result.success_count}, Failed: {result.failed_count}, Skipped: {result.skipped_count}")
print(f"Response: {greet.response}")
print(f"Tokens: {greet.usage.total_tokens}")
print(f"Duration: {greet.duration_ms:.0f}ms")
""")

md("""\
<div class="page-break"></div>

---

## Step 3: Sequential Chain with Interpolation

Use `{{name.response}}` to reference a prior step's output. FFAI resolves
these before sending the prompt to the LLM.

The dependency graph is inferred from `history` lists — steps only run after
their dependencies complete.
""")

code("""\
result = run_sync(ffai.execute_workflow(\"\"\"
workflow:
  name: chain
  prompts:
    - name: language
      prompt: "Name one programming language. Just the name."
    - name: creator
      prompt: "Who created {{language.response}}? Just the name."
      history: [language]
    - name: year
      prompt: "In what year was {{language.response}} first released?"
      history: [language]
\"\"\"))

print(f"Success: {result.success_count}/3")
print()
for name in ["language", "creator", "year"]:
    r = result.results[name]
    print(f"{name}: {r.response.strip()}")
    print(f"  status={r.status}, tokens={r.usage.total_tokens}, cost=${r.cost_usd:.6f}")
    print()
""")

md("""\
Each step receives the prior step's response via interpolation, and the
executor tracks usage and cost per step.

<div class="page-break"></div>

---

## Step 4: Variable Substitution

Use `{variable}` placeholders (single braces) for runtime values. These are
distinct from `{{name.response}}` interpolation (double braces).

Pass variables as a dict to `execute_workflow()`.
""")

code("""\
result = run_sync(ffai.execute_workflow(\"\"\"
workflow:
  name: topic_exploration
  prompts:
    - name: explain
      prompt: "Explain {topic} in one sentence."
    - name: analogy
      prompt: "Give an analogy for {topic}. One sentence."
\"\"\", variables={"topic": "quantum entanglement"}))

print("Topic: quantum entanglement")
print()
explain = result.results['explain'].response.strip()
analogy = result.results['analogy'].response.strip()
print(f"Explanation: {explain}")
print(f"Analogy: {analogy}")
""")

md("""\
Variables are substituted before the DAG is built. You can combine both
mechanisms: `{variable}` for user input and `{{name.response}}` for step chaining.

<div class="page-break"></div>

---

## Step 5: Conditional Execution

Use the `condition` field with the condition DSL to skip steps based on prior
results. The DSL supports `==`, `!=`, `and`, `or`, `not`, `len()`, and more.
""")

code("""\
result = run_sync(ffai.execute_workflow(\"\"\"
workflow:
  name: conditional
  prompts:
    - name: check
      prompt: "Say the word: yes"
    - name: should_run
      prompt: "What day comes after Monday? One word."
      condition: '{{check.status}} == "success"'
    - name: should_skip
      prompt: "This should not execute."
      condition: '{{check.response}} == "IMPOSSIBLE_VALUE_XYZ"'
\"\"\"))

print(f"Success: {result.success_count}, Skipped: {result.skipped_count}")
print()
for name in ["check", "should_run", "should_skip"]:
    r = result.results[name]
    print(f"{name}: status={r.status}")
    if r.response:
        print(f"  response: {r.response.strip()}")
    if r.condition_trace:
        print(f"  trace: {r.condition_trace}")
    print()
""")

md("""\
The condition `{{check.status}} == "success"` evaluated true, so `should_run`
executed. The condition `{{check.response}} == "IMPOSSIBLE_VALUE_XYZ"` evaluated
false, so `should_skip` was skipped.

<div class="page-break"></div>

---

## Step 6: File-Based Workflows

For real projects, store workflows in `.yaml` files. Use
`execute_workflow_file()` to load and execute in one call.
""")

code("""\
import tempfile

wf_dir = Path(tempfile.mkdtemp())
wf_file = wf_dir / "pipeline.yaml"
wf_file.write_text(\"\"\"
workflow:
  name: file_pipeline
  defaults:
    max_concurrency: 5
  prompts:
    - name: question
      prompt: "What is the speed of light? Answer with just the number and units."
    - name: context
      prompt: "Why is {{question.response}} important in physics? One sentence."
      history: [question]
\"\"\")

print(f"Workflow file: {wf_file}")
print()

result = run_sync(ffai.execute_workflow_file(str(wf_file)))
print(f"Workflow: {result.spec_name}")
print(f"Success: {result.success_count}/{len(result.results)}")
print()
for name in ["question", "context"]:
    print(f"{name}: {result.results[name].response.strip()}")
""")

md("""\
<div class="page-break"></div>

---

## Summary

YAML workflows in FFAI provide:

- **Declarative pipelines**: Define prompts, dependencies, and conditions in YAML
- **`{{name.response}}` interpolation**: Chain steps by referencing prior outputs
- **`{variable}` substitution**: Inject runtime values without editing the YAML
- **Condition DSL**: Skip or execute steps based on prior results
- **Validation**: Catch errors before making API calls
- **Usage tracking**: Per-step token counts, costs, and timing

See the **Multi-Client Workflows** notebook for per-step client routing.
""")

with open("examples/workflow_basics/workflow_basics.ipynb", "w") as f:
    nbf.write(nb, f)

print("Created workflow_basics.ipynb")
