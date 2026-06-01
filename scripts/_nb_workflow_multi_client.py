import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = []

def code(s):
    nb.cells.append(nbf.v4.new_code_cell(s))

def md(s):
    nb.cells.append(nbf.v4.new_markdown_cell(s))

md("""\
# Multi-Client YAML Workflows

This notebook demonstrates per-step client routing in YAML workflows.
Each prompt step can specify its own LLM client — different models, different
providers — all within a single workflow.

Topics covered:

1. **Inline client definitions** in the `clients:` block
2. **Per-step client assignment** via the `client:` field
3. **Cross-client interpolation** — chaining outputs between providers
4. **Workflow defaults** — setting a default client for all steps

This example uses **Mistral Small** (via LiteLLM) as the default client and
**OpenAI GPT-4o-mini** as an override for specific steps.

<div class="page-break"></div>

---
""")

code("""\
import os
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

assert os.getenv('MISTRAL_API_KEY'), 'MISTRAL_API_KEY required'
assert os.getenv('OPENAI_API_KEY'), 'OPENAI_API_KEY required'

async_client = AsyncFFLiteLLMClient(
    model_string="mistral/mistral-small-2503",
    temperature=0,
    max_tokens=100,
    system_instructions="Be concise. Answer in one short sentence.",
)

ffai = FFAI(async_client)

print(f"Default client: {async_client.model}")
print(f"OpenAI key available: {bool(os.getenv('OPENAI_API_KEY'))}")
""")

md("""\
<div class="page-break"></div>

---

## Step 1: Two Clients, One Workflow

The `clients:` block defines named client configurations inline. Each prompt
step can then reference a client by name using the `client:` field.

Steps without a `client:` field use the workflow's `defaults.client`.
""")

code("""\
result = run_sync(ffai.execute_workflow(\"\"\"
workflow:
  name: two_providers
  defaults:
    client: litellm-mistral-small
  clients:
    openai_reviewer:
      type: litellm
      provider_prefix: "openai/"
      model: "gpt-4o-mini"
      api_key_env: "OPENAI_API_KEY"
  prompts:
    - name: mistral_step
      prompt: "What is 3+4? Answer with just the number."
    - name: openai_step
      prompt: "What is 5+6? Answer with just the number."
      client: openai_reviewer
\"\"\"))

print(f"Success: {result.success_count}/2")
print()
mistral_r = result.results['mistral_step'].response.strip()
openai_r = result.results['openai_step'].response.strip()
print(f"Mistral result: {mistral_r}")
print(f"OpenAI result:  {openai_r}")
""")

md("""\
Both steps succeeded. `mistral_step` used the default Mistral client,
`openai_step` used the OpenAI client defined in the `clients:` block.

<div class="page-break"></div>

---

## Step 2: Cross-Client Interpolation

Steps running on different clients can still reference each other's outputs
using `{{name.response}}`. The interpolation happens before the prompt is
sent to the target client.
""")

code("""\
result = run_sync(ffai.execute_workflow(\"\"\"
workflow:
  name: cross_client
  clients:
    openai_reviewer:
      type: litellm
      provider_prefix: "openai/"
      model: "gpt-4o-mini"
      api_key_env: "OPENAI_API_KEY"
  prompts:
    - name: mistral_fact
      prompt: "Name the capital of France. One word only."
    - name: openai_elaborate
      prompt: "In one sentence, describe {{mistral_fact.response}}."
      history: [mistral_fact]
      client: openai_reviewer
\"\"\"))

fact = result.results['mistral_fact'].response.strip().lower()
elaboration = result.results['openai_elaborate'].response.strip()

print(f"Mistral says:    {fact}")
print(f"OpenAI responds: {elaboration}")
assert 'paris' in fact, f'Expected paris, got {fact}'
""")

md("""\
The Mistral step answered "Paris", and the OpenAI step received that
answer via `{{mistral_fact.response}}` and elaborated on it.

<div class="page-break"></div>

---

## Step 3: Three Steps, Two Providers

A realistic workflow mixing providers: generate with one, evaluate with another,
then synthesize with the first.
""")

code("""\
result = run_sync(ffai.execute_workflow(\"\"\"
workflow:
  name: generate_evaluate_synthesize
  clients:
    openai_reviewer:
      type: litellm
      provider_prefix: "openai/"
      model: "gpt-4o-mini"
      api_key_env: "OPENAI_API_KEY"
  prompts:
    - name: generate
      prompt: "List 3 benefits of exercise. Be concise."
    - name: evaluate
      prompt: "Rate these exercise benefits on a scale of 1-10 for importance: {{generate.response}}"
      history: [generate]
      client: openai_reviewer
    - name: synthesize
      prompt: "Create a one-sentence health tip based on: {{generate.response}}"
      history: [generate]
\"\"\"))

print(f"Success: {result.success_count}/3")
print()
for name in ['generate', 'evaluate', 'synthesize']:
    r = result.results[name]
    print('--- ' + name + ' ---')
    print(r.response.strip())
    print(f"  model={r.model}, tokens={r.usage.total_tokens}")
    print()
""")

md("""\
<div class="page-break"></div>

---

## Step 4: Per-Step Parameters

Each step can override `temperature`, `max_tokens`, `system_instructions`,
and `model` independently of its client configuration.
""")

code("""\
result = run_sync(ffai.execute_workflow(\"\"\"
workflow:
  name: per_step_params
  clients:
    openai_reviewer:
      type: litellm
      provider_prefix: "openai/"
      model: "gpt-4o-mini"
      api_key_env: "OPENAI_API_KEY"
  defaults:
    temperature: 0.7
    max_tokens: 100
  prompts:
    - name: creative
      prompt: "Write a haiku about programming."
      temperature: 1.0
    - name: precise
      prompt: "What is the chemical symbol for gold?"
      client: openai_reviewer
      temperature: 0
      max_tokens: 10
\"\"\"))

creative = result.results['creative'].response.strip()
precise = result.results['precise'].response.strip()

print("Creative (Mistral, temp=1.0):")
print(f"  {creative}")
print()
print("Precise (OpenAI, temp=0):")
print(f"  {precise}")
""")

md("""\
<div class="page-break"></div>

---

## Summary

Per-step client routing in YAML workflows:

- **`clients:` block**: Define named client configs inline (type, model, API key)
- **`client:` field**: Assign any named client to a specific step
- **`defaults.client`**: Set a fallback for steps without explicit client
- **Cross-client interpolation**: `{{name.response}}` works across providers
- **Per-step overrides**: `temperature`, `max_tokens`, `model` per step

Client resolution order for a named reference:

1. Workflow's inline `clients:` block
2. `config/clients.yaml` (the project-level client registry)
3. FFAI instance's default client (fallback)
""")

with open("examples/workflow_multi_client/workflow_multi_client.ipynb", "w") as f:
    nbf.write(nb, f)

print("Created workflow_multi_client.ipynb")
