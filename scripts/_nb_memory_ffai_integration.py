"""Generator for examples/memory_ffai_integration/memory_ffai_integration.ipynb.

Demonstrates the full FFAI integration: generate_response() eagerly embeds
Q+A pairs, and ffai.history.search() retrieves them semantically.

Uses a mock client with canned responses so the notebook runs without any
LLM API key. Embeddings come from local/all-MiniLM-L6-v2 via fastembed.

Run:
    python scripts/_nb_memory_ffai_integration.py
    python .opencode/skills/jupyter-notebook/nb_execute.py examples/memory_ffai_integration/memory_ffai_integration.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = []


def code(s):
    nb.cells.append(nbf.v4.new_code_cell(s))


def md(s):
    nb.cells.append(nbf.v4.new_markdown_cell(s))


md("""\
# FFAI Integration: Eager Embedding at record() Time

This notebook shows the memory feature working end-to-end inside the FFAI
wrapper. Every call to `generate_response()` embeds the Q+A pair on a
fire-and-forget background thread; `ffai.history.search()` retrieves them
by semantic similarity.

**What you'll see:**

1. Construct `FFAI` with `memory_enabled=True` (opt-in)
2. Drive `generate_response()` with a **mock client** — no API key needed
3. Watch the background embed thread index each turn
4. Call `ffai.history.search()` and inspect `TurnHit` results
5. Observe metadata propagation (`prompt_name` flows into `TurnHit.metadata`)
6. Confirm the disabled state (`memory_enabled=False`) writes nothing
7. See `memory_persist=True` write a Parquet file automatically

<div class="page-break"></div>

---
""")

code("""\
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

_cwd = Path().resolve()
_project_root = _cwd
for _p in [_cwd, *list(_cwd.parents)]:
    if (_p / 'pyproject.toml').is_file():
        _project_root = _p
        break

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ffai import FFAI  # noqa: E402

# Temp dir for the persistence demo at the end
_demo_dir = Path(tempfile.mkdtemp(prefix="ffai_integration_"))

print("Setup complete")
print(f"Demo dir: {_demo_dir}")
""")

md("""\
<div class="page-break"></div>

---

## Step 1: Build a mock client and FFAI with memory enabled

The mock client returns canned responses keyed by the prompt text. This
lets the notebook run **fully offline** — no LLM API key required.
Embeddings still come from `local/all-MiniLM-L6-v2` via `fastembed`.

In a real application, you'd substitute a `FFLiteLLMClient`:

```python
from ffai.Clients import FFLiteLLMClient

client = FFLiteLLMClient(
    model_string="mistral/mistral-small-latest",
    api_key=os.environ["MISTRAL_API_KEY"],
)
ffai = FFAI(client, memory_enabled=True)
```
""")

code("""\
# Canned responses keyed by a substring of the prompt
CANNED_RESPONSES = {
    "Postgres": (
        "The migration issue is the missing index on users.email. "
        "Run CREATE INDEX CONCURRENTLY to add it without locking the table."
    ),
    "schema": (
        "ALTER TABLE users ADD COLUMN verified_at TIMESTAMP DEFAULT NOW();"
    ),
    "GIL": (
        "The Global Interpreter Lock prevents multiple native threads "
        "from executing Python bytecodes at once."
    ),
    "deploy": (
        "Use the AWS CDK to define infrastructure as TypeScript, then "
        "cdk deploy after running cdk synth."
    ),
    "haiku": (
        "Red leaves whisper down / Branches bare against the sky / Cool wind carries them"
    ),
}

def _mock_response(prompt: str) -> str:
    for key, response in CANNED_RESPONSES.items():
        if key.lower() in prompt.lower():
            return response
    return "I don't have a canned response for that."

def make_mock_client() -> Any:
    client = MagicMock()
    client.model = "mock-model"
    client.get_conversation_history.return_value = []
    client.set_conversation_history = MagicMock(return_value=True)
    client.clear_conversation = MagicMock()
    client.last_usage = None
    client.last_cost_usd = 0.0

    def generate_response(prompt: str, **kwargs: Any) -> str:
        return _mock_response(prompt)

    client.generate_response.side_effect = generate_response
    return client

client = make_mock_client()

ffai = FFAI(client, memory_enabled=True)

print(f"FFAI constructed")
print(f"Memory instance:  {ffai.history.memory}")
print(f"Memory store:     {type(ffai.history.memory.store).__name__}")
print(f"Initial count:    {ffai.history.memory.count()}")
""")

md("""\
<div class="page-break"></div>

---

## Step 2: generate_response() eagerly embeds each Q+A pair

Each successful `generate_response()` call submits an embed task to a
background thread. The call returns **immediately** — it does not wait
for the embed to complete. We poll `count()` briefly to confirm the
background thread has caught up.
""")

code("""\
# Drive five Q+A turns through the mock client
prompts = [
    ("Let's debug the Postgres migration", "postgres_debug"),
    ("Now update the user table schema", "schema_update"),
    ("What is Python's GIL?", "gil_concept"),
    ("How do I deploy to AWS?", "aws_deploy"),
    ("Write a haiku about autumn", "haiku"),
]

for prompt, prompt_name in prompts:
    result = ffai.workflow.generate_response(prompt=prompt, prompt_name=prompt_name)
    print(f"  {prompt_name:<18} -> {str(result.response)[:60]}...")

# Wait for the fire-and-forget embeds to land
mem = ffai.history.memory
for _ in range(100):
    if mem.count() == len(prompts):
        break
    time.sleep(0.05)

print()
print(f"Indexed {mem.count()} turns (expected {len(prompts)})")
""")

md("""\
<div class="page-break"></div>

---

## Step 3: Semantic search via ffai.history.search()

`ffai.history.search()` is the public entry point. It returns
`list[TurnHit]` ranked by cosine similarity. When memory is disabled, it
returns `[]` (no exception).
""")

code("""\
hits = ffai.history.search("the database issue from earlier", top_k=3)

print(f"Query: 'the database issue from earlier'")
print(f"Top {len(hits)} hits:")
print()
for i, hit in enumerate(hits, 1):
    print(f"{i}. [{hit.score:.3f}] prompt_name={hit.metadata.get('prompt_name')}")
    print(f"   {hit.text[:90]}...")
    print()
""")

md("""\
<div class="page-break"></div>

---

## Step 4: Metadata propagation — prompt_name flows into TurnHit

Each `generate_response(prompt_name="x")` call stores
`{"prompt_name": "x"}` on the indexed turn's metadata. That metadata
travels through to search results untouched.
""")

code("""\
hits = ffai.history.search("Python threading", top_k=2)

print(f"Query: 'Python threading'")
print()
for hit in hits:
    print(f"  score:    {hit.score:.3f}")
    print(f"  metadata: {hit.metadata}")
    print(f"  text:     {hit.text[:80]}...")
    print()
""")

md("""\
<div class="page-break"></div>

---

## Step 5: Threshold filtering — confidence-gated recall

Pass `threshold=N` to drop hits below a cosine similarity floor. Useful
when you'd rather get `[]` than a noisy long tail of marginally related
turns.
""")

code("""\
query = "ship code to production"

print(f"Query: '{query}'")
print()
for threshold in [None, 0.3, 0.5]:
    hits = ffai.history.search(query, top_k=5, threshold=threshold)
    label = "no floor" if threshold is None else f">= {threshold}"
    print(f"Threshold {label:>10}: {len(hits)} hits")
    for hit in hits:
        print(f"  [{hit.score:.3f}] {hit.metadata.get('prompt_name')}")
    print()
""")

md("""\
<div class="page-break"></div>

---

## Step 6: Disabled state — memory_enabled=False writes nothing

When memory is disabled (the default), `ffai.history.memory is None`,
`ffai.history.search()` returns `[]`, and `generate_response()` makes no
embedding calls at all. No background thread is started.
""")

code("""\
client2 = make_mock_client()
ffai_disabled = FFAI(client2)  # memory_enabled defaults to False

print(f"Memory instance: {ffai_disabled.history.memory}")
print(f"Search returns:  {ffai_disabled.history.search('anything')}")

# Run a generate_response and confirm nothing was indexed
ffai_disabled.workflow.generate_response(prompt="hello", prompt_name="greet")
print(f"After generate_response, memory is still: {ffai_disabled.history.memory}")

ffai_disabled.close()
""")

md("""\
<div class="page-break"></div>

---

## Step 7: Persistence — memory_persist=True writes Parquet automatically

With `memory_persist=True`, FFAI loads `<persist_dir>/<collection_name>.parquet`
on construction (if it exists) and writes after each successful embed.
We override the config paths to point at our temp directory.
""")

code("""\
from unittest.mock import patch

# Point FFAI's config at our temp directory for the persistence demo.
# Set memory.embedding_model = None so the resolution ladder falls
# through to local backend detection (fastembed/all-MiniLM-L6-v2).
mock_config = MagicMock()
mock_config.paths.ffai_data = str(_demo_dir)
mock_config.rag.enabled = False
mock_config.memory.persist_dir = str(_demo_dir)
mock_config.memory.collection_name = "demo_turns"
mock_config.memory.embedding_model = None

client3 = make_mock_client()

with patch("ffai.FFAI.get_config", return_value=mock_config):
    ffai_persist = FFAI(
        client3,
        memory_enabled=True,
        memory_persist=True,
    )

print(f"Memory store count at startup: {ffai_persist.history.memory.count()}")

# Generate one turn and let the embed + persist land
ffai_persist.workflow.generate_response(
    prompt="What is Python's GIL?",
    prompt_name="gil_persisted",
)

mem = ffai_persist.history.memory
for _ in range(100):
    if mem.count() == 1:
        break
    time.sleep(0.05)

print(f"Memory store count after one record(): {mem.count()}")

# Allow the persist call (which runs after the embed) to complete
time.sleep(0.3)

expected_parquet = _demo_dir / "demo_turns.parquet"
print()
print(f"Expected Parquet file: {expected_parquet}")
print(f"Exists:                {expected_parquet.exists()}")
print(f"Size:                  {expected_parquet.stat().st_size} bytes")

ffai_persist.close()
""")

md("""\
<div class="page-break"></div>

---

## Step 8: Cross-process recall — load the Parquet in a fresh FFAI

The Parquet file written above is readable by any new FFAI instance
configured with the same `persist_dir` and `collection_name`. Prior turns
are searchable immediately on startup — no re-embedding required.
""")

code("""\
client4 = make_mock_client()

with patch("ffai.FFAI.get_config", return_value=mock_config):
    ffai_restarted = FFAI(
        client4,
        memory_enabled=True,
        memory_persist=True,
    )

mem = ffai_restarted.history.memory
print(f"Loaded turns on restart: {mem.count()}")

hits = ffai_restarted.history.search("Python threading", top_k=2)
print()
print(f"Search 'Python threading' on restarted instance:")
for hit in hits:
    print(f"  [{hit.score:.3f}] {hit.metadata}")
    print(f"  {hit.text[:80]}...")

ffai_restarted.close()
""")

md("""\
<div class="page-break"></div>

---

## Step 9: Clean shutdown

`ffai.close()` shuts down the background embed thread pool. Safe to call
multiple times. Recommended at process teardown in long-running services
to avoid leaking daemon threads.
""")

code("""\
ffai.close()
print(f"Embed pool after close: {ffai._recorder._embed_pool}")

# Cleanup demo directory
shutil.rmtree(_demo_dir)
print(f"Cleaned up: {_demo_dir}")
""")

md("""\
<div class="page-break"></div>

---

## Summary

- **`memory_enabled=True`** is the single opt-in flag. Everything else
  (background thread, embedding, indexing) happens automatically inside
  `generate_response()`.
- **`ffai.history.search(query)`** returns `list[TurnHit]`. Returns `[]`
  when memory is disabled — no exception.
- **`prompt_name`** flows through to `TurnHit.metadata` automatically.
  Tier 2 (auto context injection) will add `user_id` / `session_id` to
  the same dict.
- **`memory_persist=True`** wires up Parquet load-on-startup and
  write-after-embed. Restarts see prior turns immediately.
- **`ffai.close()`** shuts down the background thread pool cleanly.

### Cost model

Each `generate_response()` call adds one embedding call (network or local
CPU). The embed runs on a fire-and-forget background thread, so it does
**not** block the main call. Failures are logged at `WARNING` and dropped
— they never propagate to `record()` callers.

### What's next

- `_ideas/tier2-auto-context-injection.md` — design sketch for automatic
  retrieval-augmented context on every prompt (not yet implemented).
- Long-term: mem0-style fact extraction (Tier 3) for true user/agent
  memory across sessions.
""")

with open("examples/memory_ffai_integration/memory_ffai_integration.ipynb", "w") as f:
    nbf.write(nb, f)

print("Created examples/memory_ffai_integration/memory_ffai_integration.ipynb")
