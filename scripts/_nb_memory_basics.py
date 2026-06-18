"""Generator for examples/memory_basics/memory_basics.ipynb.

Demonstrates the Memory class directly with a local fastembed backend.
No API key required — embeddings come from local/all-MiniLM-L6-v2.

Run:
    python scripts/_nb_memory_basics.py
    python .opencode/skills/jupyter-notebook/nb_execute.py examples/memory_basics/memory_basics.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = []


def code(s):
    nb.cells.append(nbf.v4.new_code_cell(s))


def md(s):
    nb.cells.append(nbf.v4.new_markdown_cell(s))


md("""\
# Memory Vector Recall: Basic Semantic Search

This notebook demonstrates FFAI's **memory vector recall** feature, which
embeds completed conversation turns (Q+A pairs) and lets you retrieve them
by semantic similarity rather than exact keyword match.

We use the `Memory` class directly with a **local embedding model**
(`local/all-MiniLM-L6-v2` via `fastembed`). No API key required —
everything runs offline.

**What you'll see:**

1. Construct a `Memory` instance with a local embedding backend
2. Index sample Q+A pairs about Python topics
3. Search by *concept* ("database issue") and retrieve semantically related turns
4. Filter by similarity threshold
5. Inspect `TurnHit` result fields
6. Use the async API

<div class="page-break"></div>

---
""")

code("""\
import sys
import time
from pathlib import Path

_cwd = Path().resolve()
_project_root = _cwd
for _p in [_cwd, *list(_cwd.parents)]:
    if (_p / 'pyproject.toml').is_file():
        _project_root = _p
        break

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from ffai.core.embeddings import Embeddings  # noqa: E402
from ffai.core.memory import Memory, TurnHit  # noqa: E402

print("Setup complete")
print(f"Project root: {_project_root}")
""")

md("""\
<div class="page-break"></div>

---

## Step 1: Construct a Memory instance with a local embedding backend

`Embeddings("local/all-MiniLM-L6-v2")` loads a 384-dim sentence-transformer
model via `fastembed` (installed with `pip install ffai[memory]` or
`pip install ffai[rag]`). The first call downloads the model (~80 MB);
subsequent calls use the cached copy.
""")

code("""\
embeddings = Embeddings(model="local/all-MiniLM-L6-v2")
memory = Memory(embeddings)

print(f"Embedding backend: {embeddings.model}")
print(f"Is local:          {embeddings.is_local}")
print(f"Memory store:      {type(memory.store).__name__}")
print(f"Initial count:     {memory.count()}")
""")

md("""\
<div class="page-break"></div>

---

## Step 2: Index sample Q+A turns

Each turn is a `(text, turn, metadata)` triple. The **text** is what gets
embedded; the **turn** dict is the structured record stored alongside
(mirrors `PermanentHistory`'s shape); the **metadata** is caller-supplied
and carried through to search results.

We index five Q+A pairs covering different Python topics so we can show
semantic similarity retrieving related turns even when query wording
differs from the indexed text.
""")

code("""\
sample_turns = [
    {
        "text": (
            "How do I fix the Postgres migration?\\n"
            "The schema diff shows a missing index on users.email. "
            "Run CREATE INDEX CONCURRENTLY to add it without locking the table."
        ),
        "prompt_name": "postgres_debug",
    },
    {
        "text": (
            "Update the user table schema\\n"
            "ALTER TABLE users ADD COLUMN verified_at TIMESTAMP DEFAULT NOW();"
        ),
        "prompt_name": "schema_update",
    },
    {
        "text": (
            "What is Python's GIL?\\n"
            "The Global Interpreter Lock prevents multiple native threads "
            "from executing Python bytecodes at once."
        ),
        "prompt_name": "gil_concept",
    },
    {
        "text": (
            "Write a haiku about autumn\\n"
            "Red leaves whisper down / Branches bare against the sky / "
            "Cool wind carries them"
        ),
        "prompt_name": "haiku",
    },
    {
        "text": (
            "How do I deploy to AWS?\\n"
            "Use the AWS CDK to define infrastructure as TypeScript, "
            "then cdk deploy after running cdk synth."
        ),
        "prompt_name": "aws_deploy",
    },
]

for turn_data in sample_turns:
    memory.index_turn_text(
        text=turn_data["text"],
        turn={
            "role": "assistant",
            "content": [{"type": "text", "text": turn_data["text"]}],
            "timestamp": time.time(),
        },
        metadata={"prompt_name": turn_data["prompt_name"]},
    )

print(f"Indexed {memory.count()} turns")
""")

md("""\
<div class="page-break"></div>

---

## Step 3: Semantic search — "the database issue from earlier"

Notice the query wording: **"the database issue from earlier"**. None of
the indexed turns contain the words "database" or "issue" — but the
Postgres migration and schema update turns are semantically related.

Linear substring search would miss both. Vector recall catches them via
cosine similarity between the query embedding and each turn's embedding.
""")

code("""\
hits = memory.search("the database issue from earlier", top_k=3)

print(f"Query: 'the database issue from earlier'")
print(f"Top {len(hits)} hits:")
print()
for i, hit in enumerate(hits, 1):
    print(f"{i}. [{hit.score:.3f}] prompt_name={hit.metadata['prompt_name']}")
    print(f"   {hit.text[:90]}...")
    print()
""")

md("""\
<div class="page-break"></div>

---

## Step 4: Threshold filtering

The `threshold` argument excludes hits below a minimum cosine similarity.
Useful when you only want high-confidence matches and would rather get
`[]` than a noisy long tail.

Compare the same query with two thresholds:
""")

code("""\
query = "how do I ship code to production"

print(f"Query: '{query}'")
print()
for threshold in [None, 0.3, 0.5]:
    hits = memory.search(query, top_k=5, threshold=threshold)
    label = "no floor" if threshold is None else f">= {threshold}"
    print(f"Threshold {label:>10}: {len(hits)} hits")
    for hit in hits:
        print(f"  [{hit.score:.3f}] {hit.metadata['prompt_name']}")
    print()
""")

md("""\
<div class="page-break"></div>

---

## Step 5: Inspecting TurnHit fields

Every search hit is a `TurnHit` dataclass (frozen). Five fields:
""")

code("""\
hits = memory.search("Python threading", top_k=1)
hit = hits[0]

print(f"TurnHit fields:")
print(f"  score:       {hit.score:.3f}  (cosine similarity, [-1.0, 1.0])")
print(f"  turn_index:  {hit.turn_index}  (position in the store)")
print(f"  text:        {hit.text[:60]}...")
print(f"  metadata:    {hit.metadata}")
print(f"  turn.role:   {hit.turn['role']}")
print(f"  turn.content[0].type: {hit.turn['content'][0]['type']}")
print()
print(f"TurnHit is frozen — assignment raises FrozenInstanceError:")
try:
    hit.score = 0.99
except Exception as exc:
    print(f"  {type(exc).__name__}: {exc}")
""")

md("""\
<div class="page-break"></div>

---

## Step 6: Async API

`Memory` provides async variants of `index_turn*` and `search`. These
use `Embeddings.aembed()` under the hood, which is the right choice
inside async DAG execution or any other `async` context.

We use `ffai.core._async.run_sync` to drive the coroutine from this
synchronous notebook cell — the same helper FFAI uses internally.
""")

code("""\
import asyncio
from ffai.core._async import run_sync

async def demo_async():
    await memory.aindex_turn_text(
        text=(
            "What is asyncio?\\n"
            "asyncio is a Python library for writing concurrent code "
            "using async/await syntax and an event loop."
        ),
        turn={
            "role": "assistant",
            "content": [{"type": "text", "text": "What is asyncio?..."}],
            "timestamp": time.time(),
        },
        metadata={"prompt_name": "asyncio_concept"},
    )
    hits = await memory.asearch("concurrent programming", top_k=2)
    return hits

new_count_before = memory.count()
hits = run_sync(demo_async())
new_count_after = memory.count()

print(f"Count before async index: {new_count_before}")
print(f"Count after async index:  {new_count_after}")
print()
print(f"Async search returned {len(hits)} hits for 'concurrent programming':")
for hit in hits:
    print(f"  [{hit.score:.3f}] {hit.metadata['prompt_name']}")
""")

md("""\
<div class="page-break"></div>

---

## Summary

- `Memory` embeds Q+A pairs and ranks them by cosine similarity to a query.
- **Local embeddings work offline** via `fastembed` — `pip install ffai[memory]`.
- **Semantic recall** finds related turns even when query wording differs
  ("database issue" → Postgres migration + schema update).
- **`threshold`** filters out low-confidence matches.
- **`TurnHit`** carries `score`, `turn`, `turn_index`, `text`, and `metadata`
  — all the context needed for downstream use.
- **Async variants** (`aindex_turn_text`, `asearch`) work inside async contexts.

### What's next

- `memory_persistence.ipynb` — survive process restarts via Parquet
- `memory_ffai_integration.ipynb` — automatic embedding at `record()` time
""")

with open("examples/memory_basics/memory_basics.ipynb", "w") as f:
    nbf.write(nb, f)

print("Created examples/memory_basics/memory_basics.ipynb")
