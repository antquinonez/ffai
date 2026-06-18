"""Generator for examples/memory_persistence/memory_persistence.ipynb.

Demonstrates cross-session memory persistence via Parquet. Runs entirely
on local embeddings — no API key required.

Run:
    python scripts/_nb_memory_persistence.py
    python .opencode/skills/jupyter-notebook/nb_execute.py examples/memory_persistence/memory_persistence.ipynb
"""

import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.cells = []


def code(s):
    nb.cells.append(nbf.v4.new_code_cell(s))


def md(s):
    nb.cells.append(nbf.v4.new_markdown_cell(s))


md("""\
# Memory Persistence: Cross-Session Recall via Parquet

By default, `Memory` is **ephemeral** — its in-memory store dies when the
process exits. Setting `persist=True` makes it survive restarts: the store
is loaded from a Parquet file at startup and written after each successful
embed.

This notebook demonstrates:

1. Build a memory, index turns, **persist** to Parquet
2. Inspect the on-disk schema
3. **Load** the store in a fresh process (simulated restart)
4. Verify the loaded store supports search
5. Show the FFAI configuration that wires this up automatically

Runs offline using `local/all-MiniLM-L6-v2` via `fastembed`.

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

_cwd = Path().resolve()
_project_root = _cwd
for _p in [_cwd, *list(_cwd.parents)]:
    if (_p / 'pyproject.toml').is_file():
        _project_root = _p
        break

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import polars as pl  # noqa: E402

from ffai.core.embeddings import Embeddings  # noqa: E402
from ffai.core.memory import Memory, TurnVectorStore, load_store, persist_store  # noqa: E402

# Use a temp directory so the notebook is idempotent across runs
_persist_dir = Path(tempfile.mkdtemp(prefix="ffai_memory_demo_"))
_persist_path = _persist_dir / "turns.parquet"

print("Setup complete")
print(f"Persist path: {_persist_path}")
""")

md("""\
<div class="page-break"></div>

---

## Step 1: Build memory and index turns

Five Q+A pairs about Python tooling. We use `Memory.index_turn_text`
directly — in the FFAI integration flow (next notebook) this happens
automatically inside `HistoryRecorder.record()`.
""")

code("""\
embeddings = Embeddings(model="local/all-MiniLM-L6-v2")
memory = Memory(embeddings)

sample_turns = [
    ("What is pytest?", "pytest is a testing framework that uses assert statements and fixtures."),
    ("How do I configure ruff?", "Configure ruff in pyproject.toml under [tool.ruff] with select/ignore rules."),
    ("What does mypy do?", "mypy performs static type checking on Python code using type annotations."),
    ("How to use pre-commit?", "Define hooks in .pre-commit-config.yaml and run pre-commit install."),
    ("What is uv?", "uv is a fast Python package installer and resolver written in Rust."),
]

for question, answer in sample_turns:
    text = f"{question}\\n{answer}"
    memory.index_turn_text(
        text=text,
        turn={
            "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "timestamp": time.time(),
        },
        metadata={"question": question},
    )

print(f"Indexed {memory.count()} turns")
""")

md("""\
<div class="page-break"></div>

---

## Step 2: Persist the store to Parquet

`persist_store(store, path)` writes one row per turn with five columns.
The file is overwritten on each call — Parquet writes are atomic from
the reader's perspective as long as the process doesn't die mid-write.
""")

code("""\
persist_store(memory.store, path=str(_persist_path))

file_size = _persist_path.stat().st_size
print(f"Wrote: {_persist_path}")
print(f"Size:  {file_size} bytes ({file_size / 1024:.1f} KB)")
""")

md("""\
<div class="page-break"></div>

---

## Step 3: Inspect the on-disk schema

Five columns. `text` and the JSON-encoded `turn` and `metadata` are
strings; `embedding` is `List(Float64)`; `_schema_version` is `Int8`
(reserved for future format changes).
""")

code("""\
df = pl.read_parquet(_persist_path)

print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
print()
print(f"Schema:")
for col, dtype in df.schema.items():
    print(f"  {col:<20} {dtype}")
print()
print(f"First row (truncated):")
first = df.row(0, named=True)
print(f"  text:               {first['text'][:60]}...")
print(f"  embedding dim:      {len(first['embedding'])}")
print(f"  turn (parsed JSON): {first['turn'][:60]}...")
print(f"  metadata:           {first['metadata']}")
print(f"  schema_version:     {first['_schema_version']}")
""")

md("""\
<div class="page-break"></div>

---

## Step 4: Load the store in a fresh process

This simulates restarting the Python interpreter. We construct a brand
new `Memory` instance and swap in the loaded store via the settable
`memory.store` attribute.

Note that we **don't need to re-embed** anything — the vectors come
straight from disk.
""")

code("""\
# Simulate a fresh process: discard the in-memory store, then reload
loaded_store = load_store(path=str(_persist_path))

# Construct a fresh Memory and swap in the loaded store
fresh_memory = Memory(embeddings, store=loaded_store)

print(f"Loaded store count: {fresh_memory.count()}")
print(f"Original count was: {memory.count()}")
print(f"Round-trip equal:   {fresh_memory.count() == memory.count()}")
""")

md("""\
<div class="page-break"></div>

---

## Step 5: The loaded store supports search

Search works immediately after load — no warmup, no re-indexing required.
""")

code("""\
hits = fresh_memory.search("Python testing tools", top_k=3)

print(f"Query: 'Python testing tools'")
print(f"Top {len(hits)} hits from loaded store:")
print()
for i, hit in enumerate(hits, 1):
    print(f"{i}. [{hit.score:.3f}] {hit.metadata['question']}")
    print(f"   {hit.text[:80]}...")
    print()
""")

md("""\
<div class="page-break"></div>

---

## Step 6: FFAI wires this up automatically

When `config.memory.persist: true`, `FFAI.__init__()`:

1. Calls `load_store()` at startup if the file exists
2. Calls `persist_store()` after each successful embed

You only need to set two knobs in `config/main.yaml`:

```yaml
memory:
  enabled: true
  persist: true
  persist_dir: "./ffai_data/memory"
  collection_name: "ffai_turns"      # produces ffai_turns.parquet
```

Or via constructor:

```python
ffai = FFAI(
    client,
    memory_enabled=True,
    memory_persist=True,
)
```

On restart, prior turns are searchable immediately.
""")

code("""\
# Cleanup the demo directory (uncomment to keep the file for inspection)
shutil.rmtree(_persist_dir)
print(f"Cleaned up: {_persist_dir}")
""")

md("""\
<div class="page-break"></div>

---

## Summary

- **`persist_store(store, path)`** writes a Parquet file with one row per turn.
- **`load_store(path)`** returns a fresh `TurnVectorStore`, ready to search.
- **`memory.store = loaded`** swaps the store in-place — no re-embedding needed.
- **FFAI's `memory_persist=True`** automates both directions on every
  `record()` call.

### Schema versioning

The `_schema_version` column lets us migrate the format in future releases.
Loaders reject unknown versions with a clear error rather than silently
corrupting data.

### Limitations

- Writes are **overwrite-on-persist**, not append. Each write rewrites the
  full file. Fine for Tier 1 scale (<100K turns); larger workloads should
  use Chroma or Qdrant backends (future work).
- Writes are **not atomic** against process death mid-write. If the process
  crashes during a write, the file may be corrupted. Acceptable for
  short-running workflows; long-running services should snapshot-then-rename.
""")

with open("examples/memory_persistence/memory_persistence.ipynb", "w") as f:
    nbf.write(nb, f)

print("Created examples/memory_persistence/memory_persistence.ipynb")
