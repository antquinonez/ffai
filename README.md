# FFAI

Declarative multi-provider AI client with named-prompt context assembly, async DAG execution, RAG, agentic tool-call loops, and built-in cost/usage tracking.

**[Documentation](https://ffai.readthedocs.io)** | **[PyPI](https://pypi.org/project/ffai/)** | **[Source](https://github.com/antquinonez/ffai)**

## Installation

```bash
pip install ffai
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add ffai
```

Install directly from GitHub:

```bash
pip install git+https://github.com/antquinonez/ffai.git
```

### Optional extras

| Extra | pip | uv |
|-------|-----|----|
| RAG | `pip install "ffai[rag]"` | `uv add "ffai[rag]"` |
| OpenTelemetry | `pip install "ffai[otel]"` | `uv add "ffai[otel]"` |
| Both | `pip install "ffai[rag,otel]"` | `uv add "ffai[rag,otel]"` |

RAG installs ChromaDB for persistent vector storage. OpenTelemetry installs OTLP span export for tracing.

> **Note:** Quotes are required around `ffai[rag]` in zsh and some other shells, since brackets are special characters. Bash does not require quotes.

### Vector store backends

FFAI supports multiple vector store backends. ChromaDB is included with `ffai[rag]`; others install separately:

| Backend | Install | Mode |
|---------|---------|------|
| **ChromaDB** (default) | Included with `ffai[rag]` | Local files |
| **Qdrant** | `pip install qdrant-client` | Local, in-memory, server, or cloud |
| **pgvector** | `pip install psycopg asyncpg` + PostgreSQL with pgvector | Server (Docker) |
| **SQLite-vss** | `pip install sqlite-vss` | Local files |

## Features

- **Declarative context assembly** — reference earlier responses by name using `{{prompt_name.response}}` interpolation
- **RAG (Retrieval-Augmented Generation)** — chunking, embeddings, vector search, BM25 hybrid, reranking, query expansion, hierarchical indexing, deduplication, contextual embeddings, and `FFAI.query()` for one-shot retrieval-augmented answers
- **DAG execution** — dependency graph with topological-parallel prompt execution and condition-based branching, with graph validation via `validate_graph()`
- **Async support** — async client base, async DAG executor, and async RAG methods (`aquery`, `aindex`, `asearch`)
- **Agentic execution loop** — multi-round tool-call-driven execution with automatic tool dispatch, per-tool timeouts, and error continuation
- **Structured output** — Pydantic model validation with automatic retry on schema failures
- **Response validation** — LLM-as-judge PASS/FAIL validation with automatic re-execution
- **Multi-client support** — 100+ providers via LiteLLM, plus native Mistral SDK; fallback model chains on failure
- **Usage and cost tracking** — automatic per-call token counting, cost estimation, and wall-clock timing across providers
- **History management** — four history views with Polars DataFrame export, full-text search, and Parquet persistence
- **OpenTelemetry tracing** — optional span emission with model, token, and cost attributes
- **Condition DSL** — AST-safe expression evaluation with JSON navigation, string ops, regex matching, and math
- **Runtime configuration** — YAML-based config with `pyproject.toml` extras for RAG, or zero-config `RAG.from_config()`

## Quick Start

```python
from ffai.Clients import FFLiteLLMClient
from ffai.FFAI import FFAI

client = FFLiteLLMClient(
    model_string="mistral/mistral-small-latest",
    api_key="your-key",
)

ffai = FFAI(client)

result = ffai.workflow.generate_response(
    prompt="What is 2+2?",
    prompt_name="math_question"
)

print(result.response)       # "2 + 2 equals 4."
print(result.usage)          # TokenUsage(input_tokens=30, output_tokens=9, total_tokens=39)
print(result.cost_usd)       # 3e-06
print(result.duration_ms)    # 842.3
```

### Named prompt references

```python
ffai.workflow.generate_response(
    prompt="What is the capital of France?",
    prompt_name="geography"
)

result = ffai.workflow.generate_response(
    prompt="Write a poem about {{geography.response}}",
    prompt_name="poem"
)

print(result.response)
# Ode to Paris
#
# Oh, Paris, jewel of the Seine's embrace,
# A city of light, of love, of grace...
```

### Multi-step with dependencies

```python
from ffai import ResponseOptions

ffai.workflow.generate_response(
    prompt="List three programming languages",
    prompt_name="languages"
)

result = ffai.workflow.generate_response(
    "Which of {{languages.response}} is best for beginners?",
    prompt_name="recommendation",
    options=ResponseOptions(dependencies=["languages"]),
)

print(result.response)
# Based on the three languages listed—Python, JavaScript, and Rust—
# Python is the best suited for beginners...
```

### RAG: Retrieval-Augmented Generation

The fastest way to get started is `RAG.from_config()`, which reads all settings from `config/main.yaml` and picks up your API key from the environment:

```python
from ffai.rag import RAG

rag = RAG.from_config()                        # reads config/main.yaml + MISTRAL_API_KEY
rag.index("Python is a high-level programming language...", source="python_intro")

# Search for relevant chunks
hits = rag.search("programming language")
for hit in hits:
    print(f"[{hit.score:.2f}] {hit.content[:80]}...")
# [0.78] Python is a high-level programming language...

# One-shot retrieval-augmented answer via FFAI
ffai_with_rag = FFAI(client, rag=rag)
result = ffai_with_rag.query("What is Python?")
print(result.answer)
# Python is a high-level programming language known for its readability
# and versatility. It supports multiple paradigms...
print(result.sources)
# ['python_intro']
```

You can also pass an API key explicitly:

```python
rag = RAG.from_config(api_key="your-key")
```

Or construct each component manually for full control:

```python
from ffai.rag.embed import Embeddings
from ffai.rag.store import VectorStore

embed = Embeddings("mistral/mistral-embed", api_key="your-key")
store = VectorStore(collection_name="my_kb", dir="./chroma_db")

rag = RAG(embed=embed, store=store, chunk_size=500, chunk_overlap=100)
```

#### Switching vector store backends

Use `get_store()` to pick a backend, or set `store_backend` in `config/main.yaml`:

```python
from ffai.rag.stores import get_store

# Qdrant local mode (no server needed)
store = get_store("qdrant", path="./qdrant_db", embedding_dim=1024)

# ChromaDB (default)
store = get_store("chroma", collection_name="my_kb", dir="./chroma_db")

rag = RAG(embed=embed, store=store)
```

Or via config:

```yaml
# config/main.yaml
rag:
  store_backend: qdrant
  store_config:
    path: "./qdrant_db"
    embedding_dim: 1024
```

All backends implement the same `VectorStoreBase` interface — swap backends without changing any other code.

RAG supports BM25 hybrid search (`bm25_alpha`), result reranking (`reranker="diversity"`), query expansion via LLM (`query_expander`), hierarchical chunking with parent-context retrieval (`chunker="hierarchical"`), local embedding models (`local/` prefix via `sentence-transformers`), embedding caching, and custom prompt templates.

You can also manage the RAG lifecycle directly through FFAI:

```python
ffai.rag.index(text, source="doc1")               # index a document -> int
ffai.rag.index(text, source="doc2", checksum="a") # skip if checksum unchanged -> int
count = ffai.rag.count()                           # get chunk count
print(f"Indexed {count} chunks")               # "Indexed 3 chunks"
hits = ffai.rag.search("query", top_k=5)           # raw search without generation
print(f"Found {len(hits)} hits")               # "Found 2 hits"
ffai.rag.delete("doc1")                            # remove by source
result = ffai.rag.query("question",                # retrieval-augmented answer
    top_k=5,
    max_context_chars=4000,
    allow_llm_on_empty=False,
    generate_timeout=30.0,
)
print(result.answer)       # "Based on the indexed documents..."
print(result.sources)      # ["doc2"]
print(f"${result.cost_usd:.6f}")
```

### Configuration with ResponseOptions

Use `ResponseOptions` for model overrides, conditions, and history injection:

```python
from ffai import ResponseOptions

result = ffai.workflow.generate_response(
    "Translate to French: Hello",
    prompt_name="translate",
    options=ResponseOptions(model="mistral/mistral-large-latest"),
    temperature=0.3,
)

print(result.response)
# The translation of "Hello" to French is: **"Bonjour"** (formal)
print(result.model)
# "mistral/mistral-large-latest"
```

### Structured Output

Pass a Pydantic model to get validated, typed responses with automatic retry:

```python
from pydantic import BaseModel, Field
from ffai import ResponseOptions

class Sentiment(BaseModel):
    label: str = Field(description="positive, negative, or neutral")
    confidence: float = Field(ge=0.0, le=1.0)

result = ffai.workflow.generate_response(
    "The food was amazing but service was slow.",
    options=ResponseOptions(response_model=Sentiment),
)

print(result.parsed.label)       # "neutral"
print(result.parsed.confidence)  # 0.7
```

Works with any LiteLLM-backed provider (Mistral, OpenAI, Anthropic, etc.) and the native `FFMistralSmall` client.

### Conditional Execution

Use `condition` and `abort_condition` in `ResponseOptions` to skip or abort prompts based on earlier results:

```python
from ffai import ResponseOptions

ffai.workflow.generate_response("List three languages", prompt_name="languages")

result = ffai.workflow.generate_response(
    "Which is easiest?",
    prompt_name="recommendation",
    options=ResponseOptions(
        condition="len({{languages.response}}) > 0",
        dependencies=["languages"],
    ),
)

print(result.status)           # "success"
print(result.condition_trace)  # None (only populated when condition evaluates to False)

# When a condition evaluates to False, the prompt is skipped and
# condition_trace contains the resolved expression:
# result.status           -> "skipped"
# result.condition_trace  -> 'len("Python, JavaScript, Rust") > 99999'
```

The condition DSL supports comparisons, boolean logic, and built-in functions (see [Condition DSL](#condition-dsl) below).

### Fallback Models

Configure fallback models on the LiteLLM client so that if the primary model fails, alternatives are tried in order:

```python
from ffai.Clients import FFLiteLLMClient

client = FFLiteLLMClient(
    model_string="mistral/mistral-small-latest",
    api_key="your-key",
    fallbacks=["mistral/mistral-medium-latest", "openai/gpt-4o-mini"],
)
# If mistral-small-latest fails, tries mistral-medium-latest, then openai/gpt-4o-mini
```

### Runtime Client Switching

Switch the underlying client at runtime without creating a new `FFAI` instance:

```python
from ffai.Clients import FFLiteLLMClient

new_client = FFLiteLLMClient(model_string="openai/gpt-4o", api_key="key")
ffai.set_client(new_client)
# Subsequent calls use the new client; history is preserved
```

### Async API

FFAI provides async counterparts for all RAG methods, plus an async DAG executor:

```python
from ffai.Clients import AsyncFFLiteLLMClient
from ffai.FFAI import FFAI

async_client = AsyncFFLiteLLMClient(
    model_string="mistral/mistral-small-latest",
    api_key="your-key",
)
ffai = FFAI(async_client, rag=rag)

# Async RAG
result = await ffai.rag.aquery("What is Python?")
print(result.answer)
# Python is a high-level programming language known for its readability
hits = await ffai.rag.asearch("query", top_k=5)
print(f"Found {len(hits)} hits")
# Found 2 hits
count = await ffai.rag.aindex(text, source="doc1")
print(f"Indexed {count} chunks")
# Indexed 1 chunks

# Async DAG execution
prompts = [
    {"prompt_name": "topic", "prompt": "Suggest a topic"},
    {"prompt_name": "outline", "prompt": "Create an outline about {{topic.response}}",
     "history": ["topic"]},
    {"prompt_name": "article", "prompt": "Write an article based on:\n{{outline.response}}",
     "history": ["outline"]},
]
graph_result = await ffai.workflow.execute_graph(prompts, max_concurrency=10)
for name, r in graph_result.results.items():
    print(f"{name}: {r.status} ({r.duration_ms:.0f}ms)")
# topic: success (842ms)
# outline: success (1203ms)
# article: success (2156ms)
```

`execute_graph()` requires an `AsyncFFAIClient` client and executes prompts in topological-parallel order using `asyncio.gather`. Sequence numbers are auto-assigned from list position.

## ResponseResult

Every `generate_response()` call returns a `ResponseResult` dataclass:

| Field | Type | Description |
|-------|------|-------------|
| `response` | `Any` | The cleaned AI response |
| `resolved_prompt` | `str` | The fully interpolated prompt sent to the model |
| `usage` | `TokenUsage \| None` | Token counts from the API call |
| `cost_usd` | `float` | Estimated cost in USD |
| `model` | `str` | Model identifier used |
| `duration_ms` | `float` | Wall-clock duration in milliseconds |
| `status` | `str` | `"success"`, `"skipped"`, or `"failed"` |
| `condition_trace` | `str \| None` | Resolved condition expression (set when condition evaluates to `False`) |
| `condition_error` | `str \| None` | Error if condition evaluation failed |
| `parsed` | `Any` | Validated Pydantic model (when `response_model` is used) |
| `parsing_errors` | `list[str] \| None` | Validation errors from structured output |

## Condition DSL

The condition evaluator uses AST-based safe evaluation (no `eval()`). It supports:

**Comparisons and logic:**
```
len({{languages.response}}) > 0
{{languages.status}} == "success" and not is_empty({{languages.response}})
```

**JSON navigation:**
```
json_get({{analysis.response}}, "sentiment.label") == "positive"
json_has({{data.response}}, "items")
json_keys({{data.response}})
```

**String operations:**
```
"error" in lower({{result.response}})
trim({{raw.response}}) != ""
```

**Regex matching (via `%` operator):**
```
{{result.response}} % r"\d{4}"
```

**Arithmetic and ternary:**
```
len({{items.response}}) > 5 if len({{items.response}}) > 0 else False
```

**Built-in functions:** `len`, `int`, `float`, `str`, `bool`, `abs`, `min`, `max`, `round`, `lower`, `upper`, `trim`, `strip`, `lstrip`, `rstrip`, `split`, `rsplit`, `replace`, `count`, `find`, `rfind`, `slice`, `is_null`, `is_empty`, `json_parse`, `json_get`, `json_get_default`, `json_has`, `json_keys`, `json_values`, `json_type`.

## History & Query API

FFAI maintains four parallel history stores:

| Store | Access | Description |
|-------|--------|-------------|
| `history` | `ffai.history.raw` | Raw interaction list |
| `clean_history` | `ffai.history.clean` | Cleaned interaction list |
| `ordered_history` | `ffai.history.ordered` | OrderedPromptHistory with sequence numbers |
| `permanent_history` | `ffai.history.permanent` | PermanentHistory with timestamps, incremental retrieval |

### Query methods

```python
ffai.history.get_all_interactions()                              # list[dict]
ffai.history.get_latest_interaction()                             # dict | None
ffai.history.get_latest_interaction_by_prompt_name("analysis")   # dict | None
ffai.history.get_last_n_interactions(5)                           # list[dict]
ffai.history.get_interaction(sequence_number=3)                   # dict | None
ffai.history.get_model_interactions("mistral-small-latest")       # list[dict]
ffai.history.get_interactions_by_prompt_name("summary")           # list[dict]
ffai.history.get_prompt_history()                                 # list[str]
ffai.history.get_response_history()                               # list[str]
ffai.history.get_model_usage_stats()                              # dict[str, int]
# {"mistral-small-latest": 3}
ffai.history.get_prompt_name_usage_stats()                        # dict[str, int]
# {"math_question": 1, "geography": 1, "languages": 1}
ffai.history.get_prompt_dict()                                    # dict[str, list[dict]]
ffai.history.get_latest_responses_by_prompt_names(["a", "b"])     # dict
# {"a": {"prompt": "...", "response": "..."}, "b": {...}}
ffai.history.get_formatted_responses(["analysis", "summary"])     # str
```

### DataFrame export

```python
df = ffai.history.history_to_dataframe()
df = ffai.history.clean_history_to_dataframe()
df = ffai.history.ordered_history_to_dataframe()
df = ffai.history.search_history(text="error", model="gpt-4o")
df = ffai.history.get_model_stats_df()
df = ffai.history.get_prompt_name_stats_df()
df = ffai.history.get_response_length_stats()
df = ffai.history.interaction_counts_by_date()
print(df.head())
# shape: (3, 8)
# ┌──────────┬──────────┬──────────┬──────────┬──────────┬─────────┬─────────┬──────────┐
# │ prompt   ┆ response ┆ prompt_n ┆ timestam ┆ model    ┆ history ┆ status  ┆ datetime │
# │ ---      ┆ ---      ┆ ame      ┆ p        ┆ ---      ┆ ---     ┆ ---     ┆ ---      │
# │ str      ┆ str      ┆ ---      ┆ ---      ┆ str      ┆ null    ┆ str     ┆ datetime │
# │          ┆          ┆ str      ┆ f64      ┆          ┆         ┆         ┆ [μs]     │
# ╞══════════╪══════════╪══════════╪══════════╪══════════╪═════════╪═════════╪══════════╡
# │ What is… ┆ 2 + 2    ┆ math_qu… ┆ 1.78e9   ┆ mistral… ┆ null    ┆ success ┆ 2026-05… │
# │ …        ┆ …        ┆ …        ┆ …        ┆ …        ┆ …       ┆ …       ┆ …        │
# └──────────┴──────────┴──────────┴──────────┴──────────┴─────────┴─────────┴──────────┘
ffai.history.persist_all_histories()  # write Parquet to configured directory
```

### Client conversation history

```python
ffai.get_client_conversation_history()
# [{'role': 'user', 'content': 'Hello'}, {'role': 'assistant', 'content': 'Hi there!'}]
ffai.set_client_conversation_history(history)
ffai.add_client_message("user", "Hello")
ffai.clear_conversation()
```

## DAG Execution

Define prompt graphs with dependencies and conditions, then execute them in parallel:

```python
prompts = [
    {"prompt_name": "topic", "prompt": "Suggest a topic"},
    {"prompt_name": "outline", "prompt": "Create an outline about {{topic.response}}",
     "history": ["topic"]},
    {"prompt_name": "article", "prompt": "Write an article based on:\n{{outline.response}}",
     "history": ["outline"]},
]

graph, warnings = ffai.workflow.validate_graph(prompts)  # validate without executing
print(f"Graph has {len(graph.nodes)} nodes, {len(graph.edges)} edges")
# Graph has 3 nodes, 2 edges
if warnings:
    for w in warnings:
        print(f"Warning: {w}")
```

For async DAG execution with an `AsyncFFLiteLLMClient`, see [Async API](#async-api).

## Agent Tools & Validation

```python
from ffai import AgentLoop, AgentResult, ToolRegistry, ToolDefinition, ResponseValidator

registry = ToolRegistry()
registry.register(ToolDefinition(
    name="search",
    description="Search the web",
    parameters={"type": "object", "properties": {"query": {"type": "string"}}},
))
registry.register_executor("search", lambda args: f"Results for: {args['query']}")

loop = AgentLoop(
    client,
    registry,
    max_rounds=5,
    tool_timeout=30.0,
    continue_on_tool_error=True,
)
result = loop.execute(prompt="Search for info about X", tools=["search"])
print(result.response)             # LLM's synthesized answer (may be empty if max_rounds exceeded)
print(result.tool_calls_count)     # number of tool calls made
print(result.tool_calls[0].result) # "Results for: X"
print(result.status)               # "success", "failed", or "max_rounds_exceeded"

# Serialization
data = result.to_dict()
restored = AgentResult.from_dict(data)
```

### Response validation with re-execution

```python
validator = ResponseValidator(client)
result = validator.validate(
    response=some_response,
    criteria="Response must mention at least 3 key points",
)

print(result.passed)    # True or False
print(result.critique)  # None if passed, reason if failed

if not result.passed:
    # Re-execute with rejection feedback
    new_result = ffai.workflow.generate_response(
        f"Previous attempt was rejected: {result.critique}\n\nOriginal prompt: Write a summary",
    )
```

## Architecture

```
ffai/
  FFAI.py                          # High-level declarative wrapper + RAG lifecycle
  FFAIClientBase.py                # Re-export: ffai.core.client_base.FFAIClientBase
  ConversationHistory.py           # Re-export: ffai.core.history.conversation.ConversationHistory
  OrderedPromptHistory.py          # Re-export: ffai.core.history.ordered.OrderedPromptHistory
  config.py                        # YAML-based configuration (pydantic-settings)
  retry_utils.py                   # Tenacity-based retry decorators
  core/
    client_base.py                 # Abstract base class for all providers
    async_client_base.py           # Async abstract base class
    async_executor.py              # Async DAG executor (asyncio.gather per level)
    response_executor.py           # Orchestration: prompt resolve + condition + retry
    prompt_builder.py              # {{name.response}} interpolation engine
    prompt_utils.py                # Regex-based prompt substitution
    graph.py                       # Dependency graph construction and condition eval
    graph_execution_helpers.py     # Prompt resolution and abort checking for DAG
    prompt_node.py                 # PromptNode dataclass for execution dependency graph
    types.py                       # Shared TypedDicts (Interaction, PromptSpec)
    condition_evaluator.py         # AST-based safe condition evaluation + JSON DSL
    structured_output.py           # Pydantic-validated structured output
    response_options.py            # ResponseOptions frozen dataclass
    response_result.py             # Typed response container
    response_context.py            # Thread-safe shared history
    execution_result.py            # Internal ExecutionResult dataclass
    execution_state.py             # Thread-safe parallel execution state
    conversation_manager.py        # Client conversation suspend/restore
    response_utils.py              # Response cleaning, JSON extraction (json-repair)
    usage.py                       # TokenUsage dataclass
    history_exporter.py            # Polars DataFrame export + Parquet persistence
    history/
      ordered.py                   # Ordered prompt-response history (sequence numbers)
      permanent.py                 # Chronological turn history (timestamps, incremental)
      conversation.py              # Provider-facing message history (structured content blocks)
      recorder.py                  # History recording coordinator (all stores)
  Clients/
    BaseLiteLLMClient.py           # Shared mixin for sync/async LiteLLM clients
    FFLiteLLMClient.py             # Sync universal client (100+ providers, fallback chains)
    AsyncFFLiteLLMClient.py        # Async LiteLLM client (litellm.acompletion, fallback chains)
    FFMistralSmall.py              # Native Mistral SDK client (test_connection)
    model_defaults.py              # Per-model defaults + register_model_defaults()
  rag/
    rag.py                         # RAG class — index, index_many, chunk, search, query, delete, from_config
    embed.py                       # Embeddings (API + local/ models, LRU caching, cosine similarity)
    store.py                       # VectorStore backward-compat shim (re-exports ChromaVectorStore)
    stores/
      __init__.py                  # Registry, get_store(), list_stores(), list_available_stores()
      base.py                      # VectorStoreBase ABC — 10 abstract methods
      chroma.py                    # ChromaVectorStore — ChromaDB persistent client
      qdrant.py                    # QdrantVectorStore — server, local, or cloud mode
      pgvector.py                  # PgVectorStore — PostgreSQL + pgvector extension
      sqlite_vss.py                # SQLiteVssStore — zero-infrastructure SQLite extension
    types.py                       # SearchHit (parent_content), QueryResult (usage/cost/duration), TextChunk
    format.py                      # format_hits() for prompt injection
    prompts.py                     # DEFAULT_RAG_PROMPT template
    client_adapter.py              # ClientAdapter — wraps FFAIClientBase as callable
    _async.py                      # run_sync() — safe async-in-sync (handles Jupyter event loops)
    splitters/
      factory.py                   # get_chunker(), list_chunkers()
      base.py                      # ChunkerBase abstract class
      character.py                 # CharacterChunker
      recursive.py                 # RecursiveChunker
      markdown.py                  # MarkdownChunker
      code.py                      # CodeChunker
      hierarchical.py              # HierarchicalChunker (parent-child)
    indexing/
      bm25.py                      # BM25Index — Okapi BM25 keyword search
      hierarchical.py              # HierarchicalIndex — parent-child context enhancement
      contextual.py                # ContextualEmbeddings, LateChunkingEmbeddings
      deduplication.py             # ChunkDeduplicator — hash + similarity dedup
    search/
      hybrid.py                    # HybridSearch, reciprocal_rank_fusion()
      rerankers.py                 # NoopReranker, DiversityReranker, CrossEncoderReranker
      query_expansion.py           # QueryExpander, fuse_search_results()
  agent/
    agent_loop.py                  # Multi-round tool-call loop (timeouts, error continuation)
    agent_result.py                # AgentResult, ToolCallRecord (to_dict/from_dict)
    response_validator.py          # LLM-as-judge validation + re-execution
  tools/
    tool_registry.py               # Declarative tools, python: dynamic imports, from_dict
  observability/
    telemetry.py                   # OpenTelemetry spans (reset/reload for testing)
    log_context.py                 # ContextVar-based logging (batch_name, prompt_name)
```

## Examples

Runnable Jupyter notebooks in `examples/`:

| Notebook | What it demonstrates |
|----------|---------------------|
| `rag_chunking/` | 5 chunking strategies (character, recursive, markdown, code, hierarchical) |
| `rag_embeddings/` | Embedding generation, cosine similarity, caching, local models |
| `rag_pipeline/` | Full RAG pipeline: chunk -> embed -> index -> search -> format -> prompt |
| `rag_search/` | BM25, hybrid search, RRF fusion, rerankers, query expansion |
| `structured_output/` | Pydantic-validated structured output with automatic retry |
| `response_options_api/` | ResponseOptions: models, JSON mode, conditions, history |
| `async_dag_executor/` | Async DAG execution with fan-out, diamond, and conditional patterns |
| `agent_tools_and_validation/` | Agentic tool-call loop with LLM-as-judge validation |
| `conditional_execution/` | Condition-based branching and skip in prompt sequences |
| `multi_turn_sequence/` | Multi-turn conversation with history and DataFrame export |
| `vector_stores/` | Vector store backends: Qdrant (memory, local, server, cloud), ChromaDB vs Qdrant comparison |
| `dag_validation/` | DAG topology validation, cycle detection, dependency analysis |
| `message_stack.ipynb` | Conversation history stack inspection |
| `message_stack_live.ipynb` | Live conversation history stack demo |

## Configuration

FFAI reads YAML config from a `config/` directory. Values are resolved in priority order: **constructor kwargs > environment variables > YAML files**.

Config files:

- `config/main.yaml` — retry, RAG, and observability settings
- `config/clients.yaml` — client type definitions with API key env vars and model strings
- `config/paths.yaml` — file system paths for data persistence
- `config/model_defaults.yaml` — per-model default parameters (temperature, max_tokens, etc.)
- `config/logging.yaml` — logging format, level, and rotation settings

### Client configuration

FFAI uses [LiteLLM](https://github.com/BerriAI/litellm) as its primary routing layer, supporting 100+ providers through a unified interface. For the full list of supported providers and their model string formats, see the [LiteLLM Providers docs](https://docs.litellm.ai/docs/providers).

Add providers in `config/clients.yaml`:

```yaml
default_client: "litellm-mistral-small"

client_types:
  litellm-mistral-small:
    client_class: "FFLiteLLMClient"
    type: "litellm"
    provider_prefix: "mistral/"
    api_key_env: "MISTRAL_API_KEY"
    default_model: "mistral-small-latest"
    fallbacks:
      - "openai/gpt-4o-mini"

  litellm-openai:
    client_class: "FFLiteLLMClient"
    type: "litellm"
    provider_prefix: "openai/"
    api_key_env: "OPENAI_API_KEY"
    default_model: "gpt-4o-mini"
```

Common model string prefixes:

| Prefix | Example | API key env var |
|--------|---------|----------------|
| `openai/` | `openai/gpt-4o` | `OPENAI_API_KEY` |
| `anthropic/` | `anthropic/claude-3-5-sonnet-20241022` | `ANTHROPIC_API_KEY` |
| `mistral/` | `mistral/mistral-small-latest` | `MISTRAL_API_KEY` |
| `azure/` | `azure/my-deployment` | `AZURE_OPENAI_API_KEY` |
| `gemini/` | `gemini/gemini-1.5-pro` | `GEMINI_API_KEY` |
| `groq/` | `groq/llama-3.1-70b-versatile` | `GROQ_API_KEY` |
| `deepseek/` | `deepseek/deepseek-chat` | `DEEPSEEK_API_KEY` |
| `ollama/` | `ollama/llama3` | (local, no key) |

### Retry settings

Defined in `config/main.yaml`:

```yaml
retry:
  max_attempts: 3
  min_wait_seconds: 1
  max_wait_seconds: 60
  exponential_base: 2
  exponential_jitter: true
  retry_on_status_codes:
    - 429
    - 503
    - 502
    - 504
```

### RAG configuration

`RAG.from_config()` reads settings from `config/main.yaml` under the `rag:` key:

```yaml
rag:
  enabled: true
  store_backend: chroma
  store_config: {}
  persist_dir: "./chroma_db"
  collection_name: "ffai_kb"
  embedding_model: "mistral/mistral-embed"
  chunker: "recursive"
  chunk_size: 1000
  chunk_overlap: 200
  bm25_alpha: 0.5
  reranker: "diversity"
```

The `store_backend` field selects the vector store (`chroma`, `qdrant`, `pgvector`, `sqlite_vss`). Pass backend-specific constructor arguments via `store_config`. See [Vector Store Backends](#vector-store-backends) above.

```python
from ffai.rag import RAG

# Zero-config: reads embedding_model from config, API key from MISTRAL_API_KEY env var
rag = RAG.from_config()

# Or with explicit API key
rag = RAG.from_config(api_key="your-key")
```

### Model defaults

Define per-model defaults in `config/model_defaults.yaml`:

```yaml
model_defaults:
  generic:
    max_tokens: 4096
    temperature: 0.7
    system_instructions: "You are a helpful assistant."
  models:
    mistral/mistral-small-latest:
      max_tokens: 32000
      temperature: 0.7
```

Or register programmatically:

```python
from ffai.Clients.model_defaults import register_model_defaults

register_model_defaults("my-custom-model", {
    "temperature": 0.3,
    "max_tokens": 4096,
})
```

## Adding a Provider

Subclass `FFAIClientBase` and implement five abstract methods:

```python
from ffai.core.client_base import FFAIClientBase

class MyProvider(FFAIClientBase):
    def generate_response(self, prompt, **kwargs):
        ...
    def clear_conversation(self):
        ...
    def get_conversation_history(self):
        ...
    def set_conversation_history(self, history):
        ...
    def clone(self):
        ...
```

The base class provides `_extract_token_usage()`, `_trace_llm_call()`, `last_duration_ms`, `add_tool_result()`, and retry configuration out of the box.

For async providers, subclass `AsyncFFAIClientBase` with async versions of the same methods.

## Public API

The top-level `ffai` package exports:

| Symbol | Module |
|--------|--------|
| `FFAI` | `ffai.FFAI` |
| `FFAIClientBase` | `ffai.core.client_base` |
| `AsyncFFAIClientBase` | `ffai.core.async_client_base` |
| `ResponseOptions` | `ffai.core.response_options` |
| `ResponseExecutor` | `ffai.core.response_executor` |
| `ConditionEvaluator` | `ffai.core.condition_evaluator` |
| `ExecutionGraph` | `ffai.core.graph` |
| `ExecutionResult` | `ffai.core.execution_result` |
| `GraphResult` | `ffai.core.async_executor` |
| `AsyncGraphExecutor` | `ffai.core.async_executor` |
| `AgentLoop` | `ffai.agent.agent_loop` |
| `AgentResult` | `ffai.agent.agent_result` |
| `ToolCallRecord` | `ffai.agent.agent_result` |
| `ResponseValidator` | `ffai.agent.response_validator` |
| `ValidationResult` | `ffai.agent.response_validator` |
| `ToolRegistry` | `ffai.tools.tool_registry` |
| `ToolDefinition` | `ffai.tools.tool_registry` |
| `ConversationHistory` | `ffai.core.history.conversation` |
| `OrderedPromptHistory` | `ffai.core.history.ordered` |
| `PermanentHistory` | `ffai.core.history.permanent` |

With `pip install -e ".[rag]"`, additional RAG exports are available:

| Symbol | Module |
|--------|--------|
| `RAG` | `ffai.rag.rag` |
| `Embeddings` | `ffai.rag.embed` |
| `VectorStoreBase` | `ffai.rag.stores.base` |
| `get_store` | `ffai.rag.stores` |
| `list_stores` | `ffai.rag.stores` |
| `list_available_stores` | `ffai.rag.stores` |
| `is_store_available` | `ffai.rag.stores` |
| `SearchHit` | `ffai.rag.types` |
| `QueryResult` | `ffai.rag.types` |
| `TextChunk` | `ffai.rag.splitters.base` |
| `ClientAdapter` | `ffai.rag.client_adapter` |
| `DEFAULT_RAG_PROMPT` | `ffai.rag.prompts` |
| `GenerationResult` | `ffai.rag.types` |

## Requirements

- Python >= 3.10
- See `pyproject.toml` for full dependencies

## License

MIT
