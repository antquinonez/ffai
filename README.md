# FFAI

Declarative multi-provider AI client with named-prompt context assembly, async DAG execution, RAG, agentic tool-call loops, and built-in cost/usage tracking.

## Features

- **Declarative context assembly** — reference earlier responses by name using `{{prompt_name.response}}` interpolation
- **RAG (Retrieval-Augmented Generation)** — chunking, embeddings, vector search, BM25 hybrid, reranking, query expansion, and `FFAI.query()` for one-shot retrieval-augmented answers
- **DAG execution** — dependency graph with topological-parallel prompt execution and condition-based branching
- **Async support** — async client base and DAG executor for concurrent I/O-bound workloads
- **Agentic execution loop** — multi-round tool-call-driven execution with automatic tool dispatch
- **Structured output** — Pydantic model validation with automatic retry on schema failures
- **Response validation** — LLM-as-judge PASS/FAIL validation with automatic re-execution
- **Multi-client support** — 100+ providers via LiteLLM, plus native Mistral SDK
- **Usage and cost tracking** — automatic per-call token counting and cost estimation across providers
- **History management** — four history views with Polars DataFrame export
- **OpenTelemetry tracing** — optional span emission with model, token, and cost attributes

## Quick Start

```python
from src.Clients import FFLiteLLMClient
from src.FFAI import FFAI

client = FFLiteLLMClient(
    model_string="mistral/mistral-small-latest",
    api_key="your-key",
)

ffai = FFAI(client)

result = ffai.generate_response(
    prompt="What is 2+2?",
    prompt_name="math_question"
)

print(result.response)
print(f"Tokens: {result.usage}")
print(f"Cost: ${result.cost_usd:.6f}")
```

### Named prompt references

```python
ffai.generate_response(
    prompt="What is the capital of France?",
    prompt_name="geography"
)

result = ffai.generate_response(
    prompt="Write a poem about {{geography.response}}",
    prompt_name="poem"
)
```

### Multi-step with dependencies

```python
from src import ResponseOptions

ffai.generate_response(
    prompt="List three programming languages",
    prompt_name="languages"
)

result = ffai.generate_response(
    "Which of {{languages.response}} is best for beginners?",
    prompt_name="recommendation",
    options=ResponseOptions(dependencies=["languages"]),
)
```

### RAG: Retrieval-Augmented Generation

```python
from src.rag import RAG
from src.rag.embed import Embeddings
from src.rag.store import VectorStore

embed = Embeddings("mistral/mistral-embed", api_key="your-key")
store = VectorStore(collection_name="my_kb", dir="./chroma_db")

rag = RAG(embed=embed, store=store, chunk_size=500, chunk_overlap=100)
rag.index("Python is a high-level programming language...", source="python_intro")

# Search for relevant chunks
hits = rag.search("programming language")
for hit in hits:
    print(f"[{hit.score:.2f}] {hit.content[:80]}...")

# One-shot retrieval-augmented answer
result = ffai.query("What is Python?")
print(result.answer)
print(f"Sources: {result.sources}")
```

RAG also supports BM25 hybrid search (`bm25_alpha`), result reranking (`reranker="diversity"`), query expansion via LLM (`query_expander`), and custom prompt templates. See the examples below.

### Configuration with ResponseOptions

Use ``ResponseOptions`` for model overrides, structured output, conditions, and history injection:

```python
from pydantic import BaseModel
from src import ResponseOptions

class Sentiment(BaseModel):
    label: str
    confidence: float

result = ffai.generate_response(
    "Analyze the tone",
    prompt_name="sentiment",
    options=ResponseOptions(
        model="gpt-4",
        response_model=Sentiment,
        condition='{{fetch.status}} == "success"',
        history=["fetch"],
    ),
    temperature=0.3,
)

print(result.parsed.label)       # "positive"
print(result.parsed.confidence)  # 0.95
```

## Architecture

```
src/
  FFAI.py                          # High-level declarative wrapper + query()
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
    condition_evaluator.py         # AST-based safe condition evaluation
    structured_output.py           # Pydantic-validated structured output
    response_options.py            # ResponseOptions frozen dataclass
    response_result.py             # Typed response container
    response_context.py            # Thread-safe shared history
    execution_result.py            # Internal ExecutionResult dataclass
    execution_state.py             # Thread-safe parallel execution state
    conversation_manager.py        # Client conversation suspend/restore
    response_utils.py              # Response cleaning, JSON extraction
    usage.py                       # TokenUsage dataclass
    history_exporter.py            # Polars DataFrame export
    history/
      ordered.py                   # Ordered prompt-response history
      permanent.py                 # Chronological turn history
      conversation.py              # Raw conversation history
      recorder.py                  # History recording coordinator
  Clients/
    BaseLiteLLMClient.py           # Shared mixin for sync/async LiteLLM clients
    FFLiteLLMClient.py             # Sync universal client (100+ providers via LiteLLM)
    AsyncFFLiteLLMClient.py        # Async LiteLLM client (litellm.acompletion)
    FFMistralSmall.py              # Native Mistral SDK client
    model_defaults.py              # Per-model default parameters
  rag/
    rag.py                         # RAG class — index, search, query, delete
    embed.py                       # Embeddings (API and local models, with caching)
    store.py                       # VectorStore (ChromaDB, async-native)
    types.py                       # SearchHit, QueryResult dataclasses
    format.py                      # format_hits() for prompt injection
    prompts.py                     # DEFAULT_RAG_PROMPT template
    client_adapter.py              # ClientAdapter — wraps FFAIClientBase as callable
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
      hierarchical.py              # Parent-child context enhancement
      contextual.py                # Contextual embeddings
      deduplication.py             # Chunk deduplication
    search/
      hybrid.py                    # HybridSearch, reciprocal_rank_fusion()
      rerankers.py                 # NoopReranker, DiversityReranker, CrossEncoderReranker
      query_expansion.py           # QueryExpander, fuse_search_results()
  agent/
    agent_loop.py                  # Multi-round tool-call execution loop
    agent_result.py                # AgentResult and ToolCallRecord dataclasses
    response_validator.py          # LLM-as-judge response validation with retry
  tools/
    tool_registry.py               # Declarative tool definition and execution
  observability/
    telemetry.py                   # OpenTelemetry span management
    log_context.py                 # Context-aware logging
```

## Examples

Eleven runnable Jupyter notebooks in `examples/`:

| Notebook | What it demonstrates |
|----------|---------------------|
| `rag_chunking/` | 5 chunking strategies (character, recursive, markdown, code, hierarchical) |
| `rag_embeddings/` | Embedding generation, cosine similarity, caching, local models |
| `rag_pipeline/` | Full RAG pipeline: chunk → embed → index → search → format → prompt |
| `rag_search/` | BM25, hybrid search, RRF fusion, rerankers, query expansion |
| `structured_output/` | Pydantic-validated structured output with automatic retry |
| `response_options_api/` | ResponseOptions: models, JSON mode, conditions, history |
| `async_dag_executor/` | Async DAG execution with fan-out, diamond, and conditional patterns |
| `agent_tools_and_validation/` | Agentic tool-call loop with LLM-as-judge validation |
| `conditional_execution/` | Condition-based branching and skip in prompt sequences |
| `multi_turn_sequence/` | Multi-turn conversation with history and DataFrame export |
| `dag_validation/` | DAG topology validation, cycle detection, dependency analysis |

## Configuration

FFAI reads YAML config from a `config/` directory. Key files:

- `config/main.yaml` — retry, observability, and other settings
- `config/clients.yaml` — client type definitions
- `config/paths.yaml` — file system paths
- `config/model_defaults.yaml` — per-model default parameters

## Adding a Provider

Subclass `FFAIClientBase` and implement five abstract methods:

```python
from src.core.client_base import FFAIClientBase

class MyProvider(FFAIClientBase):
    def generate_response(self, prompt, **kwargs):
        # Call your provider's API
        ...

    def clear_conversation(self):
        ...

    def get_conversation_history(self):
        ...

    def set_conversation_history(self, history):
        ...

    def clone(self):
        # Return fresh copy with same config
        ...
```

The base class provides `_extract_token_usage()`, `_trace_llm_call()`, and retry configuration out of the box.

## Requirements

- Python >= 3.10
- See `pyproject.toml` for full dependencies

### Optional: RAG with vector storage

```bash
pip install -e ".[rag]"
```

This installs ChromaDB for persistent vector storage. Embeddings work without it (API-based models call the provider directly; local models use `sentence-transformers`).

## License

MIT
