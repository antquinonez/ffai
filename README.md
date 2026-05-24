# FFAI

Declarative multi-provider AI client with named-prompt context assembly, thread-safe parallel execution, and built-in cost/usage tracking.

## What It Does

FFAI wraps AI provider clients (LiteLLM for 100+ providers, Mistral native SDK) and adds:

- **Declarative context assembly** -- reference earlier responses by name using `{{prompt_name.response}}` interpolation in prompts
- **Multi-client support** -- named client profiles with lazy instantiation and thread-safe cloning for parallel execution
- **Usage and cost tracking** -- automatic per-call token counting and cost estimation across providers
- **History management** -- four history views (raw, cleaned, prompt-attribute-keyed, ordered) with Polars DataFrame export
- **OpenTelemetry tracing** -- optional span emission with model, token, and cost attributes
- **Retry with exponential backoff** -- tenacity-based retry with configurable status codes and jitter

## Quick Start

```python
from src.Clients import FFLiteLLMClient
from src.FFAI import FFAI

client = FFLiteLLMClient(
    api_key="your-key",
    model="mistral/mistral-small-latest"
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
ffai.generate_response(
    prompt="List three programming languages",
    prompt_name="languages"
)

result = ffai.generate_response(
    prompt="Which of {{languages.response}} is best for beginners?",
    prompt_name="recommendation",
    dependencies=["languages"]
)
```

## Architecture

```
src/
  FFAI.py                    # High-level declarative wrapper
  config.py                  # YAML-based configuration (pydantic-settings)
  retry_utils.py             # Tenacity-based retry decorators
  core/
    client_base.py           # Abstract base class for all providers
    types.py                 # Shared type definitions
    prompt_builder.py        # {{name.response}} interpolation engine
    prompt_utils.py          # Regex-based prompt substitution
    response_utils.py        # Response cleaning, JSON extraction
    response_result.py       # Typed response container
    response_context.py      # Thread-safe shared history
    usage.py                 # TokenUsage dataclass
    pricing.py               # Per-model cost estimation
    history_exporter.py      # Polars DataFrame export
    history/
      ordered.py             # Ordered prompt-response history
      permanent.py           # Chronological turn history
      conversation.py        # Raw conversation history
  Clients/
    FFLiteLLMClient.py       # Universal client (100+ providers via LiteLLM)
    FFMistralSmall.py        # Native Mistral SDK client (reference implementation)
    model_defaults.py        # Per-model default parameters
  observability/
    telemetry.py             # OpenTelemetry span management
    log_context.py           # Context-aware logging
```

## Configuration

FFAI reads YAML config from a `config/` directory. Key files:

- `config/main.yaml` -- retry, observability, and other settings
- `config/clients.yaml` -- client type definitions
- `config/paths.yaml` -- file system paths
- `config/model_defaults.yaml` -- per-model default parameters

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

## License

MIT
