---
name: ffai-testing
description: Use when writing, reviewing, or enhancing tests for the FFAI-standalone project. Triggers on test-related tasks including writing new tests, fixing test failures, improving test quality, or auditing test coverage. Load BEFORE writing or modifying any test file.
license: MIT
---

# FFAI Testing Principles

Read this before writing or reviewing tests. Tests must verify **correct behavior**, not exercise code paths for coverage. A test that passes without asserting anything meaningful is worse than no test.

## Organization

- Use pytest with class-based test organization
- Place shared fixtures in `conftest.py` — do not copy-paste fixtures across test classes
- Name test files as `test_<module>.py`, classes as `Test<Feature>`, methods as `test_<description>`
- Import modules inside test methods when mocking is needed

## Principles

### TP-1: Assert specific values, not just types

Every test must assert at least one specific, predictable value. `isinstance(result, float)`, `isinstance(result, list)`, and `result is not None` are not sufficient assertions on their own.

Bad:
```python
results = ffai.get_response_history()
assert isinstance(results, list)
```

Good:
```python
results = ffai.get_response_history()
assert len(results) == 2
assert results[0] == "First response"
```

### TP-2: Assert the semantics, not the implementation

Tests should verify *what* the code computes, not *how* it computes it. If the test would need to change after a correct refactoring, the test is coupled to the wrong thing.

- For prompt building: assert the assembled prompt contains expected content — not that a specific private method was called.
- For graph construction: assert node levels, edge counts, or cycle detection results — not internal data structure layout.
- For condition evaluation: assert `is True` / `is False` for known inputs — not which AST nodes were visited.
- For response results: assert specific field values on `ResponseResult`, not just that the object exists.

### TP-3: Do not enshrine bugs as expected behavior

If a function returns a wrong result, write the test to assert the **correct** behavior and let it fail. Then fix the source code. Never write a passing test that asserts incorrect output just to gain coverage.

### TP-4: Every edge-case test needs a justification

When testing an edge case (empty prompt, missing history key, malformed JSON, cyclical dependencies), document *why* this edge case matters and what the correct behavior should be. Do not construct pathological inputs just because a code path exists.

### TP-5: Test error paths by asserting the error

Assert the specific exception type and error message content. Do not catch the exception and assert `True`.

Good:
```python
with pytest.raises(ValueError, match="Cycle detected"):
    graph.add_prompt("a", dependencies=["b"])
    graph.add_prompt("b", dependencies=["a"])
```

### TP-6: Coverage is a finding tool, not a target

Use coverage reports to identify untested code paths, then write tests that verify correct behavior on those paths. Do not write tests whose only purpose is to move the coverage number upward.

### TP-7: Avoid compound weak assertions

Prefer one strong assertion over several weak ones. Use `==` when the input deterministically produces a known result. Use `>=` only when the exact count depends on non-deterministic ordering.

Never use `or` in assertions — `assert "temperature" in defaults or "max_tokens" in defaults` passes if either is present, testing nothing specific.

### TP-8: Test observable behavior over internal state

Prefer testing through the public API. Directly accessing private attributes (`_model_string`, `_fallbacks`, `_context`, `_last_usage`) is acceptable for coverage of internal logic that cannot be observed through public methods, but the test must still assert specific values on those internals, not just their existence or type.

Private method tests (`_extract_json`, `_clean_response`, `_build_prompt`, `_convert_history_to_messages`) are acceptable because these methods represent distinct operations with no public API equivalent — but the test must assert the resulting value, not just that no exception was raised.

### TP-9: Use exact assertions on deterministic outputs

When the test input fully determines the output, use `==` not `<=` or `>=`. Use `>=` or `<=` only when the output is genuinely non-deterministic (LLM response content, token counts that vary by provider) or when testing a structural property.

Bad:
```python
assert mock_completion.call_count >= 2
```

Good:
```python
assert mock_completion.call_count == 3  # 1 primary + 2 retries
```

### TP-10: Verify expected values empirically

Before writing an exact assertion, run the code in isolation to confirm the expected value. Guessing at counts, lengths, or numeric results leads to test failures that waste review time. This is especially important for prompt interpolation, graph level assignment, and condition evaluation where the output depends on parsing logic.

### TP-11: Correctness over coverage

- **Invariants**: Test bounds, identities, and conservation laws. If token counts must be non-negative, assert it. If graph levels start at 0, assert it.
- **Consistency**: Two APIs computing the same thing must agree. If `ordered_history` and `prompt_attr_history` both record a response, they must agree on the content.
- **Independent verification**: Verify against independent calculation — not by running the code under test and copying its output.
- **Property tests over single-value tests**: Prefer testing structural properties (ordering, containment, idempotency) when the output has natural invariants.

### TP-12: Mock at the boundary, not the internals

Prefer mocking at the `generate_response` boundary over setting private attributes directly. When public API is not available for configuration needed in tests, prefer adding a constructor parameter to the production code rather than bypassing it with private attribute assignment.

Bad:
```python
client._model_string = "mistral-small-latest"
client._fallbacks = ["mistral-medium-latest"]
```

Good:
```python
client = FFLiteLLMClient(
    model="mistral-small-latest",
    fallbacks=["mistral-medium-latest"],
)
```

### TP-13: Assert DataFrame content, not just structure

FFAI uses Polars DataFrames for history export and statistics. Tests that verify DataFrames must check actual cell values, not just that the frame is non-empty or has expected column names.

Bad:
```python
df = ffai.history_to_dataframe()
assert not df.is_empty()
assert "model" in df.columns
```

Good:
```python
df = ffai.history_to_dataframe()
assert df.height == 1
assert "model" in df.columns
assert df["model"][0] == "mistral-small-latest"
assert df["response"][0] == "Test response"
```

When exact values depend on LLM output (integration tests), assert structural properties instead: column types, non-negative counts, monotonic ordering, or that specific expected columns contain non-null values.

### TP-14: No vacuous tests

Every test must contain at least one `assert` statement. A test that calls methods without asserting anything is worse than no test — it inflates the test count without verifying behavior. This applies especially to NoOp/teardown tests: verify the noop behavior (e.g., no span created, no provider called, no exception raised).

Bad:
```python
def test_shutdown_safe_when_disabled():
    manager = TelemetryManager()
    manager.shutdown()
```

Good:
```python
def test_shutdown_safe_when_disabled():
    manager = TelemetryManager()
    manager.shutdown()
    assert manager._provider is None
```

### TP-15: Eliminate copy-paste test setup

Use helper functions or shared fixtures for repeated test setup patterns. The following patterns appear repeatedly and must be extracted:

- **Concrete client stubs** — `ConcreteClient(FFAIClientBase)` with 5 abstract methods is defined 7+ times. Extract to a `conftest.py` fixture or helper.
- **Mock LLM responses** — `MagicMock` with `choices[0].message.content` is copy-pasted 9+ times. Extract to a `conftest.py` fixture.
- **FFAI + mock client construction** — `FFAI(mock_ffmistralsmall)` boilerplate appears 40+ times in `test_ffai.py`. Use a class-level or module-level fixture.

### TP-16: Verify the behavior you claim to test

If a test is named `test_generate_response_with_system_instructions`, it must verify that system instructions actually reached the client — not just that `len(history) == 1`. If a test is named `test_generate_response_with_thread_lock`, it must verify thread safety behavior — not just that a response was recorded.

## Test Commands

```bash
pytest tests/ -x -q                                       # Run all tests (fast, stop on first failure)
pytest tests/test_ffai.py -v                               # Run single test file
pytest tests/test_ffai.py::TestFFAIGenerateResponse -v     # Run single test class
pytest tests/test_ffai.py::TestFFAIGenerateResponse::test_foo -v  # Run single test method
pytest tests/ --cov=src --cov-report=term-missing          # Run with coverage
```

## Known Anti-Patterns in the Current Suite

When enhancing tests, watch for these patterns that already exist:

1. **`test_ffai.py`**: DataFrame tests check `is_empty()` and column names but never cell values (~10 instances). Tests named for specific features (system instructions, thread lock, dependencies) only assert `len(history) == 1` without verifying the claimed behavior.
2. **`test_telemetry.py`**: 4 tests with zero assertions (NoOp span tests, trace_llm_call tests). `ConcreteClient` stub defined 4 times instead of using a shared fixture.
3. **`test_observability.py`**: 3 tests with zero assertions (shutdown tests, NoOp span test). `TestNoOpSpan` class duplicates `test_telemetry.py`.
4. **`test_model_defaults.py`**: `or` in assertion makes copy behavior untested (`assert "temperature" in defaults or "max_tokens" in defaults`).
5. **`test_ffmistralsmall.py` and `test_fflitellm_client.py`**: Private attribute access (`_model_string`, `_fallbacks`) for configuration that should use constructor parameters. `mock_mistral_client` fixture defined locally with different values than `conftest.py` version — potential for confusion.
6. **`test_usage.py`**: `ConcreteClient` stub defined 3 times. Tests set `_last_usage` and `_last_cost_usd` directly instead of driving through `generate_response`.
7. **`test_history_restore_and_retry_fix.py`**: Uses `>=` for deterministic retry counts (`assert call_count >= 2`). Mock response setup copy-pasted 4+ times.
8. **`test_dag_integration.py`**: Defines its own `mock_client` + `ffai` fixtures instead of using `conftest.py`. `test_condition_true_executes` only asserts `result.response is not None` — any non-None response passes.
