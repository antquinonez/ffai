---
name: docstring-writing
description: Use when writing, reviewing, or adding docstrings to Python source files. Load BEFORE writing or modifying any docstrings. Covers documentation order (outside-in), Google-style conventions, coverage rules, content quality, insertion workflow, and validation.
license: MIT
---

# Docstring Writing Skill

Read this before writing or modifying docstrings. Every docstring must be
accurate, complete, and match the project's Sphinx rendering pipeline.

## Style: Google (Napoleon)

The project uses Google-style docstrings rendered by `sphinx.ext.napoleon`.
Follow the patterns already in the codebase. Refer to `src/retry_utils.py`,
`src/tools/tool_registry.py`, and `src/agent/agent_result.py` as canonical
examples.

### Module docstrings

```python
"""Short imperative summary of the module.

Extended description of what the module provides, its design
decisions, and how it fits into the broader system.
"""
```

Module docstrings appear as the first statement after the copyright header
and any `from __future__` imports.

### Class docstrings

```python
class Foo:
    """One-line summary of the class.

    Extended description of behavior, invariants, and usage notes.

    Attributes:
        name: Human-readable identifier.
        config: Dictionary of provider-specific settings.
        enabled: Whether this instance is active.
    """
```

For dataclasses, list every field in `Attributes:` with its type and meaning.
The type annotation comes from the source — never restate it; describe the
semantic purpose instead.

### Method and function docstrings

```python
def process(self, items: list[str], strict: bool = False) -> int:
    """Process items and return the count of successful operations.

    Args:
        items: Sequence of item identifiers to process.
        strict: If True, raise on first failure instead of continuing.

    Returns:
        Number of items processed successfully.

    Raises:
        ValueError: If ``items`` is empty and ``strict`` is True.
        RuntimeError: If the processing backend is unavailable.
    """
```

Sections in order (omit empty sections):

1. Summary line (imperative mood, one sentence, first Capital)
2. Blank line + extended description (if needed)
3. `Args:` — one entry per parameter (omit `self`, omit if no parameters)
4. `Returns:` — type and meaning (omit if returns `None`)
5. `Raises:` — exception types and conditions (omit if none raised)
6. `Example:` — only when usage is non-obvious (follow `doc-writing` skill rules)

## DS-0: Documentation Order (Outside-In)

When adding or reviewing docstrings across a project, follow this order:

### Layer 1: Module docstrings

Every `.py` file gets a module docstring first. These are cheap to write and
anchor the mental model for everything that follows. A good module docstring
means method docstrings can reference concepts instead of re-explaining them.

### Layer 2: Core public classes

Document the high-centrality entry points next. These are the classes that
consumers import and interact with directly. They establish the vocabulary
that lower-level docstrings will reference. For this project:

- `FFAI` — the main facade
- `FFAIClientBase` / `AsyncFFAIClientBase` — client abstraction
- `ExecutionGraph` — DAG definition
- `RAG` — retrieval-augmented generation
- `ToolRegistry` — agentic tool management
- `ResponseOptions` — call configuration

### Layer 3: Methods of core classes

Document the public methods of the classes from Layer 2. These form the
primary API surface.

### Layer 4: Supporting classes and functions

Leaf modules, utility functions, and internal classes. These reference
concepts established in Layers 1–3, so they benefit from having those
docstrings already in place.

### Layer 5: Review pass

After all docstrings are in place, regenerate the documentation with
`/update-docs` and review for consistency. Check that terminology is
uniform across modules and that cross-references resolve correctly.

### Exception: feature-branch documentation

When working on a specific feature, document what you touch regardless of
its layer. Don't leave undocumented code behind because it's "out of order."
This rule overrides the layering for incremental work.

## DS-1: Coverage Rules

### Must have docstrings

- Every module (top-level `"""..."""`)
- Every public class
- Every public method of a public class
- Every standalone public function
- Every public dataclass field (via `Attributes:` in the class docstring)

### May omit docstrings

- `__init__` — only when the class docstring already documents construction
- Private methods (names starting with `_`, excluding dunder methods)
- Trivial methods whose behavior is obvious from the name (`to_dict`,
  `from_dict`, `__repr__`, `__len__`)
- Properties that simply return a stored attribute

### Never add docstrings to

- Test files
- `__init__.py` re-export modules (the source modules are documented)
- Type aliases and `TypeVar` declarations

## DS-2: Content Quality

### Describe contracts, not implementation

```python
# Bad — describes the code
def search(self, query: str) -> list[str]:
    """Loops through the index and filters matching entries."""

# Good — describes the contract
def search(self, query: str) -> list[str]:
    """Return index entries matching ``query``, ranked by relevance."""
```

### Derive types from signatures, not memory

Read the actual function signature before writing `Args:` or `Returns:`.
The parameter names, types, and defaults in the docstring must match the
signature exactly. A mismatch between `items: list[str]` in the signature
and `items (str)` in the docstring is a documentation bug.

### Be specific about defaults

When a parameter has a default value, state what it means:

```python
# Bad
    strict: Whether to be strict. Defaults to False.

# Good
    strict: If True, raise on invalid input instead of returning None.
```

### Document side effects

If a method modifies state, makes network calls, writes to disk, or logs
warnings, mention it in the extended description.

### Document async behavior

For `async` methods, note whether they are safe to call concurrently and
whether they require an existing event loop.

## DS-3: Mechanical Insertion

### Single docstring — direct edit

For one-off additions, use the `edit` tool to insert the docstring as the
first statement of the function/class body. Match the indentation of the
existing body.

### Batch docstrings — use the script

For adding multiple docstrings across a file or module, use the project's
AST-based insertion tool:

```bash
.venv/bin/python scripts/add_docstrings.py \
    --map "retry_utils.py:RateLimitError=Base exception for rate limit errors." \
    --map "retry_utils.py:ServiceUnavailableError=Exception for service unavailable errors (503)." \
    --dry-run
```

Target syntax:
- `file.py:module` — module-level docstring
- `file.py:ClassName` — class docstring
- `file.py:ClassName.method` — method docstring
- `file.py:function_name` — standalone function docstring

Always run with `--dry-run` first. Verify the output, then re-run without it.

For multi-line docstrings passed via `--map`, use the programmatic API:

```python
from add_docstrings import apply_docstrings

apply_docstrings({
    ("retry_utils.py", None, "get_retry_decorator"): \"\"\"Create a retry decorator with configurable parameters.

    Args:
        max_attempts: Maximum number of retry attempts.
        min_wait: Minimum wait time in seconds.

    Returns:
        Configured retry decorator.
    \"\"\",
})
```

## DS-4: Validation Workflow

After adding or modifying docstrings, run these checks in order:

1. **Syntax** — `ast.parse` (the script does this automatically for batch
   inserts; for direct edits, the linter will catch it)
2. **Lint** — `ruff check src/ tests/` (catches D-series docstring violations
   if enabled, plus general issues)
3. **Type check** — `pyright src/ tests/` (ensures edits didn't break types)
4. **Tests** — `pytest tests/ -x -q` (ensures no runtime regressions)
5. **Doc build** — `/update-docs` or `scripts/generate_api_docs.py` (verifies
   docstrings render correctly in Sphinx)

If the doc build produces warnings about a docstring (unexpected indentation,
missing blank line), fix the docstring formatting — not the Sphinx config.

## DS-5: Common Mistakes

| Mistake | Example | Fix |
|---------|---------|-----|
| Wrong parameter name | `query_str` in doc but `query` in sig | Read the actual signature |
| Missing `Args:` for non-trivial params | Omitting args because they seem "obvious" | Every non-self parameter gets documented |
| Duplicating type in prose | ``text (str): The text`` | Just ``text: The text to process`` |
| Summary in third person | `Returns the count` | `Return the count` (imperative) |
| Closing triple-quote misaligned | `"""text\n  """` at wrong indent | Match body indentation exactly |
| Sphinx-incompatible formatting | Markdown headers, ``**bold**`` in docstrings | Use reStructuredText or plain text only |

## DS-6: Interaction with Other Skills

- When a docstring contains a **code example**, follow the `doc-writing` skill
  (DW-1 through DW-14) for validating it.
- When writing docstrings as part of a **new feature**, follow the
  `layered-design` skill for the implementation, then add docstrings as part
  of the L4 (integration) layer.
- After adding docstrings, run `/update-docs` to regenerate the Sphinx
  documentation and verify rendering.
