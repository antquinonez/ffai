---
name: doc-writing
description: Use when writing, reviewing, or updating documentation that contains code examples — especially README files, guides, API references, or docstrings with embedded examples. Load BEFORE writing or modifying any documentation that includes runnable code blocks. Enforces empirical validation of all code examples against the actual source code.
license: MIT
---

# Documentation Writing Principles

Read this before writing or reviewing documentation with code examples. Every code block must be **empirically validated** against the running system. Aspirational or untested examples are bugs waiting to surface in users' code.

## Core Rules

### DW-1: Every code block must be run before publication

No exceptions. Paste the code into a script or REPL and verify it executes without error and produces output matching what the comments claim. "Looks correct" is not validation.

### DW-2: Output comments must match actual return values

After running a code block, compare the printed output against every comment that shows expected output. The most dangerous documentation bugs are code that *runs fine* but whose comments describe *wrong results*.

Example of a stealth lie:
```python
print(result.condition_trace)  # "len('Python, JavaScript') > 0"
```
When the actual value is `None` because the condition passed. The code doesn't error — it just misleads.

### DW-3: Cross-reference parameter names against source code

Never trust your memory for parameter names, dict keys, or constructor arguments. Read the actual source file. The most dangerous doc errors use plausible but wrong names that don't throw errors — they just silently produce wrong results.

Example: Using `"dependencies"` in a prompt dict when the graph builder reads `"history"`. Zero edges created, zero errors thrown, completely wrong behavior.

### DW-4: Examples must be internally consistent

Every example must be runnable with the credentials and client shown. No mixing Provider A's API key with Provider B's model. No referencing variables that haven't been defined in the example or a prior block on the same page.

Check:
- API key matches the provider in the model string
- All imports are present
- All variables are defined before use
- Return types match what the comments claim (str vs dict vs dataclass)

### DW-5: Test the full call chain, not just the immediate method

Integration bugs hide at the seams between components. If an example involves multiple layers (e.g., RAG → ClientAdapter → AsyncFFLiteLLMClient), run the complete chain end-to-end. Unit tests passing in isolation does not mean the integration works.

The most common failure mode: a sync wrapper calls an async method that returns a coroutine, and nobody awaits it. The result is a `<coroutine object>` string instead of the actual value.

### DW-6: Distinguish illustrative from empirical output

LLM-generated text varies between calls. Be explicit about what's deterministic and what's not:

- **Deterministic** (must be exact): column names, return types, field names, dict keys, status values, column counts
- **Illustrative** (may vary): response text, token counts, costs, durations, scores

For deterministic output, use exact values. For illustrative output, either:
1. Show a representative example and mark it with a comment like `# varies by run`, or
2. Show the structural shape without exact values

Never present illustrative output as if it were deterministic.

### DW-7: Verify every import resolves

Every `from X import Y` or `import X` in documentation must actually work. Check:
- The module exists at the path shown
- The symbol is exported from `__init__.py` if using a package-level import
- The symbol name is spelled correctly (capitalization matters)

### DW-8: Use representative inputs, not toys

Toy inputs ("test", "X", "hello") can trigger pathological behavior that real inputs wouldn't, or vice versa. Test examples with inputs representative of actual use. If the agent loop example hits max rounds with a toy prompt, that's a real documentation problem — users will hit it too.

### DW-9: Don't copy from examples/ or notebooks without re-running

Notebooks and example scripts may use different parameter names, older API patterns, or different import paths than what you're writing. Copy-paste is the single largest source of doc rot. Re-run every copied snippet in the exact form it will appear in the documentation.

### DW-10: After any code fix, re-run the affected doc examples

If you change source code to fix a bug revealed by documentation testing, re-run the doc example afterward. The fix may change the output, or the doc example may have been showing aspirational behavior that masked the bug.

### DW-11: Bare code blocks need output or a stated reason

A code block with no output (no `print`, no result comment) should either:
1. Add a representative `print()` or result comment, or
2. Explicitly state it's configuration/setup only

Readers copy-paste code blocks and expect to see something happen. Silent blocks are confusing.

### DW-12: Verify DataFrame/table shapes empirically

DataFrame column lists, column counts, and shapes are deterministic. Never write them from memory. Run the code and copy the actual column list and shape. APIs evolve — columns get added — and stale shapes are a trust-killer.

### DW-13: Re-verify after API changes

When the codebase changes (new fields, renamed parameters, changed defaults), re-run all affected doc examples. Treat documentation as a test suite that must pass against the current code.

### DW-14: Export table module paths must point to the definition site

When documenting a "Symbol → Module" table, verify each module path points to where the symbol is **defined**, not just where it's **importable from**. Re-exports through `__init__.py` make every chain resolve, so `from ffai.rag import TextChunk` works even though `TextChunk` is defined in `ffai.rag.splitters.base`, not `ffai.rag.types`. Import testing alone won't catch this — you must read the actual definition file.

This applies to all structured reference tables: Public API, exports, type tables, and constructor parameter tables.

## Validation Workflow

When writing or reviewing documentation with code examples:

1. **Catalog** — List every code block with its line range and whether it has output
2. **Prepare** — Set up a test environment with valid API keys and the current codebase
3. **Run** — Execute each code block individually (not all at once — isolate failures)
4. **Compare** — For each block, compare actual output against every comment
5. **Cross-reference** — For each block, read the source code for every method/class used
6. **Fix** — Fix the docs, fix the code, or both. If fixing code, re-run from step 3
7. **Re-verify** — After all fixes, run the full suite again to confirm nothing regressed

## Common Failure Patterns

These patterns recur across documentation reviews. Watch for them:

| Pattern | Symptom | Root Cause |
|---------|---------|------------|
| **Wrong dict key** | Code runs, produces wrong results silently | Parameter name guessed instead of read from source |
| **Stale shape** | Column count/names wrong in table output | API evolved, docs not re-verified |
| **Provider mismatch** | Auth error at runtime | Example mixes provider A's key with provider B's model |
| **Coroutine leak** | Output shows `<coroutine object>` | Async method called synchronously in integration path |
| **Aspirational output** | Comment shows ideal result, not actual | Writer tested the happy path mentally, not empirically |
| **Missing export** | ImportError at runtime | Symbol exists in module but not in `__init__.py` |
| **Undefined variable** | NameError | Example references a variable from a different section without redefining it |
| **Edge-case output** | Comment only shows success path | Failure/empty/None paths not documented |

## Anti-Patterns

### Writing code from memory

Every parameter name, dict key, constructor argument, and return type must be verified against the running source. If you can't point to the exact line in source that defines it, you haven't verified it.

### Showing only the happy path

If a method can return `None`, be empty, or fail, show at least one example of that case. Readers will encounter it.

### Mixing example contexts

Each example block should be self-contained or explicitly build on the immediately preceding block. Never assume the reader has run examples from a different section.

### Commenting approximate output on deterministic values

```python
# Bad: cost is deterministic for a given call
print(result.cost_usd)       # ~0.00003

# Good: show the actual value
print(result.cost_usd)       # 3e-06
```

### Skipping the "boring" blocks

Configuration-only blocks (`register_model_defaults`, `set_client`, `clear_conversation`) are often the ones that rot first because nobody tests them. Run them too — a renamed parameter or changed import will break them just as dead as the fancy examples.

## Tutorial Writing

Tutorials are a distinct documentation type from API reference and guides. A tutorial teaches a skill through a hands-on, end-to-end walkthrough. A guide explains how to accomplish a specific task. Both are needed, but tutorials come first in the reader's journey.

### DW-T1: Tutorial structure

Every tutorial follows this structure:

1. **Title and goal** — one sentence saying what the reader will build or learn
2. **Prerequisites** — what the reader needs installed, configured, or understood before starting
3. **Setup** — copy-paste-ready code to get to a working starting point
4. **Steps** — numbered, incremental, each building on the last
5. **Complete listing** — the full working code assembled from all steps
6. **Next steps** — links to related tutorials, guides, or API reference

### DW-T2: Every step must be independently runnable

Each numbered step must leave the reader with working code. If step 3 introduces a bug, the reader should not have to complete step 5 to discover it. Run each step in isolation to verify.

### DW-T3: Show the output, not just the code

Every step must show expected output. Readers use output to verify they're on track. For LLM-generated content, show structural output (types, field names, shapes) and mark variable content as illustrative.

### DW-T4: Progressive complexity

Start with the simplest working version. Add complexity one concept at a time. Do not introduce error handling, configuration, or optimization until the basic version works.

Wrong order: setup client → configure retries → handle errors → make first API call
Right order: make first API call → add configuration → add error handling → add retries

### DW-T5: Use a consistent running example

One dataset, one domain, one scenario throughout the tutorial. Switching contexts mid-tutorial forces the reader to re-orient. Good running examples: a FAQ bot, a document search tool, a code review pipeline.

### DW-T6: Tutorials go in `docs/sphinx/source/tutorials/`

File naming: `docs/sphinx/source/tutorials/<topic>.rst`. Each tutorial is a single RST file. Link from `docs/sphinx/source/index.rst` under a `Tutorials` toctree caption.

### DW-T7: Guide structure

Guides explain how to accomplish a specific task. They assume the reader has completed the quickstart. Structure:

1. **Title** — task-oriented ("Indexing Documents for RAG", not "RAG Indexing API")
2. **Overview** — 2-3 sentences on what this guide covers
3. **Steps** — numbered or headed sections, each focused on one sub-task
4. **Reference** — links to relevant API docs

Guides go in `docs/sphinx/source/guides/<topic>.rst`.

### DW-T8: DRY between tutorials, guides, and README

The README contains a quickstart. Tutorials expand on it. Guides reference tutorials. Do not duplicate full code blocks across all three. Instead:
- README: minimal working example (3-5 lines)
- Quickstart: extended example with explanation
- Tutorial: full end-to-end with setup, steps, and complete listing
- Guide: task-focused snippets with links to tutorials for full context
