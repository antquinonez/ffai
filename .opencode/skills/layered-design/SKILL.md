---
name: layered-design
description: >
  Use when designing a multi-step feature or refactoring that requires
  upfront architectural planning before implementation. Provides a structured
  approach for writing design documents in an untracked /designs directory,
  reviewing them for consistency, and then implementing layer by layer.
  Triggers when the user asks to "design", "plan", or "write designs" for
  a feature, or when a task involves porting code between projects.
license: MIT
---

# Layered Design Process

A structured approach for designing features before implementing them.
Produces design documents in an untracked `designs/` directory, reviews
them for correctness, then implements layer by layer.

## When to Use

- Porting code between projects
- Multi-file feature additions
- Architectural changes affecting multiple modules
- Any task where "measure twice, cut once" applies

## Phase 1: Research

Before writing any design:

1. **Read the source** -- understand the code being ported or changed.
   Read ALL relevant files, not just headers. Use the Task tool for large
   codebases.
2. **Read the target** -- understand where the code will land. Read existing
   modules, their `__init__.py` exports, their type definitions, their tests.
3. **Map dependencies** -- trace import chains. Identify what depends on what.
   Document the dependency graph.
4. **Identify coupling** -- what does the source code depend on that the target
   doesn't have? What needs to change?

## Phase 2: Write Designs

Create an untracked `designs/` directory. Add it to `.gitignore`.

### File structure

```
designs/
  00-overview-<feature-slug>.md  # Architecture, layer summary, dependency order
  01-L1-<name>.md                # Layer 1 design
  02-L2-<name>.md                # Layer 2 design
  ...
  NN-LN-<name>.md                # Layer N design
```

Use a `<feature-slug>` suffix to avoid collisions when the `designs/`
directory already has other work. For example: `00-overview-litellm-generate.md`.

### Overview document (00-overview-<feature-slug>.md)

Must include:

- **Objective** -- what the feature does in 1-2 sentences
- **Source material** -- for ports: what code is being ported, with file paths
  and line counts. For new features: reference patterns, prior art, issue
  documents, or RFC sketches that inform the design.
- **Layer summary** -- table with layer number, what it adds, files
  created/modified, backward-compatibility note (e.g. "Yes — new exports only")
- **Dependency order** -- explicit ordering with dependency edges. If L2
  depends on L3, say so and split L2 into sub-layers
- **Key design principles** -- 3-5 rules governing the design
- **Adaptation notes** -- for ports: what changes from source to target.
  For new features: key decisions vs. any prior sketch or issue description.
- **Exports strategy** -- what `__init__.py` files need updating per layer
- **Testing strategy** -- how each layer is tested

### Layer documents (01-L1-..., 02-L2-..., etc.)

Each layer document must include:

- **Goal** -- one sentence
- **Source** -- for ports: what file(s) are being ported, with file paths and
  line counts. For new features: reference patterns or existing modules that
  set the convention.
- **Destination** -- target file paths
- **Changes from source** -- for ports: every adaptation from the original,
  with rationale. For new features: key design decisions vs. any prior sketch
  or issue description, in a table with rationale.
- **Public API** -- code examples showing how the new module is used
- **Integration points** -- how this layer connects to existing code (or future
  layers)
- **Files created** -- table with file path, estimated lines, action
- **Files modified** -- table with file path and specific changes
- **Acceptance criteria** -- numbered, testable, specific

### Writing rules

1. **Modifications must be safe.** Any layer may modify existing files as
   long as the change is backward-compatible: new classes with defaults,
   new config fields, new exports, additive docstring changes. Refactoring
   or breaking changes to existing code are restricted to LN only. The test
   for safety: "does every existing test still pass after this layer?"
2. **Port, don't wrap.** Copy code with minimal adaptation rather than
   creating adapter layers.
3. **Show actual code.** Every design must include the actual function
   signatures, dataclass definitions, and import paths that will be used.
   Not pseudocode.
4. **Specify import paths exactly.** Write `from ffai.core.graph import ...`
   not `import the graph module`.
5. **Address `__init__.py`** in each layer.
6. **Co-locate prerequisite changes.** If a layer depends on a config
   class or helper function, add those changes in the same layer document
   under "Files modified" — do not defer them to a later layer. The
   overview's dependency order must reflect these intra-layer
   prerequisites.
7. **Split layers at verifiability boundaries.** A layer should be the
   smallest unit that can be implemented and verified independently: its
   acceptance criteria must be runnable without implementing a later layer.
   If two concerns share the same acceptance criteria, merge them. If one
   can be tested before the other exists, they can be separate layers.

## Phase 3: Review

After writing all designs, run a consistency review:

Use the Task tool to launch an explore agent that reads ALL design documents
and ALL referenced source files. Check for:

1. **Import path correctness** -- do proposed imports work given directory
   structure?
2. **API consistency** -- do designs reference each other's APIs correctly?
3. **Backward compatibility** -- will changes break existing behavior?
4. **Line number references** -- are cited line numbers still accurate?
5. **Missing pieces** -- anything needed for implementation not covered?
6. **Contradictions** -- do designs contradict each other?
7. **`__init__.py` changes** -- are new modules properly exported?
8. **Undefined references** -- are all function/class names used in code
   snippets actually defined somewhere?
9. **Test infrastructure propagation** -- if a layer introduces test-only
   mechanisms (recorders, spies, fixtures, ContextVars), verify that
   downstream layers can access them through nested call chains without
   adding test parameters to production APIs. If `L1.adapter_span()`
   accepts `_recorder` but `L3.ResilientCaller.call()` also calls
   `adapter_span()` internally, the recorder must propagate via context
   (e.g., `ContextVar`), not via parameter threading. Check every layer
   that calls a lower layer's instrumented functions.

### Issue severity levels

| Level | Meaning |
|-------|---------|
| HIGH | Will cause runtime error or incorrect behavior |
| MEDIUM | Will cause import failure, type error, or test failure |
| LOW | Missing code snippets, implicit knowledge not written down,
         style inconsistency, documentation gap |

### Fix all issues before proceeding to implementation.

LOW issues are often cheaper to fix in the design than to discover
during implementation. A missing import path or an underspecified
config change that seems obvious now won't be obvious to the
implementer — or to you in three days.

After fixing, re-verify that changes to one layer document don't
invalidate references in others. This is a targeted sanity check
on cross-document references, not a full re-review.

## Phase 4: Implement

Implement layers in dependency order. For each layer:

1. Read the design document
2. Implement the changes described
3. Run `ruff check` and `pyright` on changed files
4. Write/run tests for the layer
5. Verify acceptance criteria from the design
6. Only then move to the next layer

### Implementation checklist (per layer)

- [ ] Files created match design
- [ ] Files modified match design
- [ ] `__init__.py` exports updated
- [ ] `ruff check` passes
- [ ] `pyright` passes
- [ ] Tests written and passing
- [ ] Acceptance criteria verified

### After all layers

- Run full test suite: `pytest`
- Run full lint: `ruff check .`
- Run full typecheck: `pyright`
- Verify no regressions in existing tests
