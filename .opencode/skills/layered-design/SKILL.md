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
  00-overview.md          # Architecture, layer summary, dependency order
  01-L1-<name>.md         # Layer 1 design
  02-L2-<name>.md         # Layer 2 design
  ...
  NN-LN-<name>.md         # Layer N design
```

### Overview document (00-overview.md)

Must include:

- **Objective** -- what the feature does in 1-2 sentences
- **Source material** -- what code is being ported or adapted, with line counts
- **Layer summary** -- table with layer number, what it adds, files
  created/modified, whether it ships independently
- **Dependency order** -- explicit ordering with dependency edges. If L2
  depends on L3, say so and split L2 into sub-layers
- **Key design principles** -- 3-5 rules governing the design
- **Adaptation notes** -- what changes from source to target
- **Exports strategy** -- what `__init__.py` files need updating per layer
- **Testing strategy** -- how each layer is tested

### Layer documents (01-L1-..., 02-L2-..., etc.)

Each layer document must include:

- **Goal** -- one sentence
- **Source** -- what file(s) are being ported or modified, with line counts
- **Destination** -- target file paths
- **Changes from source** -- every adaptation from the original, with rationale
- **Public API** -- code examples showing how the new module is used
- **Integration points** -- how this layer connects to existing code (or future
  layers)
- **Files created** -- table with file path, estimated lines, action
- **Files modified** -- table with file path and specific changes
- **Acceptance criteria** -- numbered, testable, specific

### Writing rules

1. **Layers are additive.** L1-L(N-1) add new files only. Only LN modifies
   existing files. This lets early layers ship independently.
2. **Port, don't wrap.** Copy code with minimal adaptation rather than
   creating adapter layers.
3. **Show actual code.** Every design must include the actual function
   signatures, dataclass definitions, and import paths that will be used.
   Not pseudocode.
4. **Specify import paths exactly.** Write `from src.core.graph import ...`
   not `import the graph module`.
5. **Address `__init__.py`** in each layer.

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

### Issue severity levels

| Level | Meaning |
|-------|---------|
| HIGH | Will cause runtime error or incorrect behavior |
| MEDIUM | Will cause import failure, type error, or test failure |
| LOW | Style inconsistency, documentation gap |

### Fix all HIGH and MEDIUM issues before proceeding to implementation.

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
