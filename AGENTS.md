# AGENTS.md

Instructions for AI agents working on this codebase.

## Commands

- **Lint:** `ruff check src/ tests/ examples/`
- **Type check:** `pyright src/ tests/`
- **Tests:** `pytest tests/ -x -q`

Run all three after making changes. Do not skip any.

## Error Policy

**All `ruff` and `pyright` errors must be fixed when encountered, including pre-existing errors.** Never leave a known error for later.

The only acceptable suppressions:
- `# type: ignore[...]` with a specific error code when the runtime behavior is intentionally different from the type system (e.g., passing deliberately invalid data to test error handling)
- `# noqa: E402` in Jupyter notebook cells where imports must follow `sys.path` setup

## Code Style

- No comments unless explicitly requested
- Follow existing patterns in neighboring files
- Check `pyproject.toml` for configured tool settings (ruff rules, pyright config)

## Design Documents

Design documents live in `designs/` (untracked). They follow a layered approach:
- L1-L3 add new files only
- L4 modifies existing files

## Testing

- New features require new tests
- All tests must pass with zero regressions before any change is considered complete
- The test suite should remain at or above its current count (589+ tests)
