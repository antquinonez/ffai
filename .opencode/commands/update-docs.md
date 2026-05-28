---
description: Regenerate all documentation (API reference + Sphinx HTML/text/JSON)
---

Regenerate both the AI-consumable API reference and the Sphinx documentation.

1. Run `scripts/generate_api_docs.py` using the project venv Python (`.venv/bin/python`).
2. Run `scripts/generate_sphinx_docs.py` using the project venv Python (`.venv/bin/python`).
   This generates .rst stubs, then builds all three Sphinx output formats:
   - **html**: full rendered site for humans
   - **text**: plaintext per page for AI context windows
   - **json**: structured JSON per page (toctree, sections, cross-refs)
3. Report the output (number of modules, file sizes, build status).
4. Do not commit the generated files.

Both scripts share the same `PUBLIC_MODULES` list (defined in `generate_api_docs.py`).
When adding a new module, add it to `PUBLIC_MODULES` in `scripts/generate_api_docs.py`
and both doc sets will pick it up on the next regeneration.

If a script fails, investigate and fix the issue. Common causes:
- Import errors: check that the module is importable from the project root
- Missing modules: check that the module name in `PUBLIC_MODULES` list matches an actual file
- Sphinx warnings: check for duplicate object descriptions or missing imports
