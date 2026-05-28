#!/usr/bin/env python3
"""Add docstrings to Python source files using AST-based insertion.

Uses AST to find exact insertion points, avoiding the pitfalls of
text-based editing (eating body lines, wrong indentation, etc.).

Supports three target types:
    module          Top-level module docstring.
    ClassName       Class-level docstring.
    ClassName.method  Method/function docstring.

After writing, re-parses the file with ``ast.parse`` to verify the result
is valid Python.  If the parse fails, the original file is restored and
the edit is reported as an error.

Usage:
    # Single-line docstrings via --map
    .venv/bin/python scripts/add_docstrings.py \\
        --map "config.py:module=Configuration loading and environment parsing." \\
        --map "config.py:Settings=Application settings loaded from env vars." \\
        --dry-run

    # Multi-line docstrings via --map-file (JSON)
    .venv/bin/python scripts/add_docstrings.py --map-file docstrings.json --dry-run

    # Audit docstring coverage
    .venv/bin/python scripts/add_docstrings.py --audit
    .venv/bin/python scripts/add_docstrings.py --audit --audit-missing

    # Or call from another script with the apply_docstrings() API
    .venv/bin/python -c "
        from add_docstrings import apply_docstrings
        apply_docstrings({
            ('config.py', None, 'Settings'): 'Application settings.',
        })
    "
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import dataclass
from pathlib import Path

SRC_DIR = os.path.join(os.path.dirname(__file__), "..", "src")

TRIVIAL_METHODS = frozenset({"to_dict", "from_dict", "__repr__", "__len__", "__str__", "__eq__", "__hash__"})


@dataclass
class DocstringEdit:
    filepath: str
    cls_name: str | None
    target_name: str
    docstring: str


def _find_module_node(tree: ast.Module) -> ast.Module | None:
    if ast.get_docstring(tree) is None:
        return tree
    return None


def _find_class_node(tree: ast.Module, cls_name: str) -> ast.ClassDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            if ast.get_docstring(node) is None:
                return node
            break
    return None


def _find_func_node(
    tree: ast.Module, cls_name: str, method_name: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name == method_name
                    and not ast.get_docstring(item)
                ):
                    return item
            break
    return None


def _insertion_point(node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef, lines: list[str]) -> tuple[int, int]:
    first_stmt = node.body[0]
    if isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Constant) and isinstance(first_stmt.value.value, str):
        if len(node.body) > 1:
            second_stmt = node.body[1]
            line_idx = second_stmt.lineno - 1
        else:
            line_idx = first_stmt.end_lineno or first_stmt.lineno
    else:
        line_idx = first_stmt.lineno - 1
    indent = len(lines[line_idx]) - len(lines[line_idx].lstrip())
    return line_idx, indent


def _parse_target(fname: str, class_method: str) -> DocstringEdit:
    if class_method == "module":
        return DocstringEdit(
            filepath=os.path.join(SRC_DIR, fname),
            cls_name=None,
            target_name="module",
            docstring="",
        )

    if "." in class_method:
        cls_name, method_name = class_method.split(".", 1)
        return DocstringEdit(
            filepath=os.path.join(SRC_DIR, fname),
            cls_name=cls_name,
            target_name=method_name,
            docstring="",
        )

    return DocstringEdit(
        filepath=os.path.join(SRC_DIR, fname),
        cls_name=None,
        target_name=class_method,
        docstring="",
    )


def _parse_map_entry(entry: str) -> DocstringEdit:
    parts = entry.split("=", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid --map entry (need file.py:Target=Docstring text): {entry}")
    target, docstring = parts
    if ":" not in target:
        raise ValueError(f"Invalid target (need file.py:Target): {target}")
    fname, class_method = target.split(":", 1)
    edit = _parse_target(fname, class_method)
    edit.docstring = docstring.strip()
    return edit


def _load_map_file(path: str) -> dict[TargetKey, str]:
    with open(path) as f:
        raw = json.load(f)

    result: dict[TargetKey, str] = {}
    for target_key, docstring in raw.items():
        if ":" not in target_key:
            raise ValueError(f"Invalid target in map file (need file.py:Target): {target_key}")
        fname, class_method = target_key.split(":", 1)
        edit = _parse_target(fname, class_method)
        rel = os.path.relpath(edit.filepath, os.path.normpath(SRC_DIR))
        result[(rel, edit.cls_name, edit.target_name)] = docstring

    return result


TargetKey = tuple[str, str | None, str]


def _resolve_node(
    tree: ast.Module, cls_name: str | None, target_name: str, lines: list[str]
) -> tuple[int, int] | None:
    if target_name == "module" and cls_name is None:
        node = _find_module_node(tree)
        if node is None:
            return None
        return _insertion_point(node, lines)

    if cls_name is None:
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == target_name:
                if ast.get_docstring(node) is not None:
                    return None
                return _insertion_point(node, lines)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target_name:
                if ast.get_docstring(node) is not None:
                    return None
                return _insertion_point(node, lines)
        return None

    node = _find_func_node(tree, cls_name, target_name)
    if node is None:
        return None
    return _insertion_point(node, lines)


def apply_docstrings(
    docstring_map: dict[TargetKey, str],
    src_dir: str | None = None,
    *,
    dry_run: bool = False,
) -> int:
    """Apply docstrings from a (file, class, target) -> docstring mapping.

    After writing each file, re-parses it with ``ast.parse`` to verify
    syntactic validity.  If the parse fails, the original content is
    restored and the edit is reported as an error.

    Returns the number of docstrings inserted.
    """
    base = src_dir or SRC_DIR
    file_edits: dict[str, list[tuple[int, int, str]]] = {}

    for (fname, cls_name, target_name), docstring in docstring_map.items():
        filepath = os.path.join(base, fname)
        if filepath not in file_edits:
            file_edits[filepath] = []

        with open(filepath) as f:
            source = f.read()

        tree = ast.parse(source)
        lines = source.split("\n")
        result = _resolve_node(tree, cls_name, target_name, lines)
        if result is None:
            label = f"{cls_name}.{target_name}" if cls_name else target_name
            print(f"  SKIP {fname}:{label} (already has docstring or not found)")
            continue

        line_idx, indent_spaces = result
        file_edits[filepath].append((line_idx, indent_spaces, docstring))

    total = 0
    for filepath, edits in file_edits.items():
        if not edits:
            continue

        with open(filepath) as f:
            original_content = f.read()
        lines = original_content.split("\n")

        edits.sort(key=lambda x: x[0], reverse=True)

        for insert_before, indent_spaces, docstring in edits:
            indent = " " * indent_spaces
            doc_line = f'{indent}"""{docstring}"""'
            lines.insert(insert_before, doc_line)
            total += 1

        new_content = "\n".join(lines)

        if not dry_run:
            try:
                ast.parse(new_content)
            except SyntaxError as exc:
                print(f"  ERROR {filepath}: insert broke syntax ({exc}) — reverting")
                total -= len(edits)
                new_content = original_content
            with open(filepath, "w") as f:
                f.write(new_content)
        else:
            try:
                ast.parse(new_content)
            except SyntaxError:
                print(f"  ERROR {filepath}: insert would break syntax")
                total -= len(edits)

        status = "would update" if dry_run else "updated"
        print(f"  {status.capitalize()} {filepath} ({len(edits)} docstrings)")

    return total


def _is_property_or_setter(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    for d in node.decorator_list:
        if isinstance(d, ast.Name) and d.id == "property":
            return True
        if isinstance(d, ast.Attribute) and d.attr == "setter":
            return True
    return False


def audit(src_dir: str | None = None, *, missing_only: bool = False) -> None:
    """Print a docstring coverage report for the source tree."""
    base = src_dir or SRC_DIR
    total_targets = 0
    documented_targets = 0
    total_missing = 0

    for root, dirs, files in sorted(os.walk(base)):
        dirs[:] = sorted(d for d in dirs if d not in ("__pycache__",))
        for f in sorted(files):
            if not f.endswith(".py"):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, base)
            source = Path(path).read_text()
            tree = ast.parse(source)
            file_has_issues = False

            is_init = f == "__init__.py"

            if not is_init:
                mod_doc = ast.get_docstring(tree) is not None
                total_targets += 1
                if mod_doc:
                    documented_targets += 1
                else:
                    total_missing += 1
                    if not missing_only:
                        if not file_has_issues:
                            print(f"{rel}:")
                            file_has_issues = True
                        print("  - module docstring MISSING")

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if node.name.startswith("_"):
                        continue

                    cls_doc = ast.get_docstring(node) is not None
                    total_targets += 1
                    if cls_doc:
                        documented_targets += 1
                    else:
                        total_missing += 1
                        if not file_has_issues:
                            print(f"{rel}:")
                            file_has_issues = True
                        print(f"  - class {node.name}: MISSING")

                    for item in node.body:
                        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            continue
                        if item.name.startswith("_") and not item.name.startswith("__"):
                            continue
                        if item.name in TRIVIAL_METHODS:
                            continue
                        if _is_property_or_setter(item):
                            continue

                        meth_doc = ast.get_docstring(item) is not None
                        total_targets += 1
                        if meth_doc:
                            documented_targets += 1
                        else:
                            total_missing += 1
                            if not file_has_issues:
                                print(f"{rel}:")
                                file_has_issues = True
                            print(f"  - {node.name}.{item.name}()")

                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.col_offset != 0 or node.name.startswith("_"):
                        continue

                    func_doc = ast.get_docstring(node) is not None
                    total_targets += 1
                    if func_doc:
                        documented_targets += 1
                    else:
                        total_missing += 1
                        if not file_has_issues:
                            print(f"{rel}:")
                            file_has_issues = True
                        print(f"  - func {node.name}()")

    if total_targets == 0:
        print("No targets found.")
        return

    pct = documented_targets / total_targets * 100
    print(f"\n  {documented_targets}/{total_targets} documented ({pct:.1f}%), {total_missing} missing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Add docstrings to Python source files")
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="file.py:Target=Docstring text",
        help="One single-line docstring entry; repeat for multiple",
    )
    parser.add_argument(
        "--map-file",
        default=None,
        metavar="PATH",
        help="JSON file with multi-line docstrings (keys: file.py:Target, values: docstring text)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing files",
    )
    parser.add_argument(
        "--src-dir",
        default=None,
        help="Override source directory (default: src/)",
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help="Print docstring coverage report",
    )
    parser.add_argument(
        "--audit-missing",
        action="store_true",
        help="Only show undocumented targets in audit",
    )
    args = parser.parse_args()

    if args.audit:
        audit(args.src_dir, missing_only=args.audit_missing)
        return

    if not args.map and not args.map_file:
        parser.error("At least one --map entry or --map-file is required (or use --audit)")

    docstring_map: dict[TargetKey, str] = {}

    for entry in args.map:
        edit = _parse_map_entry(entry)
        fname = os.path.relpath(edit.filepath, os.path.normpath(SRC_DIR))
        docstring_map[(fname, edit.cls_name, edit.target_name)] = edit.docstring

    if args.map_file:
        file_entries = _load_map_file(args.map_file)
        docstring_map.update(file_entries)

    total = apply_docstrings(docstring_map, src_dir=args.src_dir, dry_run=args.dry_run)
    print(f"\n{total} docstring(s) {'would be ' if args.dry_run else ''}inserted.")


if __name__ == "__main__":
    main()
