#!/usr/bin/env python3
"""Generate AI-consumable API reference documentation from source.

Produces a folder of per-module Markdown files plus a concise index.
Uses inspect/importlib to extract signatures and docstrings directly.
No external documentation tools required.
"""
import importlib
import inspect
import sys
import typing
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
API_DIR = ROOT / "docs" / "api"

HEADER = """\
# FFAI API Reference Index

Auto-generated from source. Read the relevant module file for full docstrings,
args, and returns.

> Do not edit manually. Regenerate with: `scripts/generate_api_docs.py`

"""

PUBLIC_MODULES = [
    "ffai.FFAI",
    "ffai.FFAIClientBase",
    "ffai.ConversationHistory",
    "ffai.OrderedPromptHistory",
    "ffai.config",
    "ffai.retry_utils",
    "ffai.Clients",
    "ffai.Clients.FFMistralSmall",
    "ffai.Clients.AsyncFFLiteLLMClient",
    "ffai.Clients.BaseLiteLLMClient",
    "ffai.Clients.FFLiteLLMClient",
    "ffai.Clients.model_defaults",
    "ffai.core",
    "ffai.core.client_base",
    "ffai.core.structured_output",
    "ffai.core.conversation_manager",
    "ffai.core.response_options",
    "ffai.core.response_result",
    "ffai.core.types",
    "ffai.core.async_client_base",
    "ffai.core.response_executor",
    "ffai.core.async_executor",
    "ffai.core.execution_result",
    "ffai.core.graph_execution_helpers",
    "ffai.core.condition_evaluator",
    "ffai.core.graph",
    "ffai.core.prompt_node",
    "ffai.core.prompt_utils",
    "ffai.core.prompt_builder",
    "ffai.core.execution_state",
    "ffai.core.response_utils",
    "ffai.core.response_context",
    "ffai.core.usage",
    "ffai.core.history",
    "ffai.core.history.recorder",
    "ffai.core.history.ordered",
    "ffai.core.history.conversation",
    "ffai.core.history.permanent",
    "ffai.core.history_exporter",
    "ffai.agent",
    "ffai.agent.response_validator",
    "ffai.agent.agent_loop",
    "ffai.agent.agent_result",
    "ffai.tools",
    "ffai.tools.tool_registry",
    "ffai.observability",
    "ffai.observability.log_context",
    "ffai.observability.telemetry",
    "ffai.rag",
    "ffai.rag.rag",
    "ffai.rag.client_adapter",
    "ffai.rag.types",
    "ffai.rag.prompts",
    "ffai.rag.embed",
    "ffai.rag.store",
    "ffai.rag.format",
    "ffai.rag.splitters",
    "ffai.rag.splitters.base",
    "ffai.rag.splitters.character",
    "ffai.rag.splitters.markdown",
    "ffai.rag.splitters.recursive",
    "ffai.rag.splitters.hierarchical",
    "ffai.rag.splitters.factory",
    "ffai.rag.splitters.code",
    "ffai.rag.search",
    "ffai.rag.search.hybrid",
    "ffai.rag.search.rerankers",
    "ffai.rag.search.query_expansion",
    "ffai.rag.indexing",
    "ffai.rag.indexing.bm25",
    "ffai.rag.indexing.contextual",
    "ffai.rag.indexing.deduplication",
    "ffai.rag.indexing.hierarchical",
]


def _first_sentence(text: str) -> str:
    if not text:
        return ""
    text = text.strip().split("\n")[0]
    if "." in text:
        return text[: text.index(".") + 1]
    return text


def _format_signature(sig: inspect.Signature) -> str:
    params = []
    for p in sig.parameters.values():
        if p.name == "self":
            continue
        s = p.name
        if p.annotation is not inspect.Parameter.empty:
            ann = _format_annotation(p.annotation)
            s += f": {ann}"
        if p.default is not inspect.Parameter.empty:
            default_repr = repr(p.default)
            if len(default_repr) > 60:
                default_repr = "..."
            s += f" = {default_repr}"
        params.append(s)
    return f"({', '.join(params)})"


def _format_annotation(ann) -> str:
    if isinstance(ann, str):
        return ann
    if hasattr(ann, "__origin__"):
        origin = getattr(ann, "__origin__", None)
        args = getattr(ann, "__args__", ())
        origin_name = getattr(origin, "__name__", str(origin))
        if args:
            args_str = ", ".join(_format_annotation(a) for a in args)
            return f"{origin_name}[{args_str}]"
        return origin_name
    if hasattr(ann, "__name__"):
        return ann.__name__
    if ann is type(None):
        return "None"
    return str(ann)


def _render_class(cls, lines: list[str]):
    lines.append(f"### class `{cls.__name__}`\n")
    if cls.__doc__:
        doc = inspect.cleandoc(cls.__doc__)
        lines.append(f"{doc}\n")

    bases = [b.__name__ for b in cls.__bases__ if b.__name__ != "object"]
    if bases:
        lines.append(f"Bases: {', '.join(f'`{b}`' for b in bases)}\n")

    members = inspect.getmembers(cls, predicate=inspect.isfunction)
    public = [(n, m) for n, m in members if not n.startswith("_")]

    for name, method in public:
        _render_method(name, method, lines)

    lines.append("")


def _render_method(name: str, method, lines: list[str]):
    try:
        sig = inspect.signature(method)
    except (ValueError, TypeError):
        lines.append(f"#### `{name}()`\n")
        return

    sig_str = _format_signature(sig)
    lines.append(f"#### `{name}{sig_str}`\n")

    if method.__doc__:
        doc = inspect.cleandoc(method.__doc__)
        lines.append(f"{doc}\n")
    lines.append("")


def _render_function(name: str, func, lines: list[str]):
    try:
        sig = inspect.signature(func)
    except (ValueError, TypeError):
        lines.append(f"### `{name}()`\n")
        return

    sig_str = _format_signature(sig)
    lines.append(f"### `{name}{sig_str}`\n")

    if func.__doc__:
        doc = inspect.cleandoc(func.__doc__)
        lines.append(f"{doc}\n")
    lines.append("")


def _render_dataclass_fields(obj, lines: list[str]):
    if hasattr(obj, "__dataclass_fields__"):
        lines.append("**Fields:**\n")
        for fname, field in obj.__dataclass_fields__.items():
            ann = _format_annotation(field.type) if hasattr(field, "type") else ""
            default = ""
            if field.default is not inspect.Parameter.empty and field.default is not None:
                default = f" = {field.default!r}"
            lines.append(f"- `{fname}`: `{ann}`{default}")
        lines.append("")


def _extract_module_summary(content: str) -> str:
    lines = content.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            return stripped
    return ""


def _is_reexport(obj) -> bool:
    if obj is typing.Any:
        return True
    mod = getattr(obj, "__module__", "")
    return mod.startswith("typing") or mod.startswith("collections.abc")


def generate_module(module_name: str) -> str | None:
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        print(f"  SKIP {module_name}: {e}", file=sys.stderr)
        return None

    lines: list[str] = []
    short_name = module_name.removeprefix("ffai.")
    lines.append(f"# `{short_name}`\n")

    if mod.__doc__:
        lines.append(f"{inspect.cleandoc(mod.__doc__)}\n")

    all_names = getattr(mod, "__all__", None)

    classes = []
    functions = []

    for name in dir(mod):
        if name.startswith("_"):
            continue
        if all_names and name not in all_names:
            continue
        obj = getattr(mod, name, None)
        if obj is None:
            continue
        if _is_reexport(obj):
            continue
        defining = getattr(obj, "__module__", None)
        if defining and defining != module_name and not defining.startswith(
            "ffai." + module_name.removeprefix("ffai.") + "."
        ):
            continue
        if inspect.isclass(obj):
            classes.append((name, obj))
        elif inspect.isfunction(obj) or inspect.isbuiltin(obj):
            functions.append((name, obj))

    classes.sort(key=lambda x: x[0])
    functions.sort(key=lambda x: x[0])

    if classes:
        lines.append("## Classes\n")
        for _name, cls in classes:
            _render_class(cls, lines)

    if functions:
        lines.append("## Functions\n")
        for name, func in functions:
            _render_function(name, func, lines)

    return "".join(lines) if len(lines) > 2 else None


def build_index(entries: list[tuple[str, str]]):
    parts = [HEADER]

    parts.append("## Module Summary\n\n")
    for module_name, summary in entries:
        short = module_name.removeprefix("ffai.")
        filename = f"{short}.md"
        parts.append(f"- **[{short}]({filename})** — {summary}\n")

    class_entries: dict[str, tuple[str, str, str]] = {}
    for module_name, _ in entries:
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue
        all_names = getattr(mod, "__all__", None)
        for name in sorted(dir(mod)):
            if name.startswith("_"):
                continue
            if all_names and name not in all_names:
                continue
            obj = getattr(mod, name, None)
            if obj is None or not inspect.isclass(obj):
                continue
            if _is_reexport(obj):
                continue
            defining_mod = getattr(obj, "__module__", module_name)
            if defining_mod == module_name:
                short = module_name.removeprefix("ffai.")
                filename = f"{short}.md"
                summary = _first_sentence(obj.__doc__ or "")
                if name not in class_entries:
                    class_entries[name] = (filename, short, summary)

    parts.append("\n## Public Classes\n\n")
    for class_name in sorted(class_entries):
        filename, short, summary = class_entries[class_name]
        parts.append(f"- `{class_name}` [{short}]({filename})")
        if summary:
            parts.append(f" — {summary}")
        parts.append("\n")

    (API_DIR / "index.md").write_text("".join(parts))


def generate():
    sys.path.insert(0, str(ROOT))
    API_DIR.mkdir(parents=True, exist_ok=True)

    entries: list[tuple[str, str]] = []

    for module_name in PUBLIC_MODULES:
        print(f"  Generating {module_name}...", file=sys.stderr)
        content = generate_module(module_name)
        if content is None:
            continue
        short = module_name.removeprefix("ffai.")
        (API_DIR / f"{short}.md").write_text(content)
        summary = _first_sentence(_extract_module_summary(content))
        entries.append((module_name, summary))

    build_index(entries)

    total = len(list(API_DIR.glob("*.md")))
    index_kb = (API_DIR / "index.md").stat().st_size // 1024
    total_kb = sum(f.stat().st_size for f in API_DIR.glob("*.md")) // 1024
    print(f"\nGenerated docs/api/: {total} files ({total_kb}KB total, index {index_kb}KB)")


if __name__ == "__main__":
    generate()
