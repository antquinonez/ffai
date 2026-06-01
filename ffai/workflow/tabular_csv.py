from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .tabular import TabularLoadError, load_workflow_rows


@dataclass
class CommentMetadata:
    meta: dict[str, Any]
    defaults: dict[str, Any]
    clients: dict[str, Any]
    tools: dict[str, Any]
    remaining_lines: list[str]


def _parse_comment_metadata(lines: list[str]) -> CommentMetadata:
    meta: dict[str, Any] = {}
    defaults: dict[str, Any] = {}
    clients: dict[str, Any] = {}
    tools: dict[str, Any] = {}
    remaining: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("#"):
            remaining.append(line)
            continue

        content = stripped.lstrip("#").strip()
        if ":" not in content:
            remaining.append(line)
            continue

        key, _, value = content.partition(":")
        key = key.strip()
        value = value.strip()

        if key == "workflow":
            meta["name"] = value
        elif key == "description":
            meta["description"] = value
        elif key.startswith("default_"):
            field_name = key[len("default_"):]
            _add_typed_value(defaults, field_name, value)
        elif key.startswith("client."):
            client_name = key[len("client."):]
            try:
                clients[client_name] = json.loads(value)
            except json.JSONDecodeError:
                clients[client_name] = value
        elif key.startswith("tool."):
            tool_name = key[len("tool."):]
            try:
                tools[tool_name] = json.loads(value)
            except json.JSONDecodeError:
                tools[tool_name] = {"description": value}

    return CommentMetadata(
        meta=meta,
        defaults=defaults,
        clients=clients,
        tools=tools,
        remaining_lines=remaining,
    )


def _add_typed_value(target: dict[str, Any], key: str, value: str) -> None:
    if key in ("temperature",):
        try:
            target[key] = float(value)
        except ValueError:
            target[key] = value
    elif key in ("max_tokens", "max_concurrency"):
        try:
            target[key] = int(value)
        except ValueError:
            target[key] = value
    elif key in ("strict",):
        target[key] = value.lower() in ("true", "yes", "1")
    else:
        target[key] = value


def _csv_text_to_rows(csv_text: str, delimiter: str = ",") -> list[dict[str, str]]:
    reader = csv.DictReader(io.StringIO(csv_text), delimiter=delimiter)
    rows: list[dict[str, str]] = []
    for row in reader:
        cleaned = {k: v for k, v in row.items() if k is not None}
        rows.append(cleaned)
    return rows


def _merge_metadata(
    comment: CommentMetadata,
    name: str,
    description: str,
    defaults: dict[str, Any] | None,
    clients: dict[str, dict[str, Any] | str] | None,
    tools: dict[str, dict[str, Any]] | None,
) -> tuple[str, str, dict[str, Any], dict[str, Any], dict[str, Any]]:
    merged_name = name if name != "unnamed" else comment.meta.get("name", name)
    merged_desc = description if description else comment.meta.get("description", "")
    merged_defaults = {**comment.defaults, **(defaults or {})}
    merged_clients = {**comment.clients, **(clients or {})}
    merged_tools = {**comment.tools, **(tools or {})}
    return merged_name, merged_desc, merged_defaults, merged_clients, merged_tools


def load_workflow_csv(
    csv_text: str,
    *,
    name: str = "unnamed",
    description: str = "",
    defaults: dict[str, Any] | None = None,
    clients: dict[str, dict[str, Any] | str] | None = None,
    tools: dict[str, dict[str, Any]] | None = None,
    delimiter: str = ",",
) -> Any:
    lines = csv_text.splitlines()
    comment = _parse_comment_metadata(lines)

    m_name, m_desc, m_defaults, m_clients, m_tools = _merge_metadata(
        comment, name, description, defaults, clients, tools
    )

    clean_csv = "\n".join(comment.remaining_lines)
    rows = _csv_text_to_rows(clean_csv, delimiter)

    if not rows:
        raise TabularLoadError("CSV contains no data rows")

    return load_workflow_rows(
        rows,
        name=m_name,
        description=m_desc,
        defaults=m_defaults,
        clients=m_clients,
        tools=m_tools,
    )


def load_workflow_csv_file(
    path: str | Path,
    *,
    name: str = "unnamed",
    description: str = "",
    defaults: dict[str, Any] | None = None,
    clients: dict[str, dict[str, Any] | str] | None = None,
    tools: dict[str, dict[str, Any]] | None = None,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> Any:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV file not found: {p}")

    csv_text = p.read_text(encoding=encoding)
    return load_workflow_csv(
        csv_text,
        name=name,
        description=description,
        defaults=defaults,
        clients=clients,
        tools=tools,
        delimiter=delimiter,
    )
