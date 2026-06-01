from __future__ import annotations

import json
import re
from typing import Any

from .loader import WorkflowValidationError, _validate_spec
from .spec import ClientRef, PromptStep, WorkflowDefaults, WorkflowSpec


class TabularLoadError(ValueError):
    pass


_HEADER_ALIASES: dict[str, str] = {
    "step": "name",
    "step_name": "name",
    "step name": "name",
    "prompt_text": "prompt",
    "prompt text": "prompt",
    "question": "prompt",
    "depends_on": "history",
    "depends on": "history",
    "dependencies": "history",
    "deps": "history",
    "client_name": "client",
    "client name": "client",
    "model_name": "model",
    "model name": "model",
    "temp": "temperature",
    "max tokens": "max_tokens",
    "maxtokens": "max_tokens",
    "token_limit": "max_tokens",
    "system instructions": "system_instructions",
    "system": "system_instructions",
    "instructions": "system_instructions",
    "cond": "condition",
    "skip_unless": "condition",
    "abort condition": "abort_condition",
    "abort": "abort_condition",
    "response format": "response_format",
    "format": "response_format",
    "response model": "response_model",
    "tool_names": "tools",
    "tool names": "tools",
    "tool_choice": "tool_choice",
    "tool choice": "tool_choice",
}

_LIST_FIELDS = frozenset({"history", "tools"})
_NUMERIC_FIELDS: dict[str, type] = {"temperature": float, "max_tokens": int}
_BOOL_FIELDS = frozenset({"strict"})
_BOOL_TRUE_VALUES = frozenset({"true", "yes", "1"})


def _normalize_header(header: str) -> str:
    return re.sub(r"[\s-]+", "_", header.strip()).strip("_").lower()


def _canonical_column(header: str) -> str:
    normalized = _normalize_header(header)
    return _HEADER_ALIASES.get(normalized, normalized)


def _parse_list_field(value: Any) -> list[str] | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return [item.strip() for item in s.split(",") if item.strip()]


def _coerce_value(key: str, value: Any) -> Any:
    if value is None:
        return None

    if key in _LIST_FIELDS:
        return _parse_list_field(value)

    if key in _NUMERIC_FIELDS:
        s = str(value).strip()
        if not s:
            return None
        try:
            return _NUMERIC_FIELDS[key](s)
        except (ValueError, TypeError):
            return None

    if key in _BOOL_FIELDS:
        s = str(value).strip().lower()
        return s in _BOOL_TRUE_VALUES

    if key == "response_format":
        s = str(value).strip()
        if not s:
            return None
        if s.startswith("{"):
            try:
                return json.loads(s)
            except json.JSONDecodeError:
                return s
        return s

    if isinstance(value, str) and not value.strip():
        return None

    return value


def _parse_client_ref(value: Any) -> ClientRef | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return ClientRef.from_dict(value)
    s = str(value).strip()
    if not s:
        return None
    return ClientRef.from_dict(s)


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for header, value in row.items():
        canonical = _canonical_column(header)
        normalized[canonical] = _coerce_value(canonical, value)
    return normalized


def _rows_to_steps(rows: list[dict[str, Any]]) -> list[PromptStep]:
    steps: list[PromptStep] = []
    for i, raw_row in enumerate(rows):
        row = _normalize_row(raw_row)

        name = row.get("name")
        prompt = row.get("prompt")
        if not name:
            raise TabularLoadError(f"Row {i} is missing required field 'name'")
        if not prompt:
            raise TabularLoadError(
                f"Row {i} ('{name}') is missing required field 'prompt'"
            )

        client = _parse_client_ref(row.get("client"))

        steps.append(
            PromptStep(
                name=name,
                prompt=prompt,
                client=client,
                model=row.get("model"),
                history=row.get("history"),
                condition=row.get("condition"),
                abort_condition=row.get("abort_condition"),
                system_instructions=row.get("system_instructions"),
                response_format=row.get("response_format"),
                response_model=row.get("response_model"),
                strict=row.get("strict", False),
                tools=row.get("tools"),
                tool_choice=row.get("tool_choice"),
                max_tokens=row.get("max_tokens"),
                temperature=row.get("temperature"),
            )
        )
    return steps


def load_workflow_rows(
    rows: list[dict[str, Any]],
    *,
    name: str = "unnamed",
    description: str = "",
    defaults: dict[str, Any] | None = None,
    clients: dict[str, dict[str, Any] | str] | None = None,
    tools: dict[str, dict[str, Any]] | None = None,
) -> WorkflowSpec:
    if not rows:
        raise TabularLoadError("No rows provided")

    steps = _rows_to_steps(rows)

    defaults_obj = WorkflowDefaults()
    if defaults:
        defaults_obj = WorkflowDefaults.from_dict(defaults)

    clients_map: dict[str, ClientRef] = {}
    if clients:
        for cname, cval in clients.items():
            clients_map[cname] = ClientRef.from_dict(cval)

    tools_map: dict[str, dict[str, Any]] = {}
    if tools:
        for tname, tval in tools.items():
            entry = dict(tval)
            entry["name"] = tname
            tools_map[tname] = entry

    spec = WorkflowSpec(
        name=name,
        description=description,
        defaults=defaults_obj,
        clients=clients_map,
        tools=tools_map,
        prompts=steps,
    )

    try:
        _validate_spec(spec)
    except WorkflowValidationError as e:
        raise TabularLoadError(str(e)) from e

    return spec
