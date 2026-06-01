from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .spec import WorkflowSpec


class WorkflowValidationError(ValueError):
    pass


def load_workflow(yaml_text: str) -> WorkflowSpec:
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        raise WorkflowValidationError("Workflow YAML must be a mapping")
    return _parse_and_validate(data)


def load_workflow_file(path: str | Path) -> WorkflowSpec:
    filepath = Path(path)
    if not filepath.exists():
        raise FileNotFoundError(f"Workflow file not found: {filepath}")
    return load_workflow(filepath.read_text(encoding="utf-8"))


def _parse_and_validate(data: dict[str, Any]) -> WorkflowSpec:
    workflow = data.get("workflow", data)

    if "prompts" not in workflow and "prompts" not in data:
        raise WorkflowValidationError("Workflow must contain a 'prompts' list")

    spec = WorkflowSpec.from_dict(data)
    _validate_spec(spec)
    return spec


def _validate_spec(spec: WorkflowSpec) -> None:
    errors: list[str] = []

    if not spec.prompts:
        raise WorkflowValidationError("Workflow must contain at least one prompt")

    prompt_names = {p.name for p in spec.prompts}

    seen: set[str] = set()
    for p in spec.prompts:
        if p.name in seen:
            errors.append(f"Duplicate prompt name: '{p.name}'")
        seen.add(p.name)

    for p in spec.prompts:
        for ref in p.history or []:
            if ref not in prompt_names:
                errors.append(
                    f"Prompt '{p.name}' references unknown prompt '{ref}' in history"
                )
            if ref == p.name:
                errors.append(f"Prompt '{p.name}' references itself in history")

    defined_tools = set(spec.tools.keys())
    for p in spec.prompts:
        for tool_name in p.tools or []:
            if tool_name not in defined_tools:
                errors.append(
                    f"Prompt '{p.name}' references undefined tool '{tool_name}'"
                )

    if errors:
        raise WorkflowValidationError(
            f"Workflow validation failed ({len(errors)} error(s)): "
            + "; ".join(errors)
        )
