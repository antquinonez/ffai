# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Pure helper functions for prompt resolution, abort checking, and failure
propagation during graph (DAG) execution.

These functions operate on ``results_by_name`` (the executor's accumulated
result dict) rather than the shared ``prompt_attr_history``, making them
suitable for level-by-level graph execution where results are only written
to the shared history after the entire graph completes.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .graph import evaluate_condition_with_trace
from .prompt_node import PromptNode
from .prompt_utils import interpolate_prompt

logger = logging.getLogger(__name__)

REFERENCES_PATTERN = re.compile(r"<REFERENCES>.*?</REFERENCES>\s*", re.DOTALL)


def build_graph_history_dict(
    results_by_name: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Build a ``{name: response_text}`` mapping from executor results.

    Args:
        results_by_name: Dict mapping prompt_name to result dicts with
            ``response``, ``status``, etc.

    Returns:
        Dict mapping prompt_name to response text (str).

    """
    history_dict: dict[str, str] = {}
    for name, result in results_by_name.items():
        response = result.get("response", "")
        if isinstance(response, dict):
            response = json.dumps(response)
        elif response is not None:
            response = str(response)
        else:
            response = ""
        history_dict[name] = response
    return history_dict


def resolve_graph_prompt(
    prompt_spec: dict[str, Any],
    results_by_name: dict[str, dict[str, Any]],
) -> tuple[str, set[str]]:
    """Resolve interpolation and history injection for a graph node.

    Performs two-phase prompt assembly:

    1. **Interpolation**: Resolves ``{{name.response}}`` patterns using
       results from earlier DAG levels.
    2. **History injection**: For each name in ``prompt_spec["history"]``,
       formats the prior Q&A as ``<conversation_history>`` XML. Entries
       already interpolated are deduplicated.

    Args:
        prompt_spec: The prompt dict with keys ``prompt``, ``history``, etc.
        results_by_name: Accumulated results from prior DAG levels.

    Returns:
        Tuple of (resolved_prompt, set_of_interpolated_names).

    """
    prompt = prompt_spec.get("prompt", "")
    history_names: list[str] = prompt_spec.get("history") or []

    history_dict = build_graph_history_dict(results_by_name)

    resolved_prompt, interpolated_names = interpolate_prompt(
        prompt, history_dict, strict=False
    )

    history_entries: list[dict[str, Any]] = []
    for name in history_names:
        result = results_by_name.get(name)
        if result is None:
            logger.warning(f"No result found for history reference: {name}")
            continue

        response = result.get("response", "")
        stored_prompt = result.get("prompt", name)

        if isinstance(stored_prompt, str) and "{{" in stored_prompt:
            stored_prompt, _ = interpolate_prompt(
                stored_prompt, history_dict, strict=False
            )

        stored_prompt = REFERENCES_PATTERN.sub("", stored_prompt).strip()

        history_entries.append(
            {"prompt_name": name, "prompt": stored_prompt, "response": response}
        )

    filtered_history = [
        entry
        for entry in history_entries
        if entry.get("prompt_name") not in interpolated_names
    ]

    formatted_history: list[str] = []
    for entry in filtered_history:
        formatted_entry = (
            f"<interaction prompt_name='{entry['prompt_name']}'>\n"
            f"USER: {entry['prompt']}\n"
            f"SYSTEM: {entry['response']}\n"
            f"</interaction>"
        )
        formatted_history.append(formatted_entry)

    if formatted_history:
        final_prompt = (
            "<conversation_history>\n"
            + "\n".join(formatted_history)
            + "\n</conversation_history>\n"
            + "===\n"
            + "Based on the conversation history above, please answer: "
            + resolved_prompt
        )
    else:
        final_prompt = resolved_prompt

    return final_prompt, interpolated_names


def check_abort_condition(
    prompt_spec: dict[str, Any],
    results_by_name: dict[str, dict[str, Any]],
) -> tuple[bool, str | None, str | None]:
    """Check if a prompt's ``abort_condition`` triggers a cascade abort.

    Args:
        prompt_spec: Prompt dict with optional ``abort_condition`` key.
        results_by_name: Accumulated results from prior DAG levels.

    Returns:
        Tuple of (should_abort, trace_or_None, error_or_None).

    """
    abort_condition = prompt_spec.get("abort_condition")
    if not abort_condition or not str(abort_condition).strip():
        return False, None, None

    should_execute, _, error, trace = evaluate_condition_with_trace(
        prompt_spec, results_by_name, condition_field="abort_condition"
    )

    if error:
        logger.warning(f"Abort condition evaluation error: {error}")
        return False, None, error

    if should_execute:
        return True, trace, None

    return False, None, None


def should_skip_for_failed_deps(
    node: PromptNode,
    results_by_name: dict[str, dict[str, Any]],
    nodes: dict[int, PromptNode],
) -> tuple[bool, str]:
    """Check if a node should be skipped because any dependency failed.

    Only failed dependencies trigger a skip. Skipped dependencies do NOT
    cascade — a node whose only dependency was skipped will still execute
    (its condition will determine whether it should proceed).

    Args:
        node: The prompt node to check.
        results_by_name: Accumulated results indexed by prompt_name.
        nodes: All graph nodes indexed by sequence number.

    Returns:
        Tuple of (should_skip, reason_string).

    """
    failed_deps: list[str] = []
    for dep_seq in node.dependencies:
        dep_node = nodes.get(dep_seq)
        if dep_node is None:
            continue
        dep_name = dep_node.get_prompt_name() or str(dep_seq)
        dep_result = results_by_name.get(dep_name)
        if dep_result is None:
            continue
        if dep_result.get("status") == "failed":
            failed_deps.append(dep_name)

    if failed_deps:
        reason = f"Dependency failed: {', '.join(failed_deps)}"
        return True, reason

    return False, ""
