# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Dependency graph construction and condition evaluation.

Pure functions for building execution dependency graphs from prompt lists,
determining which prompts are ready for execution, and evaluating prompt
conditions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .condition_evaluator import ConditionEvaluator
from .execution_state import ExecutionState
from .prompt_node import PromptNode


@dataclass
class DependencyEdge:
    """A single dependency edge in the execution graph.

    Attributes:
        from_seq: Sequence number of the dependency (upstream prompt).
        to_seq: Sequence number of the dependent (downstream prompt).
        source: How the edge was derived -- 'history', 'condition', or 'abort_condition'.
        condition_text: The condition expression (set when source='condition' or 'abort_condition').

    """

    from_seq: int
    to_seq: int
    source: str
    condition_text: str | None = None


@dataclass
class ExecutionGraph:
    """Complete execution graph with dependency metadata.

    Attributes:
        nodes: Dictionary mapping sequence numbers to PromptNodes.
        edges: List of all dependency edges with source information.
        max_level: The deepest execution level in the graph.

    """

    nodes: dict[int, PromptNode] = field(default_factory=dict)
    edges: list[DependencyEdge] = field(default_factory=list)
    max_level: int = 0


def build_execution_graph(
    prompts: Sequence[dict[str, Any]],
) -> dict[int, PromptNode]:
    """Build dependency graph for parallel execution.

    Delegates to build_execution_graph_with_edges and returns only the nodes
    dict for backward compatibility.

    Args:
        prompts: List of prompt dictionaries with sequence, prompt_name,
            history, and condition fields.

    Returns:
        Dictionary mapping sequence numbers to PromptNodes.

    Raises:
        ValueError: If a dependency cycle is detected.

    """
    graph = build_execution_graph_with_edges(prompts)
    return graph.nodes


def build_execution_graph_with_edges(
    prompts: Sequence[dict[str, Any]],
) -> ExecutionGraph:
    """Build execution graph with full edge source metadata.

    Args:
        prompts: List of prompt dictionaries.

    Returns:
        ExecutionGraph with nodes, edges, and max_level.

    Raises:
        ValueError: If a dependency cycle is detected.

    """
    nodes: dict[int, PromptNode] = {}
    edges: list[DependencyEdge] = []
    prompt_by_name: dict[str, int] = {}

    for prompt in prompts:
        seq = prompt.get("sequence", 0)
        nodes[seq] = PromptNode(sequence=seq, prompt=prompt)
        name = prompt.get("prompt_name")
        if name:
            prompt_by_name[name] = seq

    for prompt in prompts:
        seq = prompt.get("sequence", 0)

        history = prompt.get("history") or []
        for dep_name in history:
            if dep_name in prompt_by_name:
                dep_seq = prompt_by_name[dep_name]
                nodes[seq].dependencies.add(dep_seq)
                edges.append(
                    DependencyEdge(
                        from_seq=dep_seq,
                        to_seq=seq,
                        source="history",
                    )
                )

        condition = prompt.get("condition") or ""
        seen_condition_deps: set[int] = set()
        for dep_name, _ in ConditionEvaluator.extract_referenced_names(condition):
            if dep_name in prompt_by_name:
                dep_seq = prompt_by_name[dep_name]
                if dep_seq == seq:
                    continue
                nodes[seq].dependencies.add(dep_seq)
                if dep_seq not in seen_condition_deps:
                    seen_condition_deps.add(dep_seq)
                    edges.append(
                        DependencyEdge(
                            from_seq=dep_seq,
                            to_seq=seq,
                            source="condition",
                            condition_text=str(condition),
                        )
                    )

        abort_condition = prompt.get("abort_condition") or ""
        seen_abort_deps: set[int] = set()
        for dep_name, _ in ConditionEvaluator.extract_referenced_names(abort_condition):
            if dep_name in prompt_by_name:
                dep_seq = prompt_by_name[dep_name]
                if dep_seq == seq:
                    continue
                nodes[seq].dependencies.add(dep_seq)
                if dep_seq not in seen_abort_deps:
                    seen_abort_deps.add(dep_seq)
                    edges.append(
                        DependencyEdge(
                            from_seq=dep_seq,
                            to_seq=seq,
                            source="abort_condition",
                            condition_text=str(abort_condition),
                        )
                    )

    level_cache: dict[int, int] = {}

    def assign_levels(seq: int, path: set[int]) -> int:
        if seq in level_cache:
            return level_cache[seq]
        if seq in path:
            cycle_seqs = sorted(path | {seq})
            raise ValueError(f"Dependency cycle detected involving sequences: {cycle_seqs}")
        path.add(seq)
        if not nodes[seq].dependencies:
            nodes[seq].level = 0
            level_cache[seq] = 0
            return 0
        max_dep_level = max(assign_levels(dep, path) for dep in nodes[seq].dependencies)
        nodes[seq].level = max_dep_level + 1
        level_cache[seq] = nodes[seq].level
        path.discard(seq)
        return nodes[seq].level

    max_level = 0
    for seq in nodes:
        lvl = assign_levels(seq, set())
        if lvl > max_level:
            max_level = lvl

    return ExecutionGraph(nodes=nodes, edges=edges, max_level=max_level)


def get_ready_prompts(state: ExecutionState, nodes: dict[int, PromptNode]) -> list[PromptNode]:
    """Get prompts ready for execution (all dependencies completed).

    Args:
        state: Current execution state.
        nodes: Execution graph nodes.

    Returns:
        List of PromptNodes ready for execution, sorted by level and sequence.

    """
    ready: list[PromptNode] = []
    for seq, node in nodes.items():
        if seq in state.completed or seq in state.in_progress:
            continue
        if node.dependencies.issubset(state.completed):
            ready.append(node)
    ready.sort(key=lambda n: (n.level, n.sequence))
    return ready


def evaluate_condition(
    prompt: dict[str, Any],
    results_by_name: dict[str, dict[str, Any]],
    condition_field: str = "condition",
) -> tuple[bool, bool | None, str | None]:
    """Evaluate a prompt's condition.

    Args:
        prompt: Prompt dictionary with optional condition.
        results_by_name: Results indexed by prompt_name.
        condition_field: Name of the field containing the condition expression.

    Returns:
        Tuple of (should_execute, condition_result, condition_error).

    """
    condition = prompt.get(condition_field)

    if not condition or not str(condition).strip():
        return True, None, None

    evaluator = ConditionEvaluator(results_by_name)
    result, error = evaluator.evaluate(str(condition))

    return result, result, error


def evaluate_condition_with_trace(
    prompt: dict[str, Any],
    results_by_name: dict[str, dict[str, Any]],
    condition_field: str = "condition",
) -> tuple[bool, bool | None, str | None, str | None]:
    """Evaluate a prompt's condition and return the resolved trace.

    Args:
        prompt: Prompt dictionary with optional condition.
        results_by_name: Results indexed by prompt_name.
        condition_field: Name of the field containing the condition expression.

    Returns:
        Tuple of (should_execute, condition_result, condition_error, condition_trace).

    """
    condition = prompt.get(condition_field)

    if not condition or not str(condition).strip():
        return True, None, None, None

    evaluator = ConditionEvaluator(results_by_name)
    result, error, trace = evaluator.evaluate_with_trace(str(condition))

    return result, result, error, trace


def is_abort_trigger(result: dict[str, Any]) -> bool:
    """Check whether a result triggered an abort.

    Args:
        result: A result dictionary from prompt execution.

    Returns:
        True if this result should trigger abort of remaining prompts.

    """
    return result.get("abort_trace") is not None and result["status"] == "success"
