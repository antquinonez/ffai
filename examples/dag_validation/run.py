"""DAG validation and dependency graph inspection.

Demonstrates:
1. Building a dependency graph from a prompt list
2. Cycle detection (raises ValueError on circular dependencies)
3. Edge source tracking (history vs condition-sourced)
4. Level-based parallel scheduling with get_ready_prompts()
5. Undeclared dependency warnings via validate_graph()

Usage:
    python -m examples.dag_validation.run

No API key required -- this example only uses graph-building utilities.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ffai.Clients.FFLiteLLMClient import FFLiteLLMClient
from ffai.config import Config
from ffai.core.execution_state import ExecutionState
from ffai.core.graph import build_execution_graph_with_edges, get_ready_prompts
from ffai.FFAI import FFAI


def print_section(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def main() -> None:
    mock_config = MagicMock(spec=Config)
    mock_config.paths = MagicMock()
    mock_config.paths.ffai_data = "/tmp/ffai_dag_example"

    mock_client = MagicMock(spec=FFLiteLLMClient)
    mock_client.model = "mistral/mistral-small-latest"

    with patch("ffai.FFAI.get_config", return_value=mock_config):
        ffai = FFAI(mock_client)

    # ------------------------------------------------------------------
    # Example 1: Build a simple DAG
    # ------------------------------------------------------------------
    print_section("Example 1: Build a simple DAG")

    prompts = [
        {"sequence": 0, "prompt_name": "context", "prompt": "I run a coffee shop.", "history": []},
        {
            "sequence": 1,
            "prompt_name": "problem",
            "prompt": "My electricity bill is too high.",
            "history": ["context"],
        },
        {
            "sequence": 2,
            "prompt_name": "solution",
            "prompt": "Suggest 3 ways to reduce costs.",
            "history": ["context", "problem"],
        },
    ]

    graph = build_execution_graph_with_edges(prompts)

    print(f"  Nodes:    {len(graph.nodes)}")
    print(f"  Edges:    {len(graph.edges)}")
    print(f"  Max level:{graph.max_level}")
    print()

    for seq, node in sorted(graph.nodes.items()):
        name = node.get_prompt_name() or f"seq_{seq}"
        deps = node.dependencies
        print(f"    {name:12s}  level={node.level}  dependencies={deps or 'none'}")

    # ------------------------------------------------------------------
    # Example 2: Inspect dependency edges
    # ------------------------------------------------------------------
    print_section("Example 2: Inspect dependency edges")

    for edge in graph.edges:
        from_name = graph.nodes[edge.from_seq].get_prompt_name()
        to_name = graph.nodes[edge.to_seq].get_prompt_name()
        print(f"    {from_name:12s} -> {to_name:12s}  [{edge.source}]")

    # ------------------------------------------------------------------
    # Example 3: Condition-sourced implicit edges
    # ------------------------------------------------------------------
    print_section("Example 3: Condition-sourced implicit edges")

    prompts_with_conditions = [
        {"sequence": 0, "prompt_name": "fetch", "prompt": "Retrieve the data", "history": []},
        {
            "sequence": 1,
            "prompt_name": "process",
            "prompt": "Process the data",
            "history": [],
            "condition": '{{fetch.status}} == "success"',
        },
    ]

    graph2 = build_execution_graph_with_edges(prompts_with_conditions)

    for edge in graph2.edges:
        from_name = graph2.nodes[edge.from_seq].get_prompt_name()
        to_name = graph2.nodes[edge.to_seq].get_prompt_name()
        cond = f' condition="{edge.condition_text}"' if edge.condition_text else ""
        print(f"    {from_name} -> {to_name}  [{edge.source}]{cond}")

    print()
    print(f"    process dependencies: {graph2.nodes[1].dependencies}")
    print(f"    process level:        {graph2.nodes[1].level}")

    # ------------------------------------------------------------------
    # Example 4: Cycle detection
    # ------------------------------------------------------------------
    print_section("Example 4: Cycle detection")

    circular_prompts = [
        {"sequence": 0, "prompt_name": "a", "prompt": "First", "history": ["b"]},
        {"sequence": 1, "prompt_name": "b", "prompt": "Second", "history": ["a"]},
    ]

    try:
        build_execution_graph_with_edges(circular_prompts)
        print("    No error raised (unexpected)")
    except ValueError as e:
        print(f"    Cycle detected: {e}")

    # ------------------------------------------------------------------
    # Example 5: Parallel-ready execution
    # ------------------------------------------------------------------
    print_section("Example 5: Parallel-ready execution with get_ready_prompts")

    prompts_parallel = [
        {"sequence": 0, "prompt_name": "a", "prompt": "Task A", "history": []},
        {"sequence": 1, "prompt_name": "b", "prompt": "Task B", "history": []},
        {"sequence": 2, "prompt_name": "c", "prompt": "Synthesize", "history": ["a", "b"]},
    ]

    graph3 = build_execution_graph_with_edges(prompts_parallel)
    state = ExecutionState()

    for round_num in range(1, 4):
        ready = get_ready_prompts(state, graph3.nodes)
        names = [n.get_prompt_name() for n in ready]
        print(f"    Round {round_num}: Ready={names}")
        for node in ready:
            state.completed.add(node.sequence)
        if not ready:
            break

    print(f"    All completed: {len(state.completed) == len(graph3.nodes)}")

    # ------------------------------------------------------------------
    # Example 6: validate_graph() with warnings
    # ------------------------------------------------------------------
    print_section("Example 6: validate_graph() with warnings")

    workflow = [
        {"prompt_name": "fetch", "prompt": "Retrieve the data"},
        {
            "prompt_name": "process",
            "prompt": "Process the results",
            "condition": '{{fetch.status}} == "success"',
        },
    ]

    graph4, warnings = ffai.validate_graph(workflow)

    print(f"  Nodes:    {len(graph4.nodes)}")
    print(f"  Max level:{graph4.max_level}")
    print(f"  Warnings: {len(warnings)}")
    for w in warnings:
        print(f"    - {w}")

    # ------------------------------------------------------------------
    # Example 7: Clean graph (no warnings)
    # ------------------------------------------------------------------
    print_section("Example 7: Clean graph (no warnings)")

    clean_workflow = [
        {"prompt_name": "fetch", "prompt": "Retrieve the data"},
        {
            "prompt_name": "process",
            "prompt": "Process the results",
            "history": ["fetch"],
            "condition": '{{fetch.status}} == "success"',
        },
    ]

    graph5, warnings2 = ffai.validate_graph(clean_workflow)
    print(f"  Nodes:    {len(graph5.nodes)}")
    print(f"  Max level:{graph5.max_level}")
    print(f"  Warnings: {len(warnings2)}")

    # ------------------------------------------------------------------
    # Example 8: Cycle detection via validate_graph
    # ------------------------------------------------------------------
    print_section("Example 8: Cycle detection via validate_graph")

    circular_workflow = [
        {"prompt_name": "a", "prompt": "Step A", "history": ["b"]},
        {"prompt_name": "b", "prompt": "Step B", "history": ["a"]},
    ]

    try:
        ffai.validate_graph(circular_workflow)
        print("    No error raised (unexpected)")
    except ValueError as e:
        print(f"    Cycle detected: {e}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
