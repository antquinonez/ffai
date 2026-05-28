"""Conditional prompt execution with DAG validation.

Demonstrates:
1. Condition-based prompt execution (skip prompts whose condition is false)
2. Condition error handling (unknown references produce status="failed")
3. Strict mode for catching typos in prompt references
4. History audit with status/condition_trace/condition_error fields
5. validate_graph() to check a workflow before running it

Usage:
    python -m examples.conditional_execution.run

Requires:
    - MISTRAL_API_KEY or OPENAI_API_KEY set in environment
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ffai.Clients import FFLiteLLMClient
from ffai.core.response_options import ResponseOptions
from ffai.FFAI import FFAI


def print_section(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_result(result: object) -> None:
    status = getattr(result, "status", "success")
    print(f"  Status:    {status}")
    print(f"  Response:  {result.response}")
    print(f"  Model:     {result.model}")
    trace = getattr(result, "condition_trace", None)
    error = getattr(result, "condition_error", None)
    if trace:
        print(f"  Cond trace:{trace}")
    if error:
        print(f"  Cond error:{error}")


def main() -> None:
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print("Error: Set MISTRAL_API_KEY in your environment.")
        sys.exit(1)

    client = FFLiteLLMClient(
        model_string="mistral/mistral-small-latest",
        api_key=api_key,
        temperature=0.7,
        max_tokens=300,
    )

    ffai = FFAI(client)

    # ------------------------------------------------------------------
    # Step 1: Validate the workflow graph before executing
    # ------------------------------------------------------------------
    print_section("Step 1: Validate workflow graph")

    workflow = [
        {"prompt_name": "fetch", "prompt": "Retrieve the latest sales data"},
        {
            "prompt_name": "analyze",
            "prompt": "Analyze the sales data",
            "history": ["fetch"],
            "condition": '{{fetch.status}} == "success"',
        },
        {
            "prompt_name": "skip_me",
            "prompt": "This should not run",
            "condition": "False",
        },
    ]

    graph, warnings = ffai.validate_graph(workflow)
    print(f"  Nodes:   {len(graph.nodes)}")
    print(f"  Edges:   {len(graph.edges)}")
    print(f"  Warnings:{len(warnings)}")
    for w in warnings:
        print(f"    - {w}")

    # ------------------------------------------------------------------
    # Step 2: Execute fetch (no condition, always runs)
    # ------------------------------------------------------------------
    print_section("Step 2: Fetch data (unconditional)")
    r1 = ffai.generate_response(
        "List 3 facts about renewable energy adoption in 2024.",
        prompt_name="fetch",
    )
    print_result(r1)

    # ------------------------------------------------------------------
    # Step 3: Execute analyze (condition: fetch succeeded)
    # ------------------------------------------------------------------
    print_section("Step 3: Analyze (condition: fetch succeeded)")
    r2 = ffai.generate_response(
        "Based on the facts above, which renewable source has the most growth potential?",
        prompt_name="analyze",
        options=ResponseOptions(
            history=["fetch"],
            condition='{{fetch.status}} == "success"',
        ),
    )
    print_result(r2)

    # ------------------------------------------------------------------
    # Step 4: Skip a prompt (condition is literal False)
    # ------------------------------------------------------------------
    print_section("Step 4: Skip (condition: False)")
    r3 = ffai.generate_response(
        "This prompt should never execute.",
        prompt_name="skip_me",
        options=ResponseOptions(condition="False"),
    )
    print_result(r3)

    # ------------------------------------------------------------------
    # Step 5: Error case (unknown reference)
    # ------------------------------------------------------------------
    print_section("Step 5: Error (unknown reference)")
    r4 = ffai.generate_response(
        "This references a prompt that doesn't exist.",
        prompt_name="bad_ref",
        options=ResponseOptions(condition='{{nonexistent.status}} == "success"'),
    )
    print_result(r4)

    # ------------------------------------------------------------------
    # Step 6: History audit
    # ------------------------------------------------------------------
    print_section("Step 6: History audit")

    conv_df = ffai.history_to_dataframe()
    if not conv_df.is_empty():
        cols = [c for c in ["prompt_name", "status", "response", "model"] if c in conv_df.columns]
        print(conv_df.select(cols))

    print()
    print("Done.")


if __name__ == "__main__":
    main()
