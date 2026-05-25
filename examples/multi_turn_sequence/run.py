"""Multi-turn named prompt sequence with DataFrame history export.

Demonstrates:
1. Named prompt registration via prompt_name
2. Declarative context assembly via {{name.response}} interpolation
3. Multi-step dependency chains
4. Inspecting conversation history and ordered prompt history as DataFrames

Usage:
    python -m examples.multi_turn_sequence.run

Requires:
    - MISTRAL_API_KEY or OPENAI_API_KEY set in environment
"""

from __future__ import annotations

import os
import sys

# Ensure project root is on the path when run directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.Clients import FFLiteLLMClient
from src.core.response_options import ResponseOptions
from src.FFAI import FFAI


def print_section(title: str) -> None:
    width = 72
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)


def print_result(result: object) -> None:
    print(f"  Response:  {result.response}")
    print(f"  Model:     {result.model}")
    print(f"  Tokens:    {result.usage}")
    print(f"  Cost:      ${result.cost_usd:.6f}")
    print(f"  Duration:  {result.duration_ms:.0f}ms")


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
    # Turn 1: Seed the conversation with a topic
    # ------------------------------------------------------------------
    print_section("Turn 1: Seed topic")
    r1 = ffai.generate_response(
        prompt="Name exactly three programming languages and one sentence about each.",
        prompt_name="languages",
    )
    print_result(r1)

    # ------------------------------------------------------------------
    # Turn 2: Reference Turn 1 via {{languages.response}} interpolation
    # ------------------------------------------------------------------
    print_section("Turn 2: Build on Turn 1 via interpolation")
    r2 = ffai.generate_response(
        prompt="Which of {{languages.response}} is best suited for data science, and why?",
        prompt_name="recommendation",
    )
    print_result(r2)

    # ------------------------------------------------------------------
    # Turn 3: Multi-dependency -- interpolate two prior prompts
    # ------------------------------------------------------------------
    print_section("Turn 3: Multi-dependency interpolation")
    r3 = ffai.generate_response(
        prompt=(
            "Given {{languages.response}}, and the recommendation that "
            "{{recommendation.response}}, write a one-paragraph learning plan "
            "for a beginner starting with that language."
        ),
        prompt_name="learning_plan",
    )
    print_result(r3)

    # ------------------------------------------------------------------
    # Turn 4: Use history= parameter to inject prior turns as context
    # ------------------------------------------------------------------
    print_section("Turn 4: History-based context (declarative)")
    r4 = ffai.generate_response(
        "Summarize everything discussed so far in two sentences.",
        prompt_name="summary",
        options=ResponseOptions(history=["languages", "recommendation", "learning_plan"]),
    )
    print_result(r4)

    # ==================================================================
    # Display history as DataFrames
    # ==================================================================
    print_section("Conversation History (raw)")

    conv_df = ffai.history_to_dataframe()
    if conv_df.is_empty():
        print("  (empty)")
    else:
        cols_to_show = [c for c in ["prompt_name", "model", "response", "datetime"] if c in conv_df.columns]
        print(conv_df.select(cols_to_show))

    print_section("Ordered Prompt History")

    ordered_df = ffai.ordered_history_to_dataframe()
    if ordered_df.is_empty():
        print("  (empty)")
    else:
        cols_to_show = [
            c
            for c in ["sequence_number", "prompt_name", "prompt", "response", "model", "datetime"]
            if c in ordered_df.columns
        ]
        print(ordered_df.select(cols_to_show))

    print_section("Prompt Name Usage Stats")

    stats_df = ffai.get_prompt_name_stats_df()
    print(stats_df)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
