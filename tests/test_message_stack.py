# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Tests verifying the exact messages array sent to the LLM across turns.

These tests trace ``_prepare_generate_params`` to capture what the API would
receive — without making real API calls.  Three scenarios match the notebooks:

- **Plain**: no history param, no interpolation → messages grow as 2n
- **history=[...]**: declarative context → always 2 messages with XML block
- **{{name.response}}**: interpolation → always 2 messages with inline text
"""

from unittest.mock import MagicMock, patch

from src.Clients.FFLiteLLMClient import FFLiteLLMClient
from src.FFAI import FFAI


def _make_mock_completion():
    call_count = 0

    def mock_completion(**kwargs):
        nonlocal call_count
        call_count += 1
        response = MagicMock()
        response.choices = [MagicMock()]
        response.choices[0].message.content = f"Response {call_count}"
        response.choices[0].message.tool_calls = None
        response.usage = None
        return response

    return mock_completion


def _make_traced_client():
    captured = []

    mock_completion = _make_mock_completion()

    with patch("src.Clients.FFLiteLLMClient.completion", side_effect=mock_completion):
        client = FFLiteLLMClient(
            model_string="openai/gpt-4",
            system_instructions="You are a helpful assistant.",
        )

    original_prepare = client._prepare_generate_params

    def traced_prepare(prompt, model=None, system_instructions=None,
                       temperature=None, max_tokens=None, **kwargs):
        params, model_str = original_prepare(
            prompt, model, system_instructions, temperature, max_tokens, **kwargs
        )
        captured.append({
            "turn": len(captured) + 1,
            "messages": [dict(m) for m in params["messages"]],
        })
        return params, model_str

    client._prepare_generate_params = traced_prepare
    return client, captured


class TestMessageStackPlain:
    """Scenario A: Plain calls — client conversation_history grows every turn.

    Formula: 1 system + 2*(turn-1) prior messages + 1 user = 2*turn messages.
    """

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_message_count_grows_linearly(self, mock_completion):
        mock_completion.side_effect = _make_mock_completion()
        client = FFLiteLLMClient(model_string="openai/gpt-4",
                                 system_instructions="Be brief.")
        captured = []

        original_prepare = client._prepare_generate_params

        def traced(prompt, model=None, system_instructions=None,
                   temperature=None, max_tokens=None, **kwargs):
            params, ms = original_prepare(
                prompt, model, system_instructions, temperature, max_tokens, **kwargs
            )
            captured.append({"messages": [dict(m) for m in params["messages"]]})
            return params, ms

        client._prepare_generate_params = traced

        ffai = FFAI(client)
        for i in range(1, 11):
            ffai.generate_response(f"Turn {i}", prompt_name=f"turn_{i}")

        assert len(captured) == 10
        for i, cap in enumerate(captured):
            turn = i + 1
            expected_count = 2 * turn
            assert len(cap["messages"]) == expected_count, (
                f"Turn {turn}: expected {expected_count} messages, got {len(cap['messages'])}"
            )

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_first_turn_is_system_plus_user(self, mock_completion):
        mock_completion.side_effect = _make_mock_completion()
        client = FFLiteLLMClient(model_string="openai/gpt-4",
                                 system_instructions="Be brief.")
        captured = []

        original_prepare = client._prepare_generate_params

        def traced(prompt, model=None, system_instructions=None,
                   temperature=None, max_tokens=None, **kwargs):
            params, ms = original_prepare(
                prompt, model, system_instructions, temperature, max_tokens, **kwargs
            )
            captured.append({"messages": [dict(m) for m in params["messages"]]})
            return params, ms

        client._prepare_generate_params = traced

        ffai = FFAI(client)
        ffai.generate_response("Hello", prompt_name="turn_1")

        assert len(captured) == 1
        msgs = captured[0]["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "Be brief."
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Hello"

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_third_turn_has_prior_pairs(self, mock_completion):
        mock_completion.side_effect = _make_mock_completion()
        client = FFLiteLLMClient(model_string="openai/gpt-4",
                                 system_instructions="Be brief.")
        captured = []

        original_prepare = client._prepare_generate_params

        def traced(prompt, model=None, system_instructions=None,
                   temperature=None, max_tokens=None, **kwargs):
            params, ms = original_prepare(
                prompt, model, system_instructions, temperature, max_tokens, **kwargs
            )
            captured.append({"messages": [dict(m) for m in params["messages"]]})
            return params, ms

        client._prepare_generate_params = traced

        ffai = FFAI(client)
        ffai.generate_response("Q1", prompt_name="turn_1")
        ffai.generate_response("Q2", prompt_name="turn_2")
        ffai.generate_response("Q3", prompt_name="turn_3")

        msgs = captured[2]["messages"]
        assert len(msgs) == 6
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Q1"
        assert msgs[2]["role"] == "assistant"
        assert msgs[3]["role"] == "user"
        assert msgs[3]["content"] == "Q2"
        assert msgs[4]["role"] == "assistant"
        assert msgs[5]["role"] == "user"
        assert msgs[5]["content"] == "Q3"

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_client_history_accumulates(self, mock_completion):
        mock_completion.side_effect = _make_mock_completion()
        client = FFLiteLLMClient(model_string="openai/gpt-4")
        ffai = FFAI(client)

        for i in range(5):
            ffai.generate_response(f"Turn {i+1}", prompt_name=f"turn_{i+1}")

        assert len(client.conversation_history) == 10


class TestMessageStackHistory:
    """Scenario B: history=["prev"] — client history suspended, always 2 messages.

    The user message contains a <conversation_history> XML block with the
    referenced turn's interaction.
    """

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_always_two_messages_with_history(self, mock_completion):
        mock_completion.side_effect = _make_mock_completion()
        client = FFLiteLLMClient(model_string="openai/gpt-4",
                                 system_instructions="Be brief.")
        captured = []

        original_prepare = client._prepare_generate_params

        def traced(prompt, model=None, system_instructions=None,
                   temperature=None, max_tokens=None, **kwargs):
            params, ms = original_prepare(
                prompt, model, system_instructions, temperature, max_tokens, **kwargs
            )
            captured.append({"messages": [dict(m) for m in params["messages"]]})
            return params, ms

        client._prepare_generate_params = traced

        ffai = FFAI(client)
        ffai.generate_response("I like cats.", prompt_name="turn_1")

        for i in range(2, 11):
            ffai.generate_response(
                f"Turn {i}: What animal do I like?",
                prompt_name=f"turn_{i}",
                history=[f"turn_{i-1}"],
            )

        assert len(captured) == 10
        for i, cap in enumerate(captured):
            assert len(cap["messages"]) == 2, (
                f"Turn {i+1}: expected 2 messages, got {len(cap['messages'])}"
            )

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_xml_context_present_in_history_turn(self, mock_completion):
        mock_completion.side_effect = _make_mock_completion()
        client = FFLiteLLMClient(model_string="openai/gpt-4",
                                 system_instructions="Be brief.")
        captured = []

        original_prepare = client._prepare_generate_params

        def traced(prompt, model=None, system_instructions=None,
                   temperature=None, max_tokens=None, **kwargs):
            params, ms = original_prepare(
                prompt, model, system_instructions, temperature, max_tokens, **kwargs
            )
            captured.append({"messages": [dict(m) for m in params["messages"]]})
            return params, ms

        client._prepare_generate_params = traced

        ffai = FFAI(client)
        ffai.generate_response("I like cats.", prompt_name="turn_1")
        ffai.generate_response(
            "What animal?",
            prompt_name="turn_2",
            history=["turn_1"],
        )

        msgs = captured[1]["messages"]
        user_content = msgs[1]["content"]
        assert "<conversation_history>" in user_content
        assert "<interaction" in user_content
        assert "turn_1" in user_content

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_first_turn_has_no_xml(self, mock_completion):
        mock_completion.side_effect = _make_mock_completion()
        client = FFLiteLLMClient(model_string="openai/gpt-4",
                                 system_instructions="Be brief.")
        captured = []

        original_prepare = client._prepare_generate_params

        def traced(prompt, model=None, system_instructions=None,
                   temperature=None, max_tokens=None, **kwargs):
            params, ms = original_prepare(
                prompt, model, system_instructions, temperature, max_tokens, **kwargs
            )
            captured.append({"messages": [dict(m) for m in params["messages"]]})
            return params, ms

        client._prepare_generate_params = traced

        ffai = FFAI(client)
        ffai.generate_response("Hello", prompt_name="turn_1")

        user_content = captured[0]["messages"][1]["content"]
        assert "<conversation_history>" not in user_content

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_multiple_history_refs(self, mock_completion):
        mock_completion.side_effect = _make_mock_completion()
        client = FFLiteLLMClient(model_string="openai/gpt-4",
                                 system_instructions="Be brief.")
        captured = []

        original_prepare = client._prepare_generate_params

        def traced(prompt, model=None, system_instructions=None,
                   temperature=None, max_tokens=None, **kwargs):
            params, ms = original_prepare(
                prompt, model, system_instructions, temperature, max_tokens, **kwargs
            )
            captured.append({"messages": [dict(m) for m in params["messages"]]})
            return params, ms

        client._prepare_generate_params = traced

        ffai = FFAI(client)
        ffai.generate_response("Cats", prompt_name="a")
        ffai.generate_response("Dogs", prompt_name="b")
        ffai.generate_response("Both?", prompt_name="c", history=["a", "b"])

        msgs = captured[2]["messages"]
        user_content = msgs[1]["content"]
        assert "<conversation_history>" in user_content
        assert 'prompt_name="a"' in user_content or "prompt_name='a'" in user_content
        assert 'prompt_name="b"' in user_content or "prompt_name='b'" in user_content

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_client_history_still_grows(self, mock_completion):
        mock_completion.side_effect = _make_mock_completion()
        client = FFLiteLLMClient(model_string="openai/gpt-4")
        ffai = FFAI(client)

        ffai.generate_response("Q1", prompt_name="turn_1")
        ffai.generate_response("Q2", prompt_name="turn_2", history=["turn_1"])
        ffai.generate_response("Q3", prompt_name="turn_3", history=["turn_2"])

        assert len(client.conversation_history) == 6


class TestMessageStackInterpolation:
    """Scenario C: {{name.response}} — client history suspended, always 2 messages.

    The prior response text is substituted inline into the user prompt.
    No <conversation_history> XML block.
    """

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_always_two_messages_with_interpolation(self, mock_completion):
        mock_completion.side_effect = _make_mock_completion()
        client = FFLiteLLMClient(model_string="openai/gpt-4",
                                 system_instructions="Be brief.")
        captured = []

        original_prepare = client._prepare_generate_params

        def traced(prompt, model=None, system_instructions=None,
                   temperature=None, max_tokens=None, **kwargs):
            params, ms = original_prepare(
                prompt, model, system_instructions, temperature, max_tokens, **kwargs
            )
            captured.append({"messages": [dict(m) for m in params["messages"]]})
            return params, ms

        client._prepare_generate_params = traced

        ffai = FFAI(client)
        ffai.generate_response("I like cats.", prompt_name="turn_1")

        for i in range(2, 11):
            prompt = "You said: {{turn_" + str(i-1) + ".response}} Repeat it."
            ffai.generate_response(prompt, prompt_name=f"turn_{i}")

        assert len(captured) == 10
        for i, cap in enumerate(captured):
            assert len(cap["messages"]) == 2, (
                f"Turn {i+1}: expected 2 messages, got {len(cap['messages'])}"
            )

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_interpolated_text_present(self, mock_completion):
        mock_completion.side_effect = _make_mock_completion()
        client = FFLiteLLMClient(model_string="openai/gpt-4",
                                 system_instructions="Be brief.")
        captured = []

        original_prepare = client._prepare_generate_params

        def traced(prompt, model=None, system_instructions=None,
                   temperature=None, max_tokens=None, **kwargs):
            params, ms = original_prepare(
                prompt, model, system_instructions, temperature, max_tokens, **kwargs
            )
            captured.append({"messages": [dict(m) for m in params["messages"]]})
            return params, ms

        client._prepare_generate_params = traced

        ffai = FFAI(client)
        ffai.generate_response("I like cats.", prompt_name="turn_1")
        ffai.generate_response(
            "Recall: {{turn_1.response}} What was that?",
            prompt_name="turn_2",
        )

        user_content = captured[1]["messages"][1]["content"]
        assert "Response 1" in user_content
        assert "{{turn_1.response}}" not in user_content

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_no_xml_block_in_interpolation(self, mock_completion):
        mock_completion.side_effect = _make_mock_completion()
        client = FFLiteLLMClient(model_string="openai/gpt-4",
                                 system_instructions="Be brief.")
        captured = []

        original_prepare = client._prepare_generate_params

        def traced(prompt, model=None, system_instructions=None,
                   temperature=None, max_tokens=None, **kwargs):
            params, ms = original_prepare(
                prompt, model, system_instructions, temperature, max_tokens, **kwargs
            )
            captured.append({"messages": [dict(m) for m in params["messages"]]})
            return params, ms

        client._prepare_generate_params = traced

        ffai = FFAI(client)
        ffai.generate_response("Hello", prompt_name="t1")
        ffai.generate_response("You said: {{t1.response}}", prompt_name="t2")

        user_content = captured[1]["messages"][1]["content"]
        assert "<conversation_history>" not in user_content

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_client_history_still_grows(self, mock_completion):
        mock_completion.side_effect = _make_mock_completion()
        client = FFLiteLLMClient(model_string="openai/gpt-4")
        ffai = FFAI(client)

        ffai.generate_response("Q1", prompt_name="t1")
        ffai.generate_response("You said: {{t1.response}}", prompt_name="t2")
        ffai.generate_response("You said: {{t2.response}}", prompt_name="t3")

        assert len(client.conversation_history) == 6


def _make_tracer(orig_prepare, captured):
    def traced(prompt, model=None, system_instructions=None,
               temperature=None, max_tokens=None, **kwargs):
        params, ms = orig_prepare(
            prompt, model, system_instructions, temperature, max_tokens, **kwargs
        )
        captured.append({"messages": [dict(m) for m in params["messages"]]})
        return params, ms
    return traced


class TestMessageStackMixedScenarios:
    """Cross-scenario tests: mixing plain, history, and interpolation."""

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_plain_then_history_then_plain(self, mock_completion):
        mock_completion.side_effect = _make_mock_completion()
        client = FFLiteLLMClient(model_string="openai/gpt-4",
                                 system_instructions="Be brief.")
        captured = []

        original_prepare = client._prepare_generate_params

        def traced(prompt, model=None, system_instructions=None,
                   temperature=None, max_tokens=None, **kwargs):
            params, ms = original_prepare(
                prompt, model, system_instructions, temperature, max_tokens, **kwargs
            )
            captured.append({"messages": [dict(m) for m in params["messages"]]})
            return params, ms

        client._prepare_generate_params = traced

        ffai = FFAI(client)

        ffai.generate_response("P1", prompt_name="p1")
        assert len(captured[-1]["messages"]) == 2

        ffai.generate_response("P2", prompt_name="p2")
        assert len(captured[-1]["messages"]) == 4

        ffai.generate_response("H1", prompt_name="h1", history=["p2"])
        assert len(captured[-1]["messages"]) == 2

        ffai.generate_response("P3", prompt_name="p3")
        assert len(captured[-1]["messages"]) == 8

        assert len(client.conversation_history) == 8

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_plain_then_interpolation_then_plain(self, mock_completion):
        mock_completion.side_effect = _make_mock_completion()
        client = FFLiteLLMClient(model_string="openai/gpt-4",
                                 system_instructions="Be brief.")
        captured = []

        original_prepare = client._prepare_generate_params

        def traced(prompt, model=None, system_instructions=None,
                   temperature=None, max_tokens=None, **kwargs):
            params, ms = original_prepare(
                prompt, model, system_instructions, temperature, max_tokens, **kwargs
            )
            captured.append({"messages": [dict(m) for m in params["messages"]]})
            return params, ms

        client._prepare_generate_params = traced

        ffai = FFAI(client)

        ffai.generate_response("P1", prompt_name="p1")
        assert len(captured[-1]["messages"]) == 2

        ffai.generate_response("You said: {{p1.response}}", prompt_name="i1")
        assert len(captured[-1]["messages"]) == 2

        ffai.generate_response("P3", prompt_name="p3")
        assert len(captured[-1]["messages"]) == 6

        assert len(client.conversation_history) == 6

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_ten_turn_comparison(self, mock_completion):
        """Full 10-turn comparison matching the notebook output."""
        mock_completion.side_effect = _make_mock_completion()

        results = {}

        for scenario in ("A", "B", "C"):
            client = FFLiteLLMClient(model_string="openai/gpt-4",
                                     system_instructions="Be brief.")
            cap_list: list[dict] = []
            client._prepare_generate_params = _make_tracer(
                client._prepare_generate_params, cap_list
            )
            ffai = FFAI(client)

            if scenario == "A":
                for i in range(1, 11):
                    ffai.generate_response(f"Turn {i}", prompt_name=f"turn_{i}")
            elif scenario == "B":
                ffai.generate_response("Color is blue.", prompt_name="turn_1")
                for i in range(2, 11):
                    ffai.generate_response(
                        f"Turn {i}",
                        prompt_name=f"turn_{i}",
                        history=[f"turn_{i-1}"],
                    )
            else:
                ffai.generate_response("Color is blue.", prompt_name="turn_1")
                for i in range(2, 11):
                    prompt = "You said: {{turn_" + str(i-1) + ".response}}"
                    ffai.generate_response(prompt, prompt_name=f"turn_{i}")

            results[scenario] = cap_list

        for i in range(10):
            turn = i + 1
            a_count = len(results["A"][i]["messages"])
            b_count = len(results["B"][i]["messages"])
            c_count = len(results["C"][i]["messages"])

            assert a_count == 2 * turn, f"Turn {turn} A: {a_count} != {2*turn}"
            assert b_count == 2, f"Turn {turn} B: {b_count} != 2"
            assert c_count == 2, f"Turn {turn} C: {c_count} != 2"
