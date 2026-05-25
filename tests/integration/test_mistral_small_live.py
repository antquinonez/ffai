# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Integration tests for LLM clients against live APIs.

Contract tests (TestContract*) run against every client enabled in
test_config.yaml.  Client-specific tests (TestFFMistralSmall*) run
only against matching client_class entries.
"""

import pytest

from src.Clients.FFMistralSmall import FFMistralSmall
from src.core.client_base import FFAIClientBase

pytestmark = pytest.mark.integration


# ── Contract tests: run against every enabled client ──────────────────


class TestContractBasicGeneration:
    def test_returns_non_empty_string(self, integration_client: FFAIClientBase):
        result = integration_client.generate_response("What is 2+2?")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_response_contains_relevant_content(self, integration_client: FFAIClientBase):
        result = integration_client.generate_response("What is the capital of France?")
        assert "Paris" in result or "paris" in result.lower()

    def test_empty_prompt_raises_value_error(self, integration_client: FFAIClientBase):
        with pytest.raises(ValueError, match="Empty prompt"):
            integration_client.generate_response("   ")

    def test_model_override(self, integration_client: FFAIClientBase):
        result = integration_client.generate_response(
            "Say hello in one word.", model=integration_client.model
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestContractConversationHistory:
    def test_history_accumulates(self, integration_client: FFAIClientBase):
        integration_client.generate_response("Say: apple")
        integration_client.generate_response("Say: banana")
        history = integration_client.get_conversation_history()
        assert len(history) == 4
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
        assert history[2]["role"] == "user"
        assert history[3]["role"] == "assistant"
        assert "apple" in history[0]["content"]
        assert "banana" in history[2]["content"]

    def test_clear_conversation(self, integration_client: FFAIClientBase):
        integration_client.generate_response("Say: hello")
        assert len(integration_client.get_conversation_history()) == 2
        integration_client.clear_conversation()
        assert len(integration_client.get_conversation_history()) == 0

    def test_set_and_get_history(self, integration_client: FFAIClientBase):
        msgs = [
            {"role": "user", "content": "test message"},
            {"role": "assistant", "content": "test response"},
        ]
        integration_client.set_conversation_history(msgs)
        history = integration_client.get_conversation_history()
        assert len(history) == 2
        assert history[0]["content"] == "test message"
        assert history[1]["content"] == "test response"

    def test_clone_has_empty_history(self, integration_client: FFAIClientBase):
        integration_client.generate_response("Say: original")
        assert len(integration_client.get_conversation_history()) == 2
        clone = integration_client.clone()
        assert len(clone.get_conversation_history()) == 0
        assert clone.model == integration_client.model
        assert clone.system_instructions == integration_client.system_instructions


class TestContractUsageTracking:
    def test_last_usage_populated(self, integration_client: FFAIClientBase):
        integration_client.generate_response("Say: test")
        usage = integration_client.last_usage
        assert usage is not None
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
        assert usage.total_tokens == usage.input_tokens + usage.output_tokens

    def test_last_cost_non_negative(self, integration_client: FFAIClientBase):
        integration_client.generate_response("Say: test")
        assert integration_client.last_cost_usd >= 0

    def test_usage_resets_per_call(self, integration_client: FFAIClientBase):
        integration_client.generate_response("Say: first")
        first_usage = integration_client.last_usage
        assert first_usage is not None
        integration_client.generate_response("Say: second")
        second_usage = integration_client.last_usage
        assert second_usage is not None
        assert second_usage is not first_usage


class TestContractParameterHandling:
    def test_system_instructions_override(self, integration_client: FFAIClientBase):
        result = integration_client.generate_response(
            "What is your name?",
            system_instructions="You are a bot named Integra. Always introduce yourself as Integra.",
        )
        assert "integra" in result.lower()

    def test_temperature_zero_is_determinish(self, integration_client: FFAIClientBase):
        r1 = integration_client.generate_response("Say exactly the word: hello")
        integration_client.clear_conversation()
        r2 = integration_client.generate_response("Say exactly the word: hello")
        assert r1.strip().lower()[:5] == r2.strip().lower()[:5]

    def test_max_tokens_limits_length(self, integration_client: FFAIClientBase):
        result = integration_client.generate_response(
            "Count from 1 to 100, one number per line.", max_tokens=20
        )
        word_count = len(result.split())
        assert word_count < 30


# ── Client-specific tests ─────────────────────────────────────────────


class TestFFMistralSmallToolCalling:
    WEATHER_TOOL = {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                },
                "required": ["city"],
            },
        },
    }

    def test_tool_calls_detected(self, ffmistralsmall_client: FFMistralSmall):
        result = ffmistralsmall_client.generate_response(
            "What is the weather in Paris right now?",
            tools=[self.WEATHER_TOOL],
        )
        assert "Tool calls detected" in result
        history = ffmistralsmall_client.get_conversation_history()
        assistant_msg = [m for m in history if m["role"] == "assistant"][0]
        assert "tool_calls" in assistant_msg
        assert assistant_msg["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_add_tool_result(self, ffmistralsmall_client: FFMistralSmall):
        ffmistralsmall_client.generate_response(
            "What is the weather in Paris?",
            tools=[self.WEATHER_TOOL],
        )
        history = ffmistralsmall_client.get_conversation_history()
        tc_id = history[-1]["tool_calls"][0]["id"]
        ffmistralsmall_client.add_tool_result(tc_id, '{"temp": 18, "condition": "sunny"}')
        updated = ffmistralsmall_client.get_conversation_history()
        tool_msg = [m for m in updated if m["role"] == "tool"][-1]
        assert tool_msg["tool_call_id"] == tc_id
        assert "sunny" in tool_msg["content"]
