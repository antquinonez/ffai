# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

import asyncio
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.Clients.AsyncFFLiteLLMClient import AsyncFFLiteLLMClient
from src.core.usage import TokenUsage


def _make_mock_response(content: str | None = "test response", tool_calls=None, usage=None):
    message = MagicMock()
    message.content = content
    message.tool_calls = tool_calls

    choice = MagicMock()
    choice.message = message

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def _make_usage(input_tokens=10, output_tokens=20, total_tokens=30):
    usage = MagicMock()
    usage.prompt_tokens = input_tokens
    usage.completion_tokens = output_tokens
    usage.total_tokens = total_tokens
    return usage


@pytest.fixture
def env_key():
    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key-123"}):
        yield


@pytest.fixture
def client(env_key):
    with patch("src.Clients.AsyncFFLiteLLMClient.get_model_defaults", return_value={}):
        return AsyncFFLiteLLMClient(
            model_string="openai/gpt-4",
            api_key="test-key",
            temperature=0.5,
            max_tokens=256,
        )


class TestAsyncFFLiteLLMClientInit:
    def test_model_string_parsing_with_provider(self, env_key):
        with patch("src.Clients.AsyncFFLiteLLMClient.get_model_defaults", return_value={}):
            c = AsyncFFLiteLLMClient(model_string="anthropic/claude-3-opus", api_key="k")
        assert c.model == "claude-3-opus"

    def test_model_string_without_slash(self, env_key):
        with patch("src.Clients.AsyncFFLiteLLMClient.get_model_defaults", return_value={}):
            c = AsyncFFLiteLLMClient(model_string="gpt-4", api_key="k")
        assert c.model == "gpt-4"

    def test_default_settings(self, client):
        assert client.temperature == 0.5
        assert client.max_tokens == 256
        assert client.api_key == "test-key"
        assert client.system_instructions == "You are a helpful assistant."

    def test_custom_system_instructions(self, env_key):
        with patch("src.Clients.AsyncFFLiteLLMClient.get_model_defaults", return_value={}):
            c = AsyncFFLiteLLMClient(
                model_string="openai/gpt-4",
                api_key="k",
                system_instructions="Custom instructions",
            )
        assert c.system_instructions == "Custom instructions"

    def test_fallbacks_preserved(self, env_key):
        with patch("src.Clients.AsyncFFLiteLLMClient.get_model_defaults", return_value={}):
            c = AsyncFFLiteLLMClient(
                model_string="openai/gpt-4",
                api_key="k",
                fallbacks=["anthropic/claude-3-opus"],
            )
        assert c._fallbacks == ["anthropic/claude-3-opus"]

    def test_empty_history_on_init(self, client):
        assert client.conversation_history == []

    def test_repr(self, client):
        r = repr(client)
        assert "AsyncFFLiteLLMClient" in r
        assert "gpt-4" in r


class TestAsyncFFLiteLLMClientResolveSettings:
    def test_explicit_api_key_overrides_env(self, client):
        assert client.api_key == "test-key"

    def test_temperature_from_explicit_param(self, client):
        assert client.temperature == 0.5

    def test_max_tokens_from_explicit_param(self, client):
        assert client.max_tokens == 256


class TestAsyncFFLiteLLMClientGetEnv:
    def test_openai_prefix(self, client):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            val = client._get_env("API_KEY")
        assert val == "env-key"

    def test_anthropic_prefix(self, env_key):
        with patch("src.Clients.AsyncFFLiteLLMClient.get_model_defaults", return_value={}):
            c = AsyncFFLiteLLMClient(model_string="anthropic/claude-3", api_key="k")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "anth-key"}):
            val = c._get_env("API_KEY")
        assert val == "anth-key"

    def test_litellm_fallback(self, client):
        with patch.dict(os.environ, {"LITELLM_API_BASE": "https://litellm.example.com"}, clear=False):
            val = client._get_env("API_BASE")
        assert val == "https://litellm.example.com"

    def test_returns_none_when_not_found(self, client):
        with patch.dict(os.environ, {}, clear=True):
            val = client._get_env("NONEXISTENT_SUFFIX")
        assert val is None


class TestAsyncFFLiteLLMClientGenerateResponse:
    def test_basic_async_call(self, client):
        mock_resp = _make_mock_response("hello world")
        with patch("src.Clients.AsyncFFLiteLLMClient.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(client.generate_response("hi"))
        assert result == "hello world"

    def test_empty_prompt_raises(self, client):
        with pytest.raises(ValueError, match="Empty prompt"):
            asyncio.run(client.generate_response("   "))

    def test_model_override_bare_name_gets_prefix(self, client):
        mock_resp = _make_mock_response("ok")
        with patch("src.Clients.AsyncFFLiteLLMClient.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_ac:
            asyncio.run(client.generate_response("hi", model="gpt-3.5-turbo"))
        call_kwargs = mock_ac.call_args
        assert call_kwargs.kwargs["model"] == "openai/gpt-3.5-turbo"

    def test_model_override_full_string_passes_through(self, client):
        mock_resp = _make_mock_response("ok")
        with patch("src.Clients.AsyncFFLiteLLMClient.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_ac:
            asyncio.run(client.generate_response("hi", model="anthropic/claude-3"))
        call_kwargs = mock_ac.call_args
        assert call_kwargs.kwargs["model"] == "anthropic/claude-3"

    def test_temperature_override_per_call(self, client):
        mock_resp = _make_mock_response("ok")
        with patch("src.Clients.AsyncFFLiteLLMClient.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_ac:
            asyncio.run(client.generate_response("hi", temperature=0.1))
        call_kwargs = mock_ac.call_args
        assert call_kwargs.kwargs["temperature"] == 0.1

    def test_api_params_passed(self, client):
        mock_resp = _make_mock_response("ok")
        with patch("src.Clients.AsyncFFLiteLLMClient.acompletion", new_callable=AsyncMock, return_value=mock_resp) as mock_ac:
            asyncio.run(client.generate_response("hi"))
        call_kwargs = mock_ac.call_args.kwargs
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 256

    def test_records_to_conversation_history(self, client):
        mock_resp = _make_mock_response("answer")
        with patch("src.Clients.AsyncFFLiteLLMClient.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            asyncio.run(client.generate_response("question"))
        assert len(client.conversation_history) == 2
        assert client.conversation_history[0]["role"] == "user"
        assert client.conversation_history[1]["role"] == "assistant"
        assert client.conversation_history[1]["content"] == "answer"


class TestAsyncFFLiteLLMClientToolCalls:
    def test_tool_calls_serialized(self, client):
        tc = MagicMock()
        tc.id = "tc_1"
        tc.function = MagicMock()
        tc.function.name = "get_weather"
        tc.function.arguments = '{"city": "Paris"}'

        mock_resp = _make_mock_response("checking", tool_calls=[tc])
        with patch("src.Clients.AsyncFFLiteLLMClient.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(client.generate_response("weather?"))
        assert result == "checking"
        assert len(client.conversation_history) == 2
        assert "tool_calls" in client.conversation_history[1]

    def test_none_content_returns_empty_string(self, client):
        mock_resp = _make_mock_response(content=None)
        with patch("src.Clients.AsyncFFLiteLLMClient.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            result = asyncio.run(client.generate_response("hi"))
        assert result == ""


class TestAsyncFFLiteLLMClientExtractUsage:
    def test_usage_extracted(self, client):
        usage = _make_usage(100, 50, 150)
        mock_resp = _make_mock_response("ok", usage=usage)
        with patch("src.Clients.AsyncFFLiteLLMClient.acompletion", new_callable=AsyncMock, return_value=mock_resp), \
             patch("src.Clients.AsyncFFLiteLLMClient.litellm.completion_cost", return_value=0.005):
            asyncio.run(client.generate_response("hi"))
        assert client.last_usage is not None
        assert client.last_usage.input_tokens == 100
        assert client.last_usage.output_tokens == 50
        assert client.last_cost_usd == 0.005

    def test_no_usage_defaults(self, client):
        mock_resp = _make_mock_response("ok", usage=None)
        with patch("src.Clients.AsyncFFLiteLLMClient.acompletion", new_callable=AsyncMock, return_value=mock_resp):
            asyncio.run(client.generate_response("hi"))
        assert client.last_usage is None
        assert client.last_cost_usd == 0.0

    def test_cost_failure_defaults_to_zero(self, client):
        usage = _make_usage(10, 5, 15)
        mock_resp = _make_mock_response("ok", usage=usage)
        with patch("src.Clients.AsyncFFLiteLLMClient.acompletion", new_callable=AsyncMock, return_value=mock_resp), \
             patch("src.Clients.AsyncFFLiteLLMClient.litellm.completion_cost", side_effect=Exception("no pricing")):
            asyncio.run(client.generate_response("hi"))
        assert client.last_cost_usd == 0.0


class TestAsyncFFLiteLLMClientSerializeToolCalls:
    def test_dict_tool_calls(self, client):
        tc = {"id": "tc_1", "function": {"name": "foo", "arguments": "{}"}}
        result = client._serialize_tool_calls([tc])
        assert result[0]["id"] == "tc_1"
        assert result[0]["function"]["name"] == "foo"

    def test_object_tool_calls(self, client):
        tc = MagicMock()
        tc.id = "tc_2"
        tc.function = MagicMock()
        tc.function.name = "bar"
        tc.function.arguments = '{"x": 1}'
        result = client._serialize_tool_calls([tc])
        assert result[0]["id"] == "tc_2"
        assert result[0]["function"]["name"] == "bar"


class TestAsyncFFLiteLLMClientBuildMessages:
    def test_system_prepended(self, client):
        msgs = client._build_messages()
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == client.system_instructions

    def test_override_system(self, client):
        msgs = client._build_messages(system_instructions="override")
        assert msgs[0]["content"] == "override"

    def test_none_system_uses_default(self, env_key):
        with patch("src.Clients.AsyncFFLiteLLMClient.get_model_defaults", return_value={}):
            c = AsyncFFLiteLLMClient(model_string="openai/gpt-4", api_key="k")
        assert c.system_instructions == "You are a helpful assistant."


class TestAsyncFFLiteLLMClientFallbacks:
    def test_fallback_on_primary_failure(self, client):
        primary_resp = _make_mock_response("primary")
        fallback_resp = _make_mock_response("fallback answer")

        with patch("src.Clients.AsyncFFLiteLLMClient.acompletion", new_callable=AsyncMock, side_effect=[RuntimeError("fail"), fallback_resp]):
            client._fallbacks = ["openai/gpt-3.5-turbo"]
            result = asyncio.run(client.generate_response("hi"))
        assert result == "fallback answer"

    def test_all_fallbacks_fail_raises(self, client):
        with patch("src.Clients.AsyncFFLiteLLMClient.acompletion", new_callable=AsyncMock, side_effect=RuntimeError("fail")):
            client._fallbacks = ["openai/gpt-3.5-turbo", "anthropic/claude-3"]
            with pytest.raises(RuntimeError, match="All models failed"):
                asyncio.run(client.generate_response("hi"))

    def test_no_fallbacks_raises_original(self, client):
        with patch("src.Clients.AsyncFFLiteLLMClient.acompletion", new_callable=AsyncMock, side_effect=RuntimeError("primary fail")):
            client._fallbacks = []
            with pytest.raises(RuntimeError, match="primary fail"):
                asyncio.run(client.generate_response("hi"))


class TestAsyncFFLiteLLMClientHistory:
    def test_add_tool_result(self, client):
        client.add_tool_result("tc_1", "tool output")
        assert len(client.conversation_history) == 1
        assert client.conversation_history[0]["role"] == "tool"
        assert client.conversation_history[0]["tool_call_id"] == "tc_1"

    def test_clear_conversation(self, client):
        client.conversation_history.append({"role": "user", "content": "hi"})
        client.clear_conversation()
        assert client.conversation_history == []

    def test_get_conversation_history_returns_copy(self, client):
        client.conversation_history.append({"role": "user", "content": "hi"})
        h = client.get_conversation_history()
        h.append({"role": "user", "content": "mutation"})
        assert len(client.conversation_history) == 1

    def test_set_conversation_history_copies(self, client):
        original = [{"role": "user", "content": "hi"}]
        client.set_conversation_history(original)
        original.append({"role": "user", "content": "mutation"})
        assert len(client.conversation_history) == 1


class TestAsyncFFLiteLLMClientClone:
    def test_clone_is_new_instance(self, client):
        cloned = asyncio.run(client.clone())
        assert cloned is not client

    def test_clone_same_config(self, client):
        cloned = asyncio.run(client.clone())
        assert cloned._model_string == client._model_string
        assert cloned.temperature == client.temperature
        assert cloned.max_tokens == client.max_tokens
        assert cloned.system_instructions == client.system_instructions
        assert cloned.api_key == client.api_key

    def test_clone_empty_history(self, client):
        client.conversation_history.append({"role": "user", "content": "hi"})
        cloned = asyncio.run(client.clone())
        assert cloned.conversation_history == []
        assert len(client.conversation_history) == 1

    def test_clone_fallbacks_copied(self, client):
        client._fallbacks = ["openai/gpt-3.5-turbo"]
        cloned = asyncio.run(client.clone())
        cloned._fallbacks.append("anthropic/claude-3")
        assert len(client._fallbacks) == 1

    def test_clone_resets_usage(self, client):
        client._last_usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
        client._last_cost_usd = 0.01
        cloned = asyncio.run(client.clone())
        assert cloned.last_usage is None
        assert cloned.last_cost_usd == 0.0
