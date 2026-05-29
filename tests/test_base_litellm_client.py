# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

import importlib
import os
from unittest.mock import MagicMock, patch

import pytest

from ffai.Clients.AsyncFFLiteLLMClient import AsyncFFLiteLLMClient
from ffai.Clients.BaseLiteLLMClient import BaseLiteLLMClient
from ffai.Clients.FFLiteLLMClient import FFLiteLLMClient
from ffai.core.async_client_base import AsyncFFAIClientBase
from ffai.FFAIClientBase import FFAIClientBase

_fflitellm_mod = importlib.import_module("ffai.Clients.FFLiteLLMClient")
_baselitellm_mod = importlib.import_module("ffai.Clients.BaseLiteLLMClient")


@pytest.fixture
def sync_client():
    with patch.object(_fflitellm_mod, "completion"):
        return FFLiteLLMClient(model_string="openai/gpt-4", api_key="test-key")


class TestBaseLiteLLMClientInit:
    def test_init_with_model_string(self, sync_client):
        assert sync_client._model_string == "openai/gpt-4"
        assert sync_client.model == "gpt-4"

    def test_init_with_custom_settings(self):
        with patch.object(_fflitellm_mod, "completion"):
            client = FFLiteLLMClient(
                model_string="openai/gpt-4",
                api_key="key",
                temperature=0.5,
                max_tokens=2000,
                system_instructions="Be helpful",
            )
        assert client.temperature == 0.5
        assert client.max_tokens == 2000
        assert client.system_instructions == "Be helpful"

    def test_init_with_config_dict(self):
        with patch.object(_fflitellm_mod, "completion"):
            client = FFLiteLLMClient(
                model_string="openai/gpt-4",
                config={"temperature": 0.3, "api_key": "cfg-key"},
            )
        assert client.temperature == 0.3
        assert client.api_key == "cfg-key"

    def test_init_with_fallbacks(self):
        with patch.object(_fflitellm_mod, "completion"):
            client = FFLiteLLMClient(
                model_string="openai/gpt-4",
                api_key="key",
                fallbacks=["anthropic/claude-3-opus"],
            )
        assert client._fallbacks == ["anthropic/claude-3-opus"]

    def test_init_default_retry_config(self):
        with patch.object(_fflitellm_mod, "completion"):
            client = FFLiteLLMClient(model_string="openai/gpt-4", api_key="key")
        assert client._retry_config["max_attempts"] == 3


class TestBaseLiteLLMClientResolveSettings:
    def test_constructor_over_config(self):
        with patch.object(_fflitellm_mod, "completion"):
            client = FFLiteLLMClient(
                model_string="openai/gpt-4",
                config={"temperature": 0.3},
                temperature=0.9,
                api_key="key",
            )
        assert client.temperature == 0.9

    def test_config_over_defaults(self):
        with patch("ffai.Clients.BaseLiteLLMClient.get_model_defaults", return_value={"temperature": 0.1}):
            with patch.object(_fflitellm_mod, "completion"):
                client = FFLiteLLMClient(
                    model_string="openai/gpt-4",
                    config={"temperature": 0.5},
                    api_key="key",
                )
        assert client.temperature == 0.5

    def test_defaults_as_fallback(self):
        with patch("ffai.Clients.BaseLiteLLMClient.get_model_defaults", return_value={"temperature": 0.1, "max_tokens": 2048}):
            with patch.object(_fflitellm_mod, "completion"):
                client = FFLiteLLMClient(model_string="openai/gpt-4", api_key="key")
        assert client.temperature == 0.1
        assert client.max_tokens == 2048

    def test_extra_kwargs_preserved(self):
        with patch.object(_fflitellm_mod, "completion"):
            client = FFLiteLLMClient(
                model_string="openai/gpt-4",
                api_key="key",
                top_p=0.9,
            )
        assert client._extra_kwargs == {"top_p": 0.9}


class TestBaseLiteLLMClientEnvVars:
    def test_get_env_openai_prefix(self):
        with patch.object(_fflitellm_mod, "completion"):
            client = FFLiteLLMClient(model_string="openai/gpt-4")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "env-key"}):
            assert client._get_env("API_KEY") == "env-key"

    def test_get_env_anthropic_prefix(self):
        with patch.object(_fflitellm_mod, "completion"):
            client = FFLiteLLMClient(model_string="anthropic/claude-3-opus")
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "anthropic-key"}):
            assert client._get_env("API_KEY") == "anthropic-key"

    def test_get_env_litellm_fallback(self):
        with patch.object(_fflitellm_mod, "completion"):
            client = FFLiteLLMClient(model_string="openai/gpt-4")
        with patch.dict(os.environ, {"OPENAI_API_KEY": "", "LITELLM_API_KEY": "litellm-key"}):
            result = client._get_env("API_KEY")
            assert result == "litellm-key"

    def test_get_env_generic_provider(self):
        with patch.object(_fflitellm_mod, "completion"):
            client = FFLiteLLMClient(model_string="someprovider/my-model")
        with patch.dict(os.environ, {"SOMEPROVIDER_API_KEY": "gen-key"}):
            assert client._get_env("API_KEY") == "gen-key"


class TestBaseLiteLLMClientBuildMessages:
    def test_with_system(self, sync_client):
        msgs = sync_client._build_messages()
        assert len(msgs) == 1
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == sync_client.system_instructions

    def test_override_system(self, sync_client):
        msgs = sync_client._build_messages(system_instructions="Override")
        assert msgs[0]["content"] == "Override"

    def test_no_system(self):
        with patch.object(_fflitellm_mod, "completion"):
            client = FFLiteLLMClient(
                model_string="openai/gpt-4",
                api_key="key",
            )
        client.system_instructions = ""
        msgs = client._build_messages()
        assert len(msgs) == 0

    def test_includes_history(self, sync_client):
        sync_client.conversation_history.append({"role": "user", "content": "hi"})
        msgs = sync_client._build_messages()
        assert msgs[-1]["content"] == "hi"


class TestBaseLiteLLMClientPrepareParams:
    def test_basic(self, sync_client):
        params, model_string = sync_client._prepare_generate_params(
            "Hello", None, None, None, None
        )
        assert model_string == "openai/gpt-4"
        assert params["messages"][-1]["content"] == "Hello"
        assert params["temperature"] == sync_client.temperature

    def test_empty_prompt_raises(self, sync_client):
        with pytest.raises(ValueError, match="Empty prompt"):
            sync_client._prepare_generate_params("   ", None, None, None, None)

    def test_model_override_keeps_provider(self, sync_client):
        params, model_string = sync_client._prepare_generate_params(
            "Hello", "gpt-3.5-turbo", None, None, None
        )
        assert model_string == "openai/gpt-3.5-turbo"

    def test_model_override_full_path(self, sync_client):
        params, model_string = sync_client._prepare_generate_params(
            "Hello", "anthropic/claude-3-opus", None, None, None
        )
        assert model_string == "anthropic/claude-3-opus"

    def test_includes_api_key(self, sync_client):
        params, _ = sync_client._prepare_generate_params("Hello", None, None, None, None)
        assert params["api_key"] == "test-key"

    def test_extra_kwargs_merged(self):
        with patch.object(_fflitellm_mod, "completion"):
            client = FFLiteLLMClient(
                model_string="openai/gpt-4",
                api_key="key",
                top_p=0.9,
            )
        params, _ = client._prepare_generate_params("Hello", None, None, None, None)
        assert params["top_p"] == 0.9

    def test_call_kwargs_merged(self, sync_client):
        params, _ = sync_client._prepare_generate_params(
            "Hello", None, None, None, None, custom_param="val"
        )
        assert params["custom_param"] == "val"


class TestBaseLiteLLMClientExtractUsage:
    def test_with_tokens(self, sync_client):
        mock_response = MagicMock()
        mock_response.usage.prompt_tokens = 100
        mock_response.usage.completion_tokens = 50
        mock_response.usage.total_tokens = 150

        with patch("litellm.completion_cost", return_value=0.01):
            sync_client._extract_usage(mock_response, "test-model")

        assert sync_client._last_usage.input_tokens == 100
        assert sync_client._last_usage.output_tokens == 50
        assert sync_client._last_usage.total_tokens == 150
        assert sync_client._last_cost_usd == 0.01

    def test_no_usage_attr(self, sync_client):
        mock_response = MagicMock(spec=[])
        with patch("litellm.completion_cost", return_value=0.0):
            sync_client._extract_usage(mock_response, "test-model")
        assert sync_client._last_usage is None

    def test_cost_failure_defaults_zero(self, sync_client):
        mock_response = MagicMock()
        mock_response.usage = None
        with patch("litellm.completion_cost", side_effect=Exception("no pricing")):
            sync_client._extract_usage(mock_response, "test-model")
        assert sync_client._last_cost_usd == 0.0


class TestBaseLiteLLMClientSerializeToolCalls:
    def test_dict_tool_calls(self, sync_client):
        tool_calls = [{"id": "tc_1", "function": {"name": "search", "arguments": '{"q": "test"}'}}]
        result = sync_client._serialize_tool_calls(tool_calls)
        assert len(result) == 1
        assert result[0]["id"] == "tc_1"
        assert result[0]["function"]["name"] == "search"

    def test_object_tool_calls(self, sync_client):
        tc = MagicMock()
        tc.id = "tc_2"
        fn = MagicMock()
        fn.name = "calc"
        fn.arguments = '{"expr": "1+1"}'
        tc.function = fn
        result = sync_client._serialize_tool_calls([tc])
        assert result[0]["function"]["name"] == "calc"

    def test_empty_list(self, sync_client):
        assert sync_client._serialize_tool_calls([]) == []


class TestBaseLiteLLMClientRecordResponse:
    def test_plain_response(self, sync_client):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Hello!"
        mock_resp.choices[0].message.tool_calls = None
        mock_resp.usage = None

        with patch("litellm.completion_cost", return_value=0.0):
            result = sync_client._record_response("Hi", mock_resp, "test-model")

        assert result == "Hello!"
        assert len(sync_client.conversation_history) == 2
        assert sync_client.conversation_history[0]["role"] == "user"
        assert sync_client.conversation_history[0]["content"] == "Hi"
        assert sync_client.conversation_history[1]["role"] == "assistant"
        assert sync_client.conversation_history[1]["content"] == "Hello!"

    def test_response_with_tool_calls(self, sync_client):
        tc = MagicMock()
        tc.id = "tc_1"
        fn = MagicMock()
        fn.name = "search"
        fn.arguments = "{}"
        tc.function = fn

        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Searching..."
        mock_resp.choices[0].message.tool_calls = [tc]
        mock_resp.usage = None

        with patch("litellm.completion_cost", return_value=0.0):
            result = sync_client._record_response("Search for X", mock_resp, "test-model")

        assert result == "Searching..."
        assert len(sync_client.conversation_history) == 2
        assert "tool_calls" in sync_client.conversation_history[1]


class TestBaseLiteLLMClientRecordFallbackResponse:
    def test_only_appends_assistant(self, sync_client):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.content = "Fallback answer"
        mock_resp.usage = None

        with patch("litellm.completion_cost", return_value=0.0):
            result = sync_client._record_fallback_response(mock_resp, "fallback-model")

        assert result == "Fallback answer"
        assert len(sync_client.conversation_history) == 1
        assert sync_client.conversation_history[0]["role"] == "assistant"
        assert sync_client.conversation_history[0]["content"] == "Fallback answer"


class TestBaseLiteLLMClientHistory:
    def test_get_returns_copy(self, sync_client):
        sync_client.conversation_history.append({"role": "user", "content": "hi"})
        h = sync_client.get_conversation_history()
        h.append({"role": "assistant", "content": "extra"})
        assert len(sync_client.conversation_history) == 1

    def test_set_copies(self, sync_client):
        original = [{"role": "user", "content": "hi"}]
        sync_client.set_conversation_history(original)
        original.append({"role": "assistant", "content": "extra"})
        assert len(sync_client.conversation_history) == 1

    def test_clear(self, sync_client):
        sync_client.conversation_history.append({"role": "user", "content": "hi"})
        sync_client.clear_conversation()
        assert len(sync_client.conversation_history) == 0

    def test_add_tool_result(self, sync_client):
        sync_client.add_tool_result("tc_123", "result text")
        assert len(sync_client.conversation_history) == 1
        assert sync_client.conversation_history[0]["role"] == "tool"
        assert sync_client.conversation_history[0]["tool_call_id"] == "tc_123"


class TestBaseLiteLLMClientRepr:
    def test_repr_includes_model_string(self, sync_client):
        r = repr(sync_client)
        assert "FFLiteLLMClient" in r
        assert "openai/gpt-4" in r


class TestSubclassDelegation:
    def test_sync_mro_includes_base(self):
        assert BaseLiteLLMClient in FFLiteLLMClient.__mro__

    def test_async_mro_includes_base(self):
        assert BaseLiteLLMClient in AsyncFFLiteLLMClient.__mro__

    def test_sync_is_ffaibase(self):
        with patch.object(_fflitellm_mod, "completion"):
            client = FFLiteLLMClient(model_string="openai/gpt-4", api_key="key")
        assert isinstance(client, FFAIClientBase)

    def test_async_is_async_ffaibase(self):
        with patch("ffai.Clients.BaseLiteLLMClient.get_model_defaults", return_value={}):
            client = AsyncFFLiteLLMClient(model_string="openai/gpt-4", api_key="key")
        assert isinstance(client, AsyncFFAIClientBase)

    def test_sync_inherits_init_from_mixin(self):
        with patch.object(_fflitellm_mod, "completion"):
            client = FFLiteLLMClient(
                model_string="anthropic/claude-3-opus",
                api_key="key",
                temperature=0.3,
            )
        assert client._model_string == "anthropic/claude-3-opus"
        assert client.temperature == 0.3

    def test_async_inherits_init_from_mixin(self):
        with patch("ffai.Clients.BaseLiteLLMClient.get_model_defaults", return_value={}):
            client = AsyncFFLiteLLMClient(
                model_string="anthropic/claude-3-opus",
                api_key="key",
                temperature=0.3,
            )
        assert client._model_string == "anthropic/claude-3-opus"
        assert client.temperature == 0.3
