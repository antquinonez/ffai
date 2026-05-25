# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com


class TestTokenUsage:
    """Tests for TokenUsage dataclass."""

    def test_default_values(self):
        from src.core.usage import TokenUsage

        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.total_tokens == 0

    def test_custom_values(self):
        from src.core.usage import TokenUsage

        usage = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.total_tokens == 150

    def test_addition(self):
        from src.core.usage import TokenUsage

        a = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        b = TokenUsage(input_tokens=200, output_tokens=75, total_tokens=275)
        result = a + b
        assert result.input_tokens == 300
        assert result.output_tokens == 125
        assert result.total_tokens == 425

    def test_addition_does_not_mutate(self):
        from src.core.usage import TokenUsage

        a = TokenUsage(input_tokens=100, output_tokens=50, total_tokens=150)
        b = TokenUsage(input_tokens=200, output_tokens=75, total_tokens=275)
        _ = a + b
        assert a.input_tokens == 100
        assert b.input_tokens == 200


class TestClientBaseUsage:
    """Tests for usage metadata on FFAIClientBase."""

    def test_reset_usage(self, concrete_client):
        from src.core.usage import TokenUsage

        concrete_client._last_usage = TokenUsage(input_tokens=10, output_tokens=5, total_tokens=15)
        concrete_client._last_cost_usd = 0.001

        concrete_client._reset_usage()
        assert concrete_client.last_usage is None
        assert concrete_client.last_cost_usd == 0.0

    def test_initial_state(self, concrete_client):
        assert concrete_client.last_usage is None
        assert concrete_client.last_cost_usd == 0.0


class TestClientBaseRetryFallback:
    """Tests for get_default_retry_config fallback path (lines 62-65)."""

    def test_fallback_defaults_when_config_unavailable(self):
        from unittest.mock import patch

        from src.core.client_base import FFAIClientBase

        with patch("src.config.get_config", side_effect=Exception("no config")):
            defaults = FFAIClientBase.get_default_retry_config()

        assert defaults["max_attempts"] == 3
        assert defaults["min_wait_seconds"] == 1
        assert defaults["max_wait_seconds"] == 60
        assert defaults["exponential_base"] == 2
        assert defaults["exponential_jitter"] is True
        assert defaults["log_level"] == "INFO"


class TestClientBaseAbstractMethodBodies:
    """Tests that abstract base class pass-through bodies return None (lines 184, 189, 199, 209, 238)."""

    @staticmethod
    def _make_delegating_client():
        from src.core.client_base import FFAIClientBase

        class DelegatingClient(FFAIClientBase):
            model = "test"
            system_instructions = ""

            def generate_response(self, prompt, **kwargs):
                return super().generate_response(prompt, **kwargs)

            def clear_conversation(self):
                return super().clear_conversation()

            def get_conversation_history(self):
                return super().get_conversation_history()

            def set_conversation_history(self, history):
                return super().set_conversation_history(history)

            def clone(self):
                return super().clone()

        return DelegatingClient()

    def test_super_generate_response_returns_none(self):
        client = self._make_delegating_client()
        assert client.generate_response("test") is None

    def test_super_clear_conversation_returns_none(self):
        client = self._make_delegating_client()
        assert client.clear_conversation() is None

    def test_super_get_conversation_history_returns_none(self):
        client = self._make_delegating_client()
        assert client.get_conversation_history() is None

    def test_super_set_conversation_history_returns_none(self):
        client = self._make_delegating_client()
        assert client.set_conversation_history([]) is None

    def test_super_clone_returns_none(self):
        client = self._make_delegating_client()
        assert client.clone() is None


class TestClientBaseRetryConfigFromSettings:
    def test_reads_from_config_retry_attribute(self):
        from unittest.mock import MagicMock, patch

        from src.core.client_base import FFAIClientBase

        mock_config = MagicMock()
        mock_retry = MagicMock()
        mock_retry.max_attempts = 5
        mock_retry.min_wait_seconds = 2
        mock_retry.max_wait_seconds = 120
        mock_retry.exponential_base = 3
        mock_retry.exponential_jitter = False
        mock_retry.log_level = "DEBUG"
        mock_config.retry = mock_retry

        with patch("src.config.get_config", return_value=mock_config):
            result = FFAIClientBase.get_default_retry_config()

        assert result["max_attempts"] == 5
        assert result["min_wait_seconds"] == 2
        assert result["max_wait_seconds"] == 120
        assert result["exponential_base"] == 3
        assert result["exponential_jitter"] is False
        assert result["log_level"] == "DEBUG"

    def test_returns_defaults_when_retry_is_none(self):
        from unittest.mock import MagicMock, patch

        from src.core.client_base import FFAIClientBase

        mock_config = MagicMock()
        mock_config.retry = None

        with patch("src.config.get_config", return_value=mock_config):
            result = FFAIClientBase.get_default_retry_config()

        assert result["max_attempts"] == 3
        assert result["min_wait_seconds"] == 1


class TestClientBaseConfigureRetry:
    def test_configure_retry_with_custom_config(self, concrete_client):
        concrete_client.configure_retry({"max_attempts": 10})
        assert concrete_client.retry_config == {"max_attempts": 10}

    def test_configure_retry_with_none_uses_defaults(self, concrete_client):
        from unittest.mock import MagicMock, patch

        mock_config = MagicMock()
        mock_config.retry = None

        with patch("src.config.get_config", return_value=mock_config):
            concrete_client.configure_retry(None)

        assert concrete_client.retry_config is not None
        assert concrete_client.retry_config["max_attempts"] == 3
        assert concrete_client.retry_config["min_wait_seconds"] == 1


class TestClientBaseAddToolResult:
    def test_add_tool_result_appends_to_history(self, concrete_client):
        concrete_client.add_tool_result("tc_123", "result text")
        history = concrete_client.get_conversation_history()
        assert len(history) == 1
        assert history[0] == {"role": "tool", "tool_call_id": "tc_123", "content": "result text"}
