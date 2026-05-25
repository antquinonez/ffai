# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Tests validating the history-restore fix and unified retry mechanism.

1. History-restore bug: ``FFLiteLLMClient.get_conversation_history()`` returns
   a *copy*, so the old ``append()``-on-copy pattern was a silent no-op.
   The fix uses ``set_conversation_history(saved + new)`` instead.

2. Duplicate retry: ``FFLiteLLMClient`` had a manual retry loop *and*
   ``litellm.num_retries`` set globally.  The fix delegates retry to the
   shared ``@get_configured_retry_decorator()`` from ``retry_utils`` and
   disables LiteLLM's internal retries.
"""

from unittest.mock import MagicMock, patch

import pytest

from src.Clients.FFLiteLLMClient import FFLiteLLMClient
from src.FFAI import FFAI


class TestHistoryRestoreWithCopyReturningClient:
    """Validate that client history is correctly restored even when
    ``get_conversation_history()`` returns a defensive copy (like FFLiteLLMClient).
    """

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_litellm_client_history_restored_after_declarative_call(self, mock_completion):
        """FFLiteLLMClient.get_conversation_history() returns a copy.
        Verify FFAI's suspend/restore still merges new messages back.
        """
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = None
        mock_completion.return_value = mock_response

        client = FFLiteLLMClient(model_string="openai/gpt-4")
        ffai = FFAI(client)

        # Build up client history with normal calls
        ffai.generate_response("Question A", prompt_name="q1")
        ffai.generate_response("Question B", prompt_name="q2")
        pre_history_len = len(client.conversation_history)
        assert pre_history_len == 4  # 2 user + 2 assistant

        # Use declarative context -- triggers suspend/restore
        ffai.generate_response("Question C", prompt_name="q3", history=["q1", "q2"])

        # After restore: original 4 + 2 new (user + assistant)
        assert len(client.conversation_history) == 6

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_litellm_client_original_history_preserved_after_restore(
        self, mock_completion
    ):
        """Original client history messages must appear in order before new ones."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Answer"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = None
        mock_completion.return_value = mock_response

        client = FFLiteLLMClient(model_string="openai/gpt-4")
        ffai = FFAI(client)

        ffai.generate_response("First", prompt_name="q1")
        original = client.get_conversation_history().copy()

        ffai.generate_response("Second", prompt_name="q2", history=["q1"])

        # First 2 messages should be the originals
        assert client.conversation_history[0] == original[0]
        assert client.conversation_history[1] == original[1]

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_litellm_client_interpolation_suspends_and_restores(self, mock_completion):
        """{{name.response}} interpolation also triggers suspend/restore."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Python is a language"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = None
        mock_completion.return_value = mock_response

        client = FFLiteLLMClient(model_string="openai/gpt-4")
        ffai = FFAI(client)

        ffai.generate_response("What is Python?", prompt_name="python_q")
        assert len(client.conversation_history) == 2

        ffai.generate_response(
            "Tell me more about {{python_q.response}}",
            prompt_name="followup",
        )

        assert len(client.conversation_history) == 4

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_litellm_client_get_history_returns_copy(self, mock_completion):
        """Confirm the precondition: get_conversation_history returns a copy,
        so appending to the return value does NOT modify internal state.
        """
        client = FFLiteLLMClient(model_string="openai/gpt-4")
        client.conversation_history.append({"role": "user", "content": "hi"})

        returned = client.get_conversation_history()
        returned.append({"role": "assistant", "content": "should not appear"})

        assert len(client.conversation_history) == 1

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_litellm_client_mixed_declarative_and_normal(self, mock_completion):
        """Mix of declarative and normal calls preserves full history."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = None
        mock_completion.return_value = mock_response

        client = FFLiteLLMClient(model_string="openai/gpt-4")
        ffai = FFAI(client)

        ffai.generate_response("N1", prompt_name="n1")
        assert len(client.conversation_history) == 2

        ffai.generate_response("D1", prompt_name="d1", history=["n1"])
        assert len(client.conversation_history) == 4

        ffai.generate_response("N2", prompt_name="n2")
        assert len(client.conversation_history) == 6

        ffai.generate_response("D2", prompt_name="d2", history=["n2"])
        assert len(client.conversation_history) == 8

        # All 8 messages should be in order
        roles = [m["role"] for m in client.conversation_history]
        assert roles == ["user", "assistant"] * 4


class TestFFLiteLLMClientRetryUsesSharedMechanism:
    """Validate that FFLiteLLMClient delegates retry to retry_utils."""

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_retryable_exception_retried(self, mock_completion):
        """A retryable exception should trigger retries via tenacity."""
        from src.retry_utils import RateLimitError

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Success"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = None

        mock_completion.side_effect = [
            RateLimitError("rate limited"),
            mock_response,
        ]

        client = FFLiteLLMClient(model_string="openai/gpt-4")
        response = client.generate_response("Hello")

        assert response == "Success"
        assert mock_completion.call_count == 2

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_non_retryable_exception_not_retried(self, mock_completion):
        """A non-retryable exception should NOT be retried."""
        mock_completion.side_effect = ValueError("bad input")

        client = FFLiteLLMClient(model_string="openai/gpt-4")

        with pytest.raises(ValueError, match="bad input"):
            client.generate_response("Hello")

        assert mock_completion.call_count == 1

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_all_retries_exhausted_raises(self, mock_completion):
        """When all retries are exhausted, the original exception is raised."""
        from src.retry_utils import RateLimitError

        mock_completion.side_effect = RateLimitError("still rate limited")

        client = FFLiteLLMClient(model_string="openai/gpt-4")

        with pytest.raises(RateLimitError):
            client.generate_response("Hello")

        assert mock_completion.call_count == 3

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_fallback_after_retries_exhausted(self, mock_completion):
        """Fallbacks should be tried after primary retries are exhausted."""

        mock_fallback_response = MagicMock()
        mock_fallback_response.choices = [MagicMock()]
        mock_fallback_response.choices[0].message.content = "Fallback OK"
        mock_fallback_response.choices[0].message.tool_calls = None
        mock_fallback_response.usage = None

        call_count = 0

        def side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if kwargs.get("model") == "openai/gpt-4":
                raise RateLimitError("primary rate limited")
            return mock_fallback_response

        from src.retry_utils import RateLimitError

        mock_completion.side_effect = side_effect

        client = FFLiteLLMClient(
            model_string="openai/gpt-4",
            fallbacks=["anthropic/claude-3-opus"],
        )
        response = client.generate_response("Hello")

        assert response == "Fallback OK"
        assert call_count == 4  # 3 primary retries + 1 fallback

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_litellm_internal_retries_disabled(self, mock_completion):
        """LiteLLM's own retry counter should be 0 (we use our own)."""
        client = FFLiteLLMClient(model_string="openai/gpt-4")

        import litellm

        assert litellm.num_retries == 0

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_call_primary_is_decorated(self, mock_completion):
        """_call_primary should have tenacity retry metadata."""
        client = FFLiteLLMClient(model_string="openai/gpt-4")

        assert hasattr(client._call_primary, "retry")

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_retry_does_not_corrupt_history(self, mock_completion):
        """Failed attempts should not leave partial history entries."""
        from src.retry_utils import RateLimitError

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OK"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = None

        mock_completion.side_effect = [
            RateLimitError("retry me"),
            mock_response,
        ]

        client = FFLiteLLMClient(model_string="openai/gpt-4")
        client.generate_response("Hello")

        # Only the successful attempt should have appended to history
        assert len(client.conversation_history) == 2
        assert client.conversation_history[0]["content"] == "Hello"
        assert client.conversation_history[1]["content"] == "OK"


class TestFFLiteLLMClientFallbackAfterRetry:
    """Validate fallback behavior with the new retry mechanism."""

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_all_models_fail_raises_runtime_error(self, mock_completion):
        """When primary + all fallbacks fail, raise RuntimeError."""
        from src.retry_utils import RateLimitError

        mock_completion.side_effect = RateLimitError("everything fails")

        client = FFLiteLLMClient(
            model_string="openai/gpt-4",
            fallbacks=["anthropic/claude-3-opus"],
        )

        with pytest.raises(RuntimeError, match="All models failed"):
            client.generate_response("Hello")

    @patch("src.Clients.FFLiteLLMClient.completion")
    def test_first_fallback_succeeds(self, mock_completion):
        """First fallback model succeeds."""
        from src.retry_utils import RateLimitError

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Claude response"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage = None

        def side_effect(**kwargs):
            if "anthropic" in kwargs.get("model", ""):
                return mock_response
            raise RateLimitError("primary down")

        mock_completion.side_effect = side_effect

        client = FFLiteLLMClient(
            model_string="openai/gpt-4",
            fallbacks=["anthropic/claude-3-opus"],
        )

        response = client.generate_response("Hello")
        assert response == "Claude response"
