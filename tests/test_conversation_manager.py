# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

import threading
from typing import Any
from unittest.mock import MagicMock

from ffai.core.conversation_manager import ConversationManager


def _make_client(history: list[dict[str, Any]] | None = None) -> MagicMock:
    client = MagicMock()
    client.get_conversation_history.return_value = list(history or [])
    client.set_conversation_history = MagicMock()
    client.clear_conversation = MagicMock()
    return client


class TestShouldSuspend:
    def test_returns_true_when_history_provided(self):
        cm = ConversationManager(client=_make_client())
        assert cm.should_suspend("Hello", history=["math"]) is True

    def test_returns_true_when_empty_history_list(self):
        cm = ConversationManager(client=_make_client())
        assert cm.should_suspend("Hello", history=[]) is True

    def test_returns_true_when_prompt_has_interpolation(self):
        cm = ConversationManager(client=_make_client())
        assert cm.should_suspend("{{step1.response}} then what?", history=None) is True

    def test_returns_false_when_no_history_no_interpolation(self):
        cm = ConversationManager(client=_make_client())
        assert cm.should_suspend("Plain prompt", history=None) is False


class TestSuspendRestore:
    def test_suspend_saves_and_clears_history(self):
        original = [{"role": "user", "content": "hi"}]
        client = _make_client(history=original)
        cm = ConversationManager(client=client)

        saved = cm.suspend(reason="test")

        assert saved == original
        client.set_conversation_history.assert_called_with([])

    def test_restore_merges_new_messages(self):
        client = _make_client()
        cm = ConversationManager(client=client)

        saved = [{"role": "user", "content": "old"}]
        client.get_conversation_history.return_value = [
            {"role": "assistant", "content": "new"}
        ]

        cm.restore(saved)

        client.set_conversation_history.assert_called_once_with([
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "new"},
        ])

    def test_restore_none_is_noop(self):
        client = _make_client()
        cm = ConversationManager(client=client)
        cm.restore(None)
        client.set_conversation_history.assert_not_called()

    def test_suspend_returns_none_when_client_lacks_methods(self):
        client = MagicMock(spec=[])
        cm = ConversationManager(client=client)
        assert cm.suspend() is None

    def test_full_suspend_restore_cycle(self):
        original = [{"role": "user", "content": "before"}]
        client = _make_client(history=original)
        cm = ConversationManager(client=client)

        saved = cm.suspend(reason="test")
        assert saved == original

        client.get_conversation_history.return_value = [
            {"role": "assistant", "content": "during"}
        ]
        cm.restore(saved)

        client.set_conversation_history.assert_called_with([
            {"role": "user", "content": "before"},
            {"role": "assistant", "content": "during"},
        ])


class TestGetSetHistory:
    def test_get_history_returns_client_history(self):
        history = [{"role": "user", "content": "hello"}]
        client = _make_client(history=history)
        cm = ConversationManager(client=client)

        result = cm.get_history()
        assert result == history

    def test_get_history_returns_empty_when_no_method(self):
        client = MagicMock(spec=[])
        cm = ConversationManager(client=client)
        assert cm.get_history() == []

    def test_get_history_returns_empty_on_exception(self):
        client = MagicMock()
        client.get_conversation_history.side_effect = RuntimeError("boom")
        cm = ConversationManager(client=client)
        assert cm.get_history() == []

    def test_set_history_returns_true_on_success(self):
        client = _make_client()
        cm = ConversationManager(client=client)

        result = cm.set_history([{"role": "user", "content": "x"}])
        assert result is True
        client.set_conversation_history.assert_called_once()

    def test_set_history_returns_false_when_no_method(self):
        client = MagicMock(spec=[])
        cm = ConversationManager(client=client)
        assert cm.set_history([]) is False

    def test_set_history_returns_false_on_exception(self):
        client = MagicMock()
        client.set_conversation_history.side_effect = RuntimeError("boom")
        cm = ConversationManager(client=client)
        assert cm.set_history([]) is False


class TestAddMessage:
    def test_add_message_appends_to_history(self):
        client = _make_client(history=[{"role": "user", "content": "old"}])
        cm = ConversationManager(client=client)

        result = cm.add_message("assistant", "new")
        assert result is True

        call_args = client.set_conversation_history.call_args[0][0]
        assert len(call_args) == 2
        assert call_args[1] == {"role": "assistant", "content": "new"}

    def test_add_message_with_extra_kwargs(self):
        client = _make_client(history=[])
        cm = ConversationManager(client=client)

        cm.add_message("tool", "result", tool_call_id="tc-123")
        call_args = client.set_conversation_history.call_args[0][0]
        assert call_args[0]["tool_call_id"] == "tc-123"

    def test_add_message_returns_false_on_exception(self):
        client = MagicMock(spec=[])
        cm = ConversationManager(client=client)
        assert cm.add_message("user", "hi") is False


class TestClear:
    def test_clear_delegates_to_client(self):
        client = _make_client()
        cm = ConversationManager(client=client)
        cm.clear()
        client.clear_conversation.assert_called_once()

    def test_clear_safe_when_no_method(self):
        client = MagicMock(spec=[])
        cm = ConversationManager(client=client)
        cm.clear()


class TestClientProperty:
    def test_client_getter(self):
        client = _make_client()
        cm = ConversationManager(client=client)
        assert cm.client is client

    def test_client_setter(self):
        client1 = _make_client()
        client2 = _make_client()
        cm = ConversationManager(client=client1)
        cm.client = client2
        assert cm.client is client2


class TestThreadSafety:
    def test_suspend_uses_lock(self):
        lock = threading.Lock()
        client = _make_client(history=[{"role": "user", "content": "x"}])
        cm = ConversationManager(client=client, lock=lock)

        cm.suspend()
        client.set_conversation_history.assert_called_with([])

    def test_restore_uses_lock(self):
        lock = threading.Lock()
        client = _make_client()
        cm = ConversationManager(client=client, lock=lock)

        client.get_conversation_history.return_value = []
        cm.restore([{"role": "user", "content": "saved"}])
        client.set_conversation_history.assert_called_once()
