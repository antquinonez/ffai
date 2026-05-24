# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT

from __future__ import annotations

import time

from src.core.history.permanent import PermanentHistory


class TestPermanentHistoryUserTurnCoalescing:
    def test_consecutive_user_turns_coalesce(self):
        ph = PermanentHistory()
        ph.add_turn_user("first")
        ph.add_turn_user("second")
        turns = ph.get_all_turns()
        assert len(turns) == 1
        assert turns[0]["content"][0]["text"] == "first\nsecond"

    def test_user_assistant_user_no_coalesce(self):
        ph = PermanentHistory()
        ph.add_turn_user("q1")
        ph.add_turn_assistant("a1")
        ph.add_turn_user("q2")
        turns = ph.get_all_turns()
        assert len(turns) == 3

    def test_coalesced_timestamp_updated(self):
        ph = PermanentHistory()
        ph.add_turn_user("first")
        t1 = ph.turns[-1]["timestamp"]
        time.sleep(0.01)
        ph.add_turn_user("second")
        t2 = ph.turns[-1]["timestamp"]
        assert t2 > t1


class TestPermanentHistoryGetAllTurns:
    def test_returns_deep_copy(self):
        ph = PermanentHistory()
        ph.add_turn_user("original")
        turns = ph.get_all_turns()
        turns[0]["content"][0]["text"] = "modified"
        assert ph.turns[0]["content"][0]["text"] == "original"

    def test_empty_returns_empty_list(self):
        ph = PermanentHistory()
        assert ph.get_all_turns() == []


class TestPermanentHistoryGetTurnsSince:
    def test_returns_only_later_turns(self):
        ph = PermanentHistory()
        ph.add_turn_user("before")
        ph.add_turn_assistant("reply")
        t = time.time()
        time.sleep(0.01)
        ph.add_turn_user("after")
        turns = ph.get_turns_since(t)
        assert len(turns) == 1
        assert turns[0]["content"][0]["text"] == "after"

    def test_returns_all_if_all_after(self):
        ph = PermanentHistory()
        t_before = time.time()
        time.sleep(0.01)
        ph.add_turn_user("a")
        ph.add_turn_assistant("b")
        turns = ph.get_turns_since(t_before)
        assert len(turns) == 2
