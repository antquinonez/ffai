# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT

from __future__ import annotations

import dataclasses

import pytest

from ffai.core.memory import Entry, TurnHit


class TestTurnHitFields:
    def test_has_exactly_five_fields(self):
        fields = {f.name for f in dataclasses.fields(TurnHit)}
        assert fields == {"score", "turn", "turn_index", "text", "metadata"}

    def test_field_types(self):
        hit = TurnHit(
            score=0.5,
            turn={"role": "assistant"},
            turn_index=2,
            text="hello",
            metadata={"k": "v"},
        )
        assert isinstance(hit.score, float)
        assert isinstance(hit.turn, dict)
        assert isinstance(hit.turn_index, int)
        assert isinstance(hit.text, str)
        assert isinstance(hit.metadata, dict)


class TestTurnHitIsFrozen:
    def test_setting_field_raises_frozen_instance_error(self):
        hit = TurnHit(
            score=0.5,
            turn={},
            turn_index=0,
            text="x",
            metadata={},
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            hit.score = 0.9  # type: ignore[reportAttributeAccessIssue]

    def test_setting_metadata_raises_frozen_instance_error(self):
        hit = TurnHit(score=0.5, turn={}, turn_index=0, text="x", metadata={})
        with pytest.raises(dataclasses.FrozenInstanceError):
            hit.metadata = {"new": "value"}  # type: ignore[reportAttributeAccessIssue]


class TestEntryNamedTuple:
    def test_has_exactly_four_fields(self):
        assert Entry._fields == ("text", "embedding", "turn", "metadata")

    def test_field_access_by_name(self):
        entry = Entry(
            text="hello",
            embedding=[0.1, 0.2],
            turn={"role": "user"},
            metadata={"prompt_name": "q1"},
        )
        assert entry.text == "hello"
        assert entry.embedding == [0.1, 0.2]
        assert entry.turn == {"role": "user"}
        assert entry.metadata == {"prompt_name": "q1"}

    def test_field_access_by_index(self):
        entry = Entry(
            text="hello",
            embedding=[0.1],
            turn={"role": "user"},
            metadata={"k": "v"},
        )
        assert entry[0] == "hello"
        assert entry[1] == [0.1]
        assert entry[2] == {"role": "user"}
        assert entry[3] == {"k": "v"}

    def test_unpacks_into_four_tuple(self):
        entry = Entry(text="t", embedding=[1.0], turn={"a": 1}, metadata={"b": 2})
        text, embedding, turn, metadata = entry
        assert (text, embedding, turn, metadata) == ("t", [1.0], {"a": 1}, {"b": 2})
