# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

import time

from src.core.history.ordered import OrderedPromptHistory
from src.core.history.permanent import PermanentHistory
from src.core.history.recorder import HistoryRecorder
from src.core.response_context import ResponseContext


class TestHistoryRecorderRecord:
    def test_records_to_all_five_stores(self):
        context = ResponseContext()
        permanent = PermanentHistory()
        ordered = OrderedPromptHistory()
        recorder = HistoryRecorder(context=context, permanent_history=permanent, ordered_history=ordered)

        recorder.record(
            prompt="What is 2+2?",
            response="4",
            model="test-model",
            prompt_name="math",
            history=["greeting"],
            status="success",
        )

        assert len(recorder.history) == 1
        assert len(recorder.clean_history) == 1
        assert len(permanent.turns) == 2
        assert permanent.turns[0]["role"] == "user"
        assert permanent.turns[1]["role"] == "assistant"
        assert permanent.turns[1]["content"][0]["text"] == "4"
        assert len(context.prompt_attr_history) == 1
        assert context.prompt_attr_history[0]["prompt_name"] == "math"
        interactions = ordered.get_all_interactions()
        assert len(interactions) == 1
        assert interactions[0].prompt_name == "math"
        assert interactions[0].response == "4"

    def test_interaction_dict_has_expected_keys(self):
        context = ResponseContext()
        recorder = HistoryRecorder(
            context=context,
            permanent_history=PermanentHistory(),
            ordered_history=OrderedPromptHistory(),
        )

        before = time.time()
        recorder.record(
            prompt="Hello",
            response="Hi there",
            model="test-model",
            prompt_name="greeting",
            status="success",
        )
        after = time.time()

        entry = recorder.history[0]
        assert entry["prompt"] == "Hello"
        assert entry["response"] == "Hi there"
        assert entry["prompt_name"] == "greeting"
        assert entry["model"] == "test-model"
        assert entry["status"] == "success"
        assert before <= entry["timestamp"] <= after
        assert entry["history"] is None

    def test_history_and_clean_history_share_same_dict(self):
        context = ResponseContext()
        recorder = HistoryRecorder(
            context=context,
            permanent_history=PermanentHistory(),
            ordered_history=OrderedPromptHistory(),
        )

        recorder.record(prompt="p", response="r", model="m")

        assert recorder.history[0] is recorder.clean_history[0]

    def test_dict_response_creates_per_key_prompt_attr_entries(self):
        context = ResponseContext()
        recorder = HistoryRecorder(
            context=context,
            permanent_history=PermanentHistory(),
            ordered_history=OrderedPromptHistory(),
        )

        recorder.record(
            prompt="analyze",
            response={"sentiment": "positive", "score": 0.95},
            model="test-model",
            prompt_name="analysis",
        )

        assert len(recorder.history) == 1
        assert recorder.history[0]["response"] == {"sentiment": "positive", "score": 0.95}
        assert len(context.prompt_attr_history) == 2
        assert context.prompt_attr_history[0]["prompt"] == "sentiment"
        assert context.prompt_attr_history[0]["response"] == "positive"
        assert context.prompt_attr_history[1]["prompt"] == "score"
        assert context.prompt_attr_history[1]["response"] == 0.95

    def test_record_with_status_skipped(self):
        context = ResponseContext()
        ordered = OrderedPromptHistory()
        recorder = HistoryRecorder(
            context=context,
            permanent_history=PermanentHistory(),
            ordered_history=ordered,
        )

        recorder.record(
            prompt="Skip me",
            response=None,
            model="test-model",
            prompt_name="skipper",
            status="skipped",
        )

        assert recorder.history[0]["status"] == "skipped"
        assert recorder.history[0]["response"] is None
        interactions = ordered.get_all_interactions()
        assert len(interactions) == 1
        assert interactions[0].response == ""

    def test_multiple_records_accumulate(self):
        context = ResponseContext()
        recorder = HistoryRecorder(
            context=context,
            permanent_history=PermanentHistory(),
            ordered_history=OrderedPromptHistory(),
        )

        recorder.record(prompt="p1", response="r1", model="m", prompt_name="a")
        recorder.record(prompt="p2", response="r2", model="m", prompt_name="b")

        assert len(recorder.history) == 2
        assert len(recorder.clean_history) == 2
        assert recorder.history[0]["prompt_name"] == "a"
        assert recorder.history[1]["prompt_name"] == "b"

    def test_default_status_is_success(self):
        context = ResponseContext()
        recorder = HistoryRecorder(
            context=context,
            permanent_history=PermanentHistory(),
            ordered_history=OrderedPromptHistory(),
        )

        recorder.record(prompt="p", response="r", model="m")
        assert recorder.history[0]["status"] == "success"

    def test_default_history_is_none(self):
        context = ResponseContext()
        recorder = HistoryRecorder(
            context=context,
            permanent_history=PermanentHistory(),
            ordered_history=OrderedPromptHistory(),
        )

        recorder.record(prompt="p", response="r", model="m")
        assert recorder.history[0]["history"] is None

    def test_history_parameter_stored(self):
        context = ResponseContext()
        recorder = HistoryRecorder(
            context=context,
            permanent_history=PermanentHistory(),
            ordered_history=OrderedPromptHistory(),
        )

        recorder.record(
            prompt="p", response="r", model="m",
            history=["a", "b"],
        )
        assert recorder.history[0]["history"] == ["a", "b"]
