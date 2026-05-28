# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

from ffai.core.graph_execution_helpers import (
    build_graph_history_dict,
    check_abort_condition,
    resolve_graph_prompt,
    should_skip_for_failed_deps,
)
from ffai.core.prompt_node import PromptNode


class TestBuildGraphHistoryDict:
    def test_string_responses(self):
        results = {
            "a": {"status": "success", "response": "hello"},
            "b": {"status": "success", "response": "world"},
        }
        history = build_graph_history_dict(results)
        assert history == {"a": "hello", "b": "world"}

    def test_dict_response_serialized(self):
        results = {
            "eval": {"status": "success", "response": {"score": 8, "reason": "good"}},
        }
        history = build_graph_history_dict(results)
        assert "eval" in history
        assert '"score"' in history["eval"]

    def test_none_response(self):
        results = {
            "empty": {"status": "skipped", "response": None},
        }
        history = build_graph_history_dict(results)
        assert history == {"empty": ""}

    def test_empty_results(self):
        history = build_graph_history_dict({})
        assert history == {}

    def test_numeric_response(self):
        results = {"num": {"status": "success", "response": 42}}
        history = build_graph_history_dict(results)
        assert history == {"num": "42"}

    def test_missing_response_key(self):
        results = {"x": {"status": "success"}}
        history = build_graph_history_dict(results)
        assert history == {"x": ""}


class TestResolveGraphPrompt:
    def test_no_interpolation_no_history(self):
        spec = {"prompt": "What is 2+2?", "prompt_name": "math"}
        resolved, interp = resolve_graph_prompt(spec, {})
        assert resolved == "What is 2+2?"
        assert interp == set()

    def test_basic_interpolation(self):
        results = {
            "research": {"status": "success", "response": "Python is great", "prompt": "research topic"},
        }
        spec = {"prompt": "Summarize: {{research.response}}", "prompt_name": "summary"}
        resolved, interp = resolve_graph_prompt(spec, results)
        assert "Python is great" in resolved
        assert "research" in interp
        assert "{{research.response}}" not in resolved

    def test_unknown_reference_replaced_with_empty(self):
        spec = {"prompt": "Based on {{missing.response}}, elaborate", "prompt_name": "test"}
        resolved, interp = resolve_graph_prompt(spec, {})
        assert resolved == "Based on , elaborate"
        assert interp == set()

    def test_history_injection(self):
        results = {
            "step1": {
                "status": "success",
                "response": "First answer",
                "prompt": "First question",
            },
        }
        spec = {
            "prompt": "Follow up question",
            "prompt_name": "step2",
            "history": ["step1"],
        }
        resolved, interp = resolve_graph_prompt(spec, results)
        assert "<conversation_history>" in resolved
        assert "First question" in resolved
        assert "First answer" in resolved
        assert "Follow up question" in resolved
        assert interp == set()

    def test_history_deduplication_with_interpolation(self):
        results = {
            "step1": {
                "status": "success",
                "response": "Important data",
                "prompt": "Get data",
            },
        }
        spec = {
            "prompt": "Analyze {{step1.response}} in detail",
            "prompt_name": "step2",
            "history": ["step1"],
        }
        resolved, interp = resolve_graph_prompt(spec, results)
        assert "Important data" in resolved
        assert "{{step1.response}}" not in resolved
        assert "step1" in interp
        assert "<conversation_history>" not in resolved

    def test_multiple_history_entries(self):
        results = {
            "a": {"status": "success", "response": "Answer A", "prompt": "Question A"},
            "b": {"status": "success", "response": "Answer B", "prompt": "Question B"},
        }
        spec = {
            "prompt": "Synthesize",
            "prompt_name": "synth",
            "history": ["a", "b"],
        }
        resolved, interp = resolve_graph_prompt(spec, results)
        assert "<conversation_history>" in resolved
        assert "Question A" in resolved
        assert "Answer A" in resolved
        assert "Question B" in resolved
        assert "Answer B" in resolved

    def test_missing_history_name_skipped(self):
        results = {}
        spec = {
            "prompt": "Hello",
            "prompt_name": "test",
            "history": ["nonexistent"],
        }
        resolved, interp = resolve_graph_prompt(spec, results)
        assert resolved == "Hello"
        assert interp == set()

    def test_json_field_interpolation(self):
        results = {
            "eval": {
                "status": "success",
                "response": '{"score": 8, "reason": "good"}',
                "prompt": "evaluate",
            },
        }
        spec = {"prompt": "Score is {{eval.response.score}}", "prompt_name": "report"}
        resolved, interp = resolve_graph_prompt(spec, results)
        assert "8" in resolved
        assert "eval" in interp

    def test_empty_prompt(self):
        spec = {"prompt": "", "prompt_name": "empty"}
        resolved, interp = resolve_graph_prompt(spec, {"a": {"status": "success", "response": "x", "prompt": "q"}})
        assert resolved == ""
        assert interp == set()

    def test_history_strips_references_tags(self):
        results = {
            "step1": {
                "status": "success",
                "response": "Answer",
                "prompt": "<REFERENCES>ref data</REFERENCES>Actual question",
            },
        }
        spec = {
            "prompt": "Follow up",
            "prompt_name": "step2",
            "history": ["step1"],
        }
        resolved, interp = resolve_graph_prompt(spec, results)
        assert "<REFERENCES>" not in resolved
        assert "Actual question" in resolved


class TestCheckAbortCondition:
    def test_no_abort_condition(self):
        spec = {"prompt": "test", "prompt_name": "a"}
        should_abort, trace, error = check_abort_condition(spec, {})
        assert should_abort is False
        assert trace is None
        assert error is None

    def test_empty_abort_condition(self):
        spec = {"prompt": "test", "abort_condition": ""}
        should_abort, trace, error = check_abort_condition(spec, {})
        assert should_abort is False

    def test_abort_triggered(self):
        spec = {
            "prompt": "test",
            "prompt_name": "check",
            "abort_condition": '{{fetch.status}} == "failed"',
        }
        results = {
            "fetch": {"status": "failed", "response": "", "attempts": 1, "error": "timeout", "has_response": False},
        }
        should_abort, trace, error = check_abort_condition(spec, results)
        assert should_abort is True
        assert trace is not None
        assert error is None

    def test_abort_not_triggered(self):
        spec = {
            "prompt": "test",
            "prompt_name": "check",
            "abort_condition": '{{fetch.status}} == "failed"',
        }
        results = {
            "fetch": {"status": "success", "response": "data", "attempts": 1, "error": "", "has_response": True},
        }
        should_abort, trace, error = check_abort_condition(spec, results)
        assert should_abort is False
        assert trace is None
        assert error is None

    def test_abort_condition_error(self):
        spec = {
            "prompt": "test",
            "prompt_name": "check",
            "abort_condition": '{{nonexistent.status}} == "failed"',
        }
        should_abort, trace, error = check_abort_condition(spec, {})
        assert should_abort is False
        assert error is not None


class TestShouldSkipForFailedDeps:
    def _make_node(self, seq: int, name: str, dep_seqs: set[int]) -> PromptNode:
        node = PromptNode(
            sequence=seq,
            prompt={"sequence": seq, "prompt_name": name, "prompt": "test"},
        )
        node.dependencies = dep_seqs
        return node

    def test_no_deps(self):
        node = self._make_node(0, "a", set())
        nodes = {0: node}
        skip, reason = should_skip_for_failed_deps(node, {}, nodes)
        assert skip is False
        assert reason == ""

    def test_all_deps_succeeded(self):
        node_a = self._make_node(0, "a", set())
        node_b = self._make_node(1, "b", {0})
        nodes = {0: node_a, 1: node_b}
        results = {"a": {"status": "success", "response": "ok"}}
        skip, reason = should_skip_for_failed_deps(node_b, results, nodes)
        assert skip is False
        assert reason == ""

    def test_dep_failed(self):
        node_a = self._make_node(0, "a", set())
        node_b = self._make_node(1, "b", {0})
        nodes = {0: node_a, 1: node_b}
        results = {"a": {"status": "failed", "response": "", "error": "boom"}}
        skip, reason = should_skip_for_failed_deps(node_b, results, nodes)
        assert skip is True
        assert "a" in reason

    def test_skipped_dep_does_not_cascade(self):
        node_a = self._make_node(0, "a", set())
        node_b = self._make_node(1, "b", {0})
        nodes = {0: node_a, 1: node_b}
        results = {"a": {"status": "skipped", "response": ""}}
        skip, reason = should_skip_for_failed_deps(node_b, results, nodes)
        assert skip is False
        assert reason == ""

    def test_multiple_deps_one_failed(self):
        node_a = self._make_node(0, "a", set())
        node_b = self._make_node(1, "b", set())
        node_c = self._make_node(2, "c", {0, 1})
        nodes = {0: node_a, 1: node_b, 2: node_c}
        results = {
            "a": {"status": "success", "response": "ok"},
            "b": {"status": "failed", "response": "", "error": "err"},
        }
        skip, reason = should_skip_for_failed_deps(node_c, results, nodes)
        assert skip is True
        assert "b" in reason

    def test_unnamed_dep(self):
        node_a = PromptNode(
            sequence=0,
            prompt={"sequence": 0, "prompt": "unnamed"},
        )
        node_b = self._make_node(1, "b", {0})
        nodes = {0: node_a, 1: node_b}
        results = {"0": {"status": "failed", "response": ""}}
        skip, reason = should_skip_for_failed_deps(node_b, results, nodes)
        assert skip is True
        assert "0" in reason

    def test_dep_not_in_results(self):
        node_a = self._make_node(0, "a", set())
        node_b = self._make_node(1, "b", {0})
        nodes = {0: node_a, 1: node_b}
        skip, reason = should_skip_for_failed_deps(node_b, {}, nodes)
        assert skip is False
        assert reason == ""
