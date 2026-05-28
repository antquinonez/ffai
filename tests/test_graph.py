# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT

from __future__ import annotations

import pytest

from ffai.core.execution_state import ExecutionState
from ffai.core.graph import (
    build_execution_graph,
    build_execution_graph_with_edges,
    evaluate_condition,
    evaluate_condition_with_trace,
    get_ready_prompts,
    is_abort_trigger,
)
from ffai.core.prompt_node import PromptNode


def _make_prompts():
    return [
        {"sequence": 0, "prompt_name": "a", "prompt": "first", "history": []},
        {"sequence": 1, "prompt_name": "b", "prompt": "second", "history": ["a"]},
        {"sequence": 2, "prompt_name": "c", "prompt": "third", "history": ["a", "b"]},
    ]


class TestBuildExecutionGraph:
    def test_basic_dag(self):
        prompts = _make_prompts()
        nodes = build_execution_graph(prompts)
        assert 0 in nodes
        assert 1 in nodes
        assert 2 in nodes

    def test_levels(self):
        prompts = _make_prompts()
        nodes = build_execution_graph(prompts)
        assert nodes[0].level == 0
        assert nodes[1].level == 1
        assert nodes[2].level == 2

    def test_cycle_detection_simple(self):
        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "first", "history": ["b"]},
            {"sequence": 1, "prompt_name": "b", "prompt": "second", "history": ["a"]},
        ]
        with pytest.raises(ValueError, match="Dependency cycle"):
            build_execution_graph(prompts)

    def test_cycle_detection_three_node(self):
        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "first", "history": ["c"]},
            {"sequence": 1, "prompt_name": "b", "prompt": "second", "history": ["a"]},
            {"sequence": 2, "prompt_name": "c", "prompt": "third", "history": ["b"]},
        ]
        with pytest.raises(ValueError, match="Dependency cycle"):
            build_execution_graph(prompts)

    def test_self_referencing_cycle(self):
        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "first", "history": ["a"]},
        ]
        with pytest.raises(ValueError, match="Dependency cycle"):
            build_execution_graph(prompts)

    def test_independent_prompts_same_level(self):
        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "first", "history": []},
            {"sequence": 1, "prompt_name": "b", "prompt": "second", "history": []},
            {"sequence": 2, "prompt_name": "c", "prompt": "third", "history": ["a", "b"]},
        ]
        nodes = build_execution_graph(prompts)
        assert nodes[0].level == 0
        assert nodes[1].level == 0
        assert nodes[2].level == 1


class TestBuildExecutionGraphWithEdges:
    def test_history_edges(self):
        prompts = _make_prompts()
        graph = build_execution_graph_with_edges(prompts)
        history_edges = [e for e in graph.edges if e.source == "history"]
        assert len(history_edges) == 3  # b->a, c->a, c->b

    def test_condition_creates_implicit_edge(self):
        prompts = [
            {"sequence": 0, "prompt_name": "fetch", "prompt": "get data", "history": []},
            {
                "sequence": 1,
                "prompt_name": "process",
                "prompt": "process data",
                "history": [],
                "condition": '{{fetch.status}} == "success"',
            },
        ]
        graph = build_execution_graph_with_edges(prompts)
        condition_edges = [e for e in graph.edges if e.source == "condition"]
        assert len(condition_edges) == 1
        assert condition_edges[0].from_seq == 0
        assert condition_edges[0].to_seq == 1

    def test_self_referencing_condition_excluded(self):
        prompts = [
            {
                "sequence": 0,
                "prompt_name": "check",
                "prompt": "check",
                "history": [],
                "condition": '{{check.status}} == "success"',
            },
        ]
        graph = build_execution_graph_with_edges(prompts)
        assert graph.nodes[0].dependencies == set()
        assert graph.nodes[0].level == 0


class TestGetReadyPrompts:
    def test_initial_ready(self):
        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "first", "history": []},
            {"sequence": 1, "prompt_name": "b", "prompt": "second", "history": ["a"]},
        ]
        graph = build_execution_graph_with_edges(prompts)
        state = ExecutionState()
        ready = get_ready_prompts(state, graph.nodes)
        assert len(ready) == 1
        assert ready[0].sequence == 0

    def test_after_first_completed(self):
        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "first", "history": []},
            {"sequence": 1, "prompt_name": "b", "prompt": "second", "history": ["a"]},
        ]
        graph = build_execution_graph_with_edges(prompts)
        state = ExecutionState()
        state.completed.add(0)
        ready = get_ready_prompts(state, graph.nodes)
        assert len(ready) == 1
        assert ready[0].sequence == 1

    def test_parallel_ready(self):
        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "first", "history": []},
            {"sequence": 1, "prompt_name": "b", "prompt": "second", "history": []},
            {"sequence": 2, "prompt_name": "c", "prompt": "third", "history": ["a", "b"]},
        ]
        graph = build_execution_graph_with_edges(prompts)
        state = ExecutionState()
        ready = get_ready_prompts(state, graph.nodes)
        assert len(ready) == 2


class TestEvaluateCondition:
    def test_condition_true(self):
        prompt = {"sequence": 0, "prompt_name": "step", "prompt": "go", "condition": '{{fetch.status}} == "success"'}
        results = {"fetch": {"status": "success", "response": "ok", "attempts": 1, "error": "", "has_response": True}}
        should_execute, result, error = evaluate_condition(prompt, results)
        assert should_execute is True

    def test_condition_false(self):
        prompt = {"sequence": 0, "prompt_name": "step", "prompt": "go", "condition": '{{fetch.status}} == "success"'}
        results = {"fetch": {"status": "failed", "response": "", "attempts": 1, "error": "err", "has_response": False}}
        should_execute, result, error = evaluate_condition(prompt, results)
        assert should_execute is False

    def test_no_condition(self):
        prompt = {"sequence": 0, "prompt_name": "step", "prompt": "go"}
        results = {}
        should_execute, result, error = evaluate_condition(prompt, results)
        assert should_execute is True


class TestPromptNode:
    def test_is_ready(self):
        node = PromptNode(sequence=0, prompt={"sequence": 0, "prompt_name": "a", "prompt": "go"})
        node.dependencies = {1, 2}
        assert node.is_ready({1, 2}) is True
        assert node.is_ready({1}) is False

    def test_get_prompt_name(self):
        node = PromptNode(sequence=0, prompt={"sequence": 0, "prompt_name": "a", "prompt": "go"})
        assert node.get_prompt_name() == "a"

    def test_hash_and_eq(self):
        a = PromptNode(sequence=0, prompt={"sequence": 0, "prompt_name": "a", "prompt": "go"})
        b = PromptNode(sequence=0, prompt={"sequence": 0, "prompt_name": "a", "prompt": "go"})
        assert a == b
        assert hash(a) == hash(b)


class TestAbortConditionEdges:
    def test_abort_condition_creates_edge(self):
        prompts = [
            {"sequence": 0, "prompt_name": "first", "prompt": "do stuff", "history": []},
            {
                "sequence": 1,
                "prompt_name": "second",
                "prompt": "do more",
                "history": [],
                "abort_condition": '{{first.status}} == "failed"',
            },
        ]
        graph = build_execution_graph_with_edges(prompts)
        abort_edges = [e for e in graph.edges if e.source == "abort_condition"]
        assert len(abort_edges) == 1
        assert abort_edges[0].from_seq == 0
        assert abort_edges[0].to_seq == 1
        assert abort_edges[0].condition_text == '{{first.status}} == "failed"'

    def test_abort_condition_adds_dependency(self):
        prompts = [
            {"sequence": 0, "prompt_name": "first", "prompt": "do stuff", "history": []},
            {
                "sequence": 1,
                "prompt_name": "second",
                "prompt": "do more",
                "history": [],
                "abort_condition": '{{first.status}} == "failed"',
            },
        ]
        graph = build_execution_graph_with_edges(prompts)
        assert 0 in graph.nodes[1].dependencies


class TestEvaluateConditionWithTrace:
    def test_returns_four_tuple(self):
        prompt = {
            "sequence": 0,
            "prompt_name": "step",
            "prompt": "go",
            "condition": '{{fetch.status}} == "success"',
        }
        results = {
            "fetch": {"status": "success", "response": "ok", "attempts": 1, "error": "", "has_response": True}
        }
        result = evaluate_condition_with_trace(prompt, results)
        assert len(result) == 4
        should_execute, cond_result, error, trace = result
        assert should_execute is True
        assert cond_result is True
        assert error is None
        assert trace is not None

    def test_no_condition_returns_none_trace(self):
        prompt = {"sequence": 0, "prompt_name": "step", "prompt": "go"}
        should_execute, cond_result, error, trace = evaluate_condition_with_trace(prompt, {})
        assert should_execute is True
        assert trace is None


class TestIsAbortTrigger:
    def test_abort_trace_with_success(self):
        result = {"abort_trace": "resolved condition", "status": "success"}
        assert is_abort_trigger(result) is True

    def test_no_abort_trace(self):
        result = {"status": "success"}
        assert is_abort_trigger(result) is False

    def test_abort_trace_with_failed_status(self):
        result = {"abort_trace": "resolved condition", "status": "failed"}
        assert is_abort_trigger(result) is False
