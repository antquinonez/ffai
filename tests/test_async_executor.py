# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

import asyncio

import pytest

from src.core.async_client_base import AsyncFFAIClientBase
from src.core.async_executor import AsyncGraphExecutor, GraphResult
from src.core.response_result import ResponseResult


class _MockAsyncClient(AsyncFFAIClientBase):
    model = "test-model"
    system_instructions = "test"

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or ["default response"]
        self._call_count = 0
        self.conversation_history: list[dict] = []

    async def generate_response(self, prompt: str, **kwargs):
        idx = min(self._call_count, len(self._responses) - 1)
        response = self._responses[idx]
        self._call_count += 1
        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": response})
        return response


    def clear_conversation(self):
        self.conversation_history = []

    def get_conversation_history(self):
        return list(self.conversation_history)

    def set_conversation_history(self, history):
        self.conversation_history = list(history)

    async def clone(self):
        cloned = _MockAsyncClient(self._responses)
        return cloned


def _make_executor(responses: list[str] | None = None):
    client = _MockAsyncClient(responses)

    async def run_prompt(**kwargs):
        cloned = await client.clone()
        prompt = kwargs.get("prompt", "")
        response = await cloned.generate_response(prompt=prompt)
        return ResponseResult(
            response=response,
            model=cloned.model,
            status="success",
        )

    return AsyncGraphExecutor(executor_fn=run_prompt), client


class TestGraphResult:
    def test_defaults(self):
        result = GraphResult()
        assert result.results == {}
        assert result.success_count == 0
        assert result.failed_count == 0
        assert result.skipped_count == 0
        assert result.aborted is False


class TestAsyncGraphExecutorBasic:
    def test_single_prompt(self):
        executor, _ = _make_executor(["hello"])

        prompts = [
            {"sequence": 0, "prompt_name": "p1", "prompt": "say hello"},
        ]

        result = asyncio.run(executor.execute(prompts))
        assert "p1" in result.results
        assert result.results["p1"].status == "success"
        assert result.results["p1"].response == "hello"
        assert result.success_count == 1

    def test_sequential_levels(self):
        executor, _ = _make_executor(["alpha", "beta"])

        prompts = [
            {"sequence": 0, "prompt_name": "first", "prompt": "step 1"},
            {"sequence": 1, "prompt_name": "second", "prompt": "step 2", "history": ["first"]},
        ]

        result = asyncio.run(executor.execute(prompts))
        assert result.results["first"].status == "success"
        assert result.results["second"].status == "success"
        assert result.success_count == 2

    def test_concurrent_same_level(self):
        call_order: list[str] = []
        client = _MockAsyncClient(["a", "b", "c"])

        async def run_prompt(**kwargs):
            name = kwargs.get("prompt_name", "")
            call_order.append(name)
            cloned = await client.clone()
            prompt = kwargs.get("prompt", "")
            response = await cloned.generate_response(prompt=prompt)
            return ResponseResult(response=response, model=cloned.model, status="success")

        executor = AsyncGraphExecutor(executor_fn=run_prompt)

        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "p1"},
            {"sequence": 1, "prompt_name": "b", "prompt": "p2"},
            {"sequence": 2, "prompt_name": "c", "prompt": "p3"},
        ]

        result = asyncio.run(executor.execute(prompts))
        assert result.success_count == 3
        assert set(call_order) == {"a", "b", "c"}

    def test_diamond_dag(self):
        executor, _ = _make_executor(["A", "B", "C", "D"])

        prompts = [
            {"sequence": 0, "prompt_name": "A", "prompt": "root"},
            {"sequence": 1, "prompt_name": "B", "prompt": "left", "history": ["A"]},
            {"sequence": 2, "prompt_name": "C", "prompt": "right", "history": ["A"]},
            {"sequence": 3, "prompt_name": "D", "prompt": "merge", "history": ["B", "C"]},
        ]

        result = asyncio.run(executor.execute(prompts))
        assert result.success_count == 4
        assert all(r.status == "success" for r in result.results.values())


class TestAsyncGraphExecutorConditions:
    def test_condition_skips_node(self):
        executor, _ = _make_executor(["data"])

        async def run_prompt(**kwargs):
            name = kwargs.get("prompt_name", "")
            if name == "setup":
                return ResponseResult(response="setup data", model="test", status="success")
            return ResponseResult(response="should be skipped", model="test", status="success")

        prompts = [
            {"prompt_name": "setup", "prompt": "init", "sequence": 0},
            {
                "prompt_name": "check",
                "prompt": "check",
                "sequence": 1,
                "history": ["setup"],
                "condition": 'len({{setup.response}}) > 100',
            },
        ]

        executor = AsyncGraphExecutor(executor_fn=run_prompt)
        result = asyncio.run(executor.execute(prompts))
        assert result.results["check"].status == "skipped"
        assert result.skipped_count == 1

    def test_condition_passes(self):
        async def run_prompt(**kwargs):
            name = kwargs.get("prompt_name", "")
            if name == "setup":
                return ResponseResult(
                    response="a" * 200, model="test", status="success"
                )
            return ResponseResult(response="proceeded", model="test", status="success")

        prompts = [
            {"prompt_name": "setup", "prompt": "init", "sequence": 0},
            {
                "prompt_name": "check",
                "prompt": "check",
                "sequence": 1,
                "history": ["setup"],
                "condition": 'len({{setup.response}}) > 100',
            },
        ]

        executor = AsyncGraphExecutor(executor_fn=run_prompt)
        result = asyncio.run(executor.execute(prompts))
        assert result.results["check"].status == "success"
        assert result.success_count == 2


class TestAsyncGraphExecutorErrors:
    def test_exception_caught(self):
        async def run_prompt(**kwargs):
            raise RuntimeError("API error")

        prompts = [
            {"sequence": 0, "prompt_name": "fail", "prompt": "boom"},
        ]

        executor = AsyncGraphExecutor(executor_fn=run_prompt)
        result = asyncio.run(executor.execute(prompts))
        assert result.results["fail"].status == "failed"
        assert "API error" in (result.results["fail"].condition_error or "")
        assert result.failed_count == 1

    def test_mixed_success_failure(self):
        async def run_prompt(**kwargs):
            name = kwargs.get("prompt_name", "")
            if name == "bad":
                raise ValueError("bad prompt")
            return ResponseResult(response="ok", model="test", status="success")

        prompts = [
            {"sequence": 0, "prompt_name": "good", "prompt": "ok"},
            {"sequence": 1, "prompt_name": "bad", "prompt": "fail"},
        ]

        executor = AsyncGraphExecutor(executor_fn=run_prompt)
        result = asyncio.run(executor.execute(prompts))
        assert result.success_count == 1
        assert result.failed_count == 1


class TestAsyncGraphExecutorConcurrency:
    def test_max_concurrency_respected(self):
        concurrent_count = 0
        max_seen = 0

        async def run_prompt(**kwargs):
            nonlocal concurrent_count, max_seen
            concurrent_count += 1
            max_seen = max(max_seen, concurrent_count)
            await asyncio.sleep(0.05)
            concurrent_count -= 1
            return ResponseResult(response="done", model="test", status="success")

        prompts = [
            {"sequence": i, "prompt_name": f"p{i}", "prompt": f"prompt {i}"} for i in range(5)
        ]

        executor = AsyncGraphExecutor(executor_fn=run_prompt, max_concurrency=2)
        result = asyncio.run(executor.execute(prompts))
        assert result.success_count == 5
        assert max_seen <= 2


class TestAsyncGraphExecutorCycle:
    def test_cycle_raises_value_error(self):
        async def run_prompt(**kwargs):
            return ResponseResult(response="ok", model="test", status="success")

        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "a", "history": ["b"]},
            {"sequence": 1, "prompt_name": "b", "prompt": "b", "history": ["a"]},
        ]

        executor = AsyncGraphExecutor(executor_fn=run_prompt)
        with pytest.raises(ValueError, match="cycle"):
            asyncio.run(executor.execute(prompts))


class TestAsyncClientBaseABC:
    def test_cannot_instantiate_directly(self):
        with pytest.raises(TypeError):
            AsyncFFAIClientBase()  # type: ignore[reportAbstractUsage]

    def test_subclass_must_implement_async_methods(self):
        class IncompleteClient(AsyncFFAIClientBase):
            model = "test"
            system_instructions = ""

            def clear_conversation(self):
                pass

            def get_conversation_history(self):
                return []

            def set_conversation_history(self, history):
                pass

        with pytest.raises(TypeError):
            IncompleteClient()  # type: ignore[reportAbstractUsage]

    def test_complete_subclass_instantiates(self):
        client = _MockAsyncClient()
        assert isinstance(client, AsyncFFAIClientBase)


class TestFFAIExecuteGraph:
    """Integration tests for FFAI.execute_graph()."""

    def test_execute_graph_with_async_client(self):
        from src.FFAI import FFAI

        client = _MockAsyncClient(["A response", "B response"])
        ffai = FFAI(client)

        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "prompt A"},
            {"sequence": 1, "prompt_name": "b", "prompt": "prompt B", "history": ["a"]},
        ]

        result = asyncio.run(ffai.execute_graph(prompts))
        assert "a" in result.results
        assert "b" in result.results
        assert result.results["a"].status == "success"
        assert result.results["b"].status == "success"

    def test_execute_graph_rejects_sync_client(self, mock_ffmistralsmall):
        from src.FFAI import FFAI

        ffai = FFAI(mock_ffmistralsmall)
        prompts = [{"sequence": 0, "prompt_name": "x", "prompt": "test"}]

        with pytest.raises(TypeError, match="async client"):
            asyncio.run(ffai.execute_graph(prompts))

    def test_execute_graph_records_to_context(self):
        from src.FFAI import FFAI

        client = _MockAsyncClient(["hello world"])
        ffai = FFAI(client)

        prompts = [
            {"sequence": 0, "prompt_name": "greet", "prompt": "say hello"},
        ]

        result = asyncio.run(ffai.execute_graph(prompts))
        assert result.success_count == 1

        pah = ffai.prompt_attr_history
        assert any(e.get("prompt_name") == "greet" for e in pah)

    def test_execute_graph_concurrent_nodes(self):
        from src.FFAI import FFAI

        client = _MockAsyncClient(["A", "B", "C"])
        ffai = FFAI(client)

        prompts = [
            {"sequence": 0, "prompt_name": "x", "prompt": "x"},
            {"sequence": 1, "prompt_name": "y", "prompt": "y"},
            {"sequence": 2, "prompt_name": "z", "prompt": "z", "history": ["x", "y"]},
        ]

        result = asyncio.run(ffai.execute_graph(prompts))
        assert result.success_count == 3
        assert result.results["z"].status == "success"

    def test_execute_graph_failed_not_recorded_to_context(self):
        from src.FFAI import FFAI

        async def failing_then_succeeding(**kwargs):
            name = kwargs.get("prompt_name", "")
            if name == "bad":
                raise ValueError("boom")
            return ResponseResult(response="ok", model="test", status="success")

        client = _MockAsyncClient()
        ffai = FFAI(client)

        prompts = [
            {"sequence": 0, "prompt_name": "good", "prompt": "ok"},
            {"sequence": 1, "prompt_name": "bad", "prompt": "fail"},
        ]

        executor = AsyncGraphExecutor(executor_fn=failing_then_succeeding)
        result = asyncio.run(executor.execute(prompts))

        assert result.results["bad"].status == "failed"
        assert result.results["good"].status == "success"

    def test_execute_graph_cleans_response(self):
        from src.FFAI import FFAI

        client = _MockAsyncClient(['<think reasoning</think >clean answer'])
        ffai = FFAI(client)

        prompts = [
            {"sequence": 0, "prompt_name": "p", "prompt": "think then answer"},
        ]

        result = asyncio.run(ffai.execute_graph(prompts))
        assert result.results["p"].status == "success"
        assert "<think" not in result.results["p"].response

    def test_execute_graph_empty_prompts(self):
        from src.FFAI import FFAI

        client = _MockAsyncClient()
        ffai = FFAI(client)

        result = asyncio.run(ffai.execute_graph([]))
        assert result.results == {}
        assert result.success_count == 0


class TestAsyncGraphExecutorConditionError:
    def test_condition_error_returns_failed(self):
        async def run_prompt(**kwargs):
            return ResponseResult(response="ok", model="test", status="success")

        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "init"},
            {
                "sequence": 1,
                "prompt_name": "b",
                "prompt": "check",
                "history": ["a"],
                "condition": '{{nonexistent.status}} == "success"',
            },
        ]

        executor = AsyncGraphExecutor(executor_fn=run_prompt)
        result = asyncio.run(executor.execute(prompts))
        assert result.results["b"].status == "failed"
        assert result.results["b"].condition_error is not None
        assert result.failed_count == 1


class TestAsyncGraphExecutorUnnamedNodes:
    def test_results_keyed_by_sequence_number(self):
        async def run_prompt(**kwargs):
            return ResponseResult(response="ok", model="test", status="success")

        prompts = [
            {"sequence": 0, "prompt": "first"},
            {"sequence": 1, "prompt": "second"},
        ]

        executor = AsyncGraphExecutor(executor_fn=run_prompt)
        result = asyncio.run(executor.execute(prompts))
        assert "0" in result.results
        assert "1" in result.results
        assert result.success_count == 2


class TestAsyncGraphExecutorEmptyPrompts:
    def test_empty_list_returns_empty_result(self):
        async def run_prompt(**kwargs):
            return ResponseResult(response="ok", model="test", status="success")

        executor = AsyncGraphExecutor(executor_fn=run_prompt)
        result = asyncio.run(executor.execute([]))
        assert result.results == {}
        assert result.success_count == 0
        assert result.failed_count == 0
        assert result.skipped_count == 0


class TestAsyncGraphExecutorAbort:
    def test_abort_condition_triggers_when_true(self):
        async def run_prompt(**kwargs):
            return ResponseResult(response="ok", model="test", status="success")

        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "init"},
            {
                "sequence": 1,
                "prompt_name": "b",
                "prompt": "check",
                "history": ["a"],
                "abort_condition": '{{a.status}} == "success"',
            },
        ]

        executor = AsyncGraphExecutor(executor_fn=run_prompt)
        result = asyncio.run(executor.execute(prompts))
        assert result.aborted is True
        assert result.success_count == 2

    def test_abort_condition_does_not_trigger_when_false(self):
        async def run_prompt(**kwargs):
            return ResponseResult(response="ok", model="test", status="success")

        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "init"},
            {
                "sequence": 1,
                "prompt_name": "b",
                "prompt": "check",
                "history": ["a"],
                "abort_condition": '{{a.status}} == "failed"',
            },
        ]

        executor = AsyncGraphExecutor(executor_fn=run_prompt)
        result = asyncio.run(executor.execute(prompts))
        assert result.aborted is False
        assert result.success_count == 2

    def test_abort_skips_remaining_levels(self):
        async def run_prompt(**kwargs):
            return ResponseResult(response="ok", model="test", status="success")

        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "init"},
            {
                "sequence": 1,
                "prompt_name": "b",
                "prompt": "check",
                "history": ["a"],
                "abort_condition": '{{a.status}} == "success"',
            },
            {
                "sequence": 2,
                "prompt_name": "c",
                "prompt": "should be skipped",
                "history": ["b"],
            },
        ]

        executor = AsyncGraphExecutor(executor_fn=run_prompt)
        result = asyncio.run(executor.execute(prompts))
        assert result.aborted is True
        assert result.success_count == 2
        assert result.results["c"].status == "skipped"
        assert "aborted" in (result.results["c"].condition_trace or "").lower()
        assert result.aborted_count == 1


class TestAsyncGraphExecutorFailurePropagation:
    def test_dependent_skipped_when_dep_fails(self):
        async def run_prompt(**kwargs):
            name = kwargs.get("prompt_name", "")
            if name == "bad":
                raise ValueError("API error")
            return ResponseResult(response="ok", model="test", status="success")

        prompts = [
            {"sequence": 0, "prompt_name": "bad", "prompt": "will fail"},
            {"sequence": 1, "prompt_name": "downstream", "prompt": "depends on bad", "history": ["bad"]},
        ]

        executor = AsyncGraphExecutor(executor_fn=run_prompt)
        result = asyncio.run(executor.execute(prompts))
        assert result.results["bad"].status == "failed"
        assert result.results["downstream"].status == "skipped"
        assert "bad" in (result.results["downstream"].condition_trace or "")

    def test_failure_propagation_in_diamond(self):
        async def run_prompt(**kwargs):
            name = kwargs.get("prompt_name", "")
            if name == "left":
                raise RuntimeError("left failed")
            return ResponseResult(response="ok", model="test", status="success")

        prompts = [
            {"sequence": 0, "prompt_name": "root", "prompt": "root"},
            {"sequence": 1, "prompt_name": "left", "prompt": "left", "history": ["root"]},
            {"sequence": 2, "prompt_name": "right", "prompt": "right", "history": ["root"]},
            {"sequence": 3, "prompt_name": "merge", "prompt": "merge", "history": ["left", "right"]},
        ]

        executor = AsyncGraphExecutor(executor_fn=run_prompt)
        result = asyncio.run(executor.execute(prompts))
        assert result.results["root"].status == "success"
        assert result.results["left"].status == "failed"
        assert result.results["right"].status == "success"
        assert result.results["merge"].status == "skipped"


class TestAsyncGraphExecutorPromptResolution:
    def test_prompt_resolver_called(self):
        from src.core.graph_execution_helpers import resolve_graph_prompt

        resolved_prompts: dict[str, str] = {}

        async def run_prompt(**kwargs):
            name = kwargs.get("prompt_name", "")
            prompt_text = kwargs.get("prompt", "")
            resolved_prompts[name] = prompt_text
            return ResponseResult(response=f"response for {name}", model="test", status="success")

        prompts = [
            {"sequence": 0, "prompt_name": "first", "prompt": "What is AI?"},
            {
                "sequence": 1,
                "prompt_name": "second",
                "prompt": "Summarize: {{first.response}}",
                "history": ["first"],
            },
        ]

        executor = AsyncGraphExecutor(
            executor_fn=run_prompt,
            prompt_resolver=resolve_graph_prompt,
        )
        result = asyncio.run(executor.execute(prompts))
        assert result.success_count == 2
        assert "response for first" in resolved_prompts["second"]

    def test_prompt_resolver_without_interpolation(self):
        async def run_prompt(**kwargs):
            return ResponseResult(response="ok", model="test", status="success")

        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "hello"},
        ]

        executor = AsyncGraphExecutor(
            executor_fn=run_prompt,
            prompt_resolver=lambda spec, results: (spec.get("prompt", ""), set()),
        )
        result = asyncio.run(executor.execute(prompts))
        assert result.success_count == 1

    def test_no_prompt_resolver_sends_raw(self):
        raw_prompts: list[str] = []

        async def run_prompt(**kwargs):
            raw_prompts.append(kwargs.get("prompt", ""))
            return ResponseResult(response="ok", model="test", status="success")

        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "What is {{ref.response}}?"},
        ]

        executor = AsyncGraphExecutor(executor_fn=run_prompt)
        result = asyncio.run(executor.execute(prompts))
        assert result.success_count == 1
        assert raw_prompts[0] == "What is {{ref.response}}?"


class TestFFAIExecuteGraphEnhanced:
    def test_execute_graph_resolves_interpolation(self):
        from src.FFAI import FFAI

        client = _MockAsyncClient(["AI stands for Artificial Intelligence", "Got it"])
        ffai = FFAI(client)

        prompts = [
            {"sequence": 0, "prompt_name": "define", "prompt": "Define AI"},
            {
                "sequence": 1,
                "prompt_name": "followup",
                "prompt": "Comment on: {{define.response}}",
                "history": ["define"],
            },
        ]

        result = asyncio.run(ffai.execute_graph(prompts))
        assert result.success_count == 2
        assert "Artificial Intelligence" in result.results["followup"].resolved_prompt

    def test_execute_graph_records_to_all_histories(self):
        from src.FFAI import FFAI

        client = _MockAsyncClient(["hello"])
        ffai = FFAI(client)

        prompts = [
            {"sequence": 0, "prompt_name": "greet", "prompt": "say hello"},
        ]

        result = asyncio.run(ffai.execute_graph(prompts))
        assert result.success_count == 1

        assert len(ffai.history) == 1
        assert ffai.history[0]["prompt_name"] == "greet"
        assert len(ffai.clean_history) == 1
        assert len(ffai.prompt_attr_history) >= 1
        assert len(ffai.ordered_history.get_all_interactions()) == 1
        assert len(ffai.permanent_history.get_turns_since(0)) >= 1

    def test_execute_graph_failure_propagation(self):
        from src.FFAI import FFAI

        class FailingClient(_MockAsyncClient):
            async def generate_response(self, prompt, **kwargs):
                if "fail" in prompt:
                    raise ValueError("API error")
                return "ok"

            async def clone(self):
                return FailingClient()

        client = FailingClient()
        ffai = FFAI(client)

        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "will fail"},
            {"sequence": 1, "prompt_name": "b", "prompt": "depends on a", "history": ["a"]},
        ]

        result = asyncio.run(ffai.execute_graph(prompts))
        assert result.results["a"].status == "failed"
        assert result.results["b"].status == "skipped"
        assert len(ffai.history) == 0

    def test_execute_graph_abort_cascades(self):
        from src.FFAI import FFAI

        client = _MockAsyncClient(["ok", "ok", "ok"])
        ffai = FFAI(client)

        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "init"},
            {
                "sequence": 1,
                "prompt_name": "b",
                "prompt": "check",
                "history": ["a"],
                "abort_condition": '{{a.status}} == "success"',
            },
            {
                "sequence": 2,
                "prompt_name": "c",
                "prompt": "should be aborted",
                "history": ["b"],
            },
        ]

        result = asyncio.run(ffai.execute_graph(prompts))
        assert result.aborted is True
        assert result.results["c"].status == "skipped"


class TestFFAIExecuteGraphAutoSequence:
    def test_auto_sequence_single_prompt(self):
        from src.FFAI import FFAI

        client = _MockAsyncClient(["hello"])
        ffai = FFAI(client)

        prompts = [
            {"prompt_name": "a", "prompt": "say hello"},
        ]

        result = asyncio.run(ffai.execute_graph(prompts))
        assert result.results["a"].status == "success"
        assert result.success_count == 1

    def test_auto_sequence_linear_chain(self):
        from src.FFAI import FFAI

        client = _MockAsyncClient(["A", "B", "C"])
        ffai = FFAI(client)

        prompts = [
            {"prompt_name": "topic", "prompt": "Suggest a topic"},
            {"prompt_name": "outline", "prompt": "Outline about {{topic.response}}", "history": ["topic"]},
            {"prompt_name": "article", "prompt": "Article based on {{outline.response}}", "history": ["outline"]},
        ]

        result = asyncio.run(ffai.execute_graph(prompts))
        assert result.success_count == 3
        assert result.results["topic"].status == "success"
        assert result.results["outline"].status == "success"
        assert result.results["article"].status == "success"

    def test_auto_sequence_diamond_dag(self):
        from src.FFAI import FFAI

        client = _MockAsyncClient(["root", "left", "right", "merged"])
        ffai = FFAI(client)

        prompts = [
            {"prompt_name": "root", "prompt": "start"},
            {"prompt_name": "left", "prompt": "left branch", "history": ["root"]},
            {"prompt_name": "right", "prompt": "right branch", "history": ["root"]},
            {"prompt_name": "merge", "prompt": "merge branches", "history": ["left", "right"]},
        ]

        result = asyncio.run(ffai.execute_graph(prompts))
        assert result.success_count == 4
        assert all(r.status == "success" for r in result.results.values())

    def test_auto_sequence_preserves_explicit_sequence(self):
        from src.FFAI import FFAI

        client = _MockAsyncClient(["X", "Y"])
        ffai = FFAI(client)

        prompts = [
            {"sequence": 5, "prompt_name": "a", "prompt": "first"},
            {"sequence": 10, "prompt_name": "b", "prompt": "second", "history": ["a"]},
        ]

        result = asyncio.run(ffai.execute_graph(prompts))
        assert result.success_count == 2
        assert result.results["b"].status == "success"

    def test_auto_sequence_mixed_with_and_without(self):
        from src.FFAI import FFAI

        client = _MockAsyncClient(["A", "B", "C"])
        ffai = FFAI(client)

        prompts = [
            {"sequence": 0, "prompt_name": "a", "prompt": "explicit seq"},
            {"prompt_name": "b", "prompt": "auto seq", "history": ["a"]},
            {"prompt_name": "c", "prompt": "also auto", "history": ["b"]},
        ]

        result = asyncio.run(ffai.execute_graph(prompts))
        assert result.success_count == 3
        assert result.results["c"].status == "success"

    def test_auto_sequence_empty_list(self):
        from src.FFAI import FFAI

        client = _MockAsyncClient()
        ffai = FFAI(client)

        result = asyncio.run(ffai.execute_graph([]))
        assert result.results == {}
        assert result.success_count == 0

    def test_auto_sequence_resolves_interpolation(self):
        from src.FFAI import FFAI

        client = _MockAsyncClient(["Paris", "Paris is beautiful"])
        ffai = FFAI(client)

        prompts = [
            {"prompt_name": "capital", "prompt": "Capital of France?"},
            {"prompt_name": "describe", "prompt": "Describe {{capital.response}}", "history": ["capital"]},
        ]

        result = asyncio.run(ffai.execute_graph(prompts))
        assert result.success_count == 2
        assert "Paris" in result.results["describe"].resolved_prompt


class TestValidateGraphAutoSequence:
    def test_validate_graph_without_sequence_numbers(self):
        from src.FFAI import FFAI

        client = _MockAsyncClient()
        ffai = FFAI(client)

        prompts = [
            {"prompt_name": "topic", "prompt": "Suggest a topic"},
            {"prompt_name": "outline", "prompt": "Outline about {{topic.response}}", "history": ["topic"]},
            {"prompt_name": "article", "prompt": "Article based on {{outline.response}}", "history": ["outline"]},
        ]

        graph, warnings = ffai.validate_graph(prompts)
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2

    def test_validate_graph_diamond_without_sequence(self):
        from src.FFAI import FFAI

        client = _MockAsyncClient()
        ffai = FFAI(client)

        prompts = [
            {"prompt_name": "root", "prompt": "start"},
            {"prompt_name": "left", "prompt": "left branch", "history": ["root"]},
            {"prompt_name": "right", "prompt": "right branch", "history": ["root"]},
            {"prompt_name": "merge", "prompt": "merge", "history": ["left", "right"]},
        ]

        graph, warnings = ffai.validate_graph(prompts)
        assert len(graph.nodes) == 4
        assert len(graph.edges) == 4


class TestAsyncClientBasePassthrough:
    def test_super_generate_response_returns_none(self):
        class DelegatingAsyncClient(AsyncFFAIClientBase):
            model = "test"
            system_instructions = ""

            async def generate_response(self, prompt: str, **kwargs):
                return await super().generate_response(prompt, **kwargs)

            async def clone(self):
                return await super().clone()

            def clear_conversation(self):
                pass

            def get_conversation_history(self):
                return []

            def set_conversation_history(self, history):
                pass

        client = DelegatingAsyncClient()
        assert asyncio.run(client.generate_response("test")) is None
        assert asyncio.run(client.clone()) is None
