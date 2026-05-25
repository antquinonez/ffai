# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT

import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel

from src.core.execution_result import ExecutionResult
from src.core.response_executor import ResponseExecutor
from src.core.response_options import ResponseOptions


def _make_executor(
    prompt_responses: dict[str, str] | None = None,
    condition_result: tuple[bool, str | None, str | None] = (True, None, None),
    results_by_name: dict[str, dict[str, Any]] | None = None,
):
    prompt_responses = prompt_responses or {}

    def mock_prompt_builder(prompt, history=None, dependencies=None, strict=False):
        result = prompt_responses[prompt] if prompt in prompt_responses else prompt
        return result, set()  # type: ignore[return-value]

    def mock_evaluate_condition(condition, results):
        return condition_result

    def mock_build_results():
        return results_by_name or {}

    return ResponseExecutor(
        prompt_builder=mock_prompt_builder,
        condition_evaluator_fn=mock_evaluate_condition,
        results_by_name_fn=mock_build_results,
    )


class TestExecuteBasic:
    def test_simple_call_returns_success(self):
        executor = _make_executor()
        generate_fn = MagicMock(return_value="Hello world")

        result = executor.execute(
            generate_fn=generate_fn,
            prompt="Say hello",
            prompt_name="greeting",
            options=ResponseOptions(),
            default_model="test-model",
        )

        assert result.status == "success"
        assert result.response == "Hello world"
        assert result.model == "test-model"
        assert result.duration_ms > 0
        assert result.parsed is None
        assert result.parsing_errors is None

    def test_model_override(self):
        executor = _make_executor()
        generate_fn = MagicMock(return_value="ok")

        result = executor.execute(
            generate_fn=generate_fn,
            prompt="test",
            prompt_name="t",
            options=ResponseOptions(model="gpt-4"),
            default_model="test-model",
        )

        assert result.model == "gpt-4"
        call_kwargs = generate_fn.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4"

    def test_system_instructions_forwarded(self):
        executor = _make_executor()
        generate_fn = MagicMock(return_value="ok")

        executor.execute(
            generate_fn=generate_fn,
            prompt="test",
            prompt_name="t",
            options=ResponseOptions(system_instructions="Be concise"),
            default_model="test-model",
        )

        call_kwargs = generate_fn.call_args.kwargs
        assert "system_instructions" in call_kwargs
        assert "Be concise" in call_kwargs["system_instructions"]

    def test_response_format_forwarded(self):
        executor = _make_executor()
        generate_fn = MagicMock(return_value="ok")

        executor.execute(
            generate_fn=generate_fn,
            prompt="test",
            prompt_name="t",
            options=ResponseOptions(response_format={"type": "json_object"}),
            default_model="test-model",
        )

        call_kwargs = generate_fn.call_args.kwargs
        assert call_kwargs["response_format"] == {"type": "json_object"}

    def test_dependencies_deduplicated(self):
        deps_seen = []

        def tracking_builder(prompt, history=None, dependencies=None, strict=False):
            deps_seen.extend(dependencies or [])
            return prompt, set()

        executor = ResponseExecutor(
            prompt_builder=tracking_builder,
            condition_evaluator_fn=lambda c, r: (True, None, None),
            results_by_name_fn=lambda: {},
        )
        generate_fn = MagicMock(return_value="ok")

        executor.execute(
            generate_fn=generate_fn,
            prompt="test",
            prompt_name="t",
            options=ResponseOptions(dependencies=["a", "a", "b", "b"]),
            default_model="test-model",
        )

        assert sorted(deps_seen) == ["a", "b"]


class TestExecuteCondition:
    def test_condition_true_executes(self):
        executor = _make_executor(condition_result=(True, None, None))
        generate_fn = MagicMock(return_value="executed")

        result = executor.execute(
            generate_fn=generate_fn,
            prompt="test",
            prompt_name="t",
            options=ResponseOptions(condition='{{x.status}} == "success"'),
            default_model="test-model",
        )

        assert result.status == "success"
        assert result.response == "executed"
        generate_fn.assert_called_once()

    def test_condition_false_skips(self):
        executor = _make_executor(condition_result=(False, None, "trace: False"))
        generate_fn = MagicMock(return_value="should not run")

        result = executor.execute(
            generate_fn=generate_fn,
            prompt="test",
            prompt_name="t",
            options=ResponseOptions(condition="False"),
            default_model="test-model",
        )

        assert result.status == "skipped"
        assert result.response is None
        assert result.condition_trace == "trace: False"
        generate_fn.assert_not_called()

    def test_condition_error_returns_failed(self):
        executor = _make_executor(condition_result=(False, "Unknown reference", None))
        generate_fn = MagicMock(return_value="should not run")

        result = executor.execute(
            generate_fn=generate_fn,
            prompt="test",
            prompt_name="t",
            options=ResponseOptions(condition='{{bad.status}} == "ok"'),
            default_model="test-model",
        )

        assert result.status == "failed"
        assert result.response is None
        assert result.condition_error == "Unknown reference"
        generate_fn.assert_not_called()

    def test_skip_condition_true_bypasses_evaluation(self):
        call_count = 0

        def counting_evaluator(condition, results):
            nonlocal call_count
            call_count += 1
            return (True, None, None)

        executor = ResponseExecutor(
            prompt_builder=lambda p, h=None, d=None, strict=False: (p, set()),
            condition_evaluator_fn=counting_evaluator,
            results_by_name_fn=lambda: {},
        )
        generate_fn = MagicMock(return_value="ok")

        result = executor.execute(
            generate_fn=generate_fn,
            prompt="test",
            prompt_name="t",
            options=ResponseOptions(condition="True"),
            default_model="test-model",
            skip_condition=True,
        )

        assert result.status == "success"
        assert call_count == 0


class TestExecuteStructuredOutput:
    def test_valid_json_parses_on_first_attempt(self):
        executor = _make_executor()

        class Score(BaseModel):
            value: int

        generate_fn = MagicMock(return_value=json.dumps({"value": 42}))

        result = executor.execute(
            generate_fn=generate_fn,
            prompt="score",
            prompt_name="score",
            options=ResponseOptions(response_model=Score),
            default_model="test-model",
        )

        assert result.status == "success"
        assert result.parsed is not None
        assert result.parsed.value == 42
        assert result.parsing_errors is None
        assert generate_fn.call_count == 1

    def test_invalid_then_valid_retries(self):
        executor = _make_executor()

        class Score(BaseModel):
            value: int

        generate_fn = MagicMock(
            side_effect=["not json", json.dumps({"value": 7})]
        )

        result = executor.execute(
            generate_fn=generate_fn,
            prompt="score",
            prompt_name="score",
            options=ResponseOptions(response_model=Score),
            default_model="test-model",
        )

        assert result.parsed is not None
        assert result.parsed.value == 7
        assert generate_fn.call_count == 2

    def test_all_retries_fail(self):
        executor = _make_executor()

        class Strict(BaseModel):
            score: int

        generate_fn = MagicMock(return_value="not json at all")

        result = executor.execute(
            generate_fn=generate_fn,
            prompt="score",
            prompt_name="score",
            options=ResponseOptions(response_model=Strict),
            default_model="test-model",
        )

        assert result.parsed is None
        assert result.parsing_errors is not None
        assert len(result.parsing_errors) > 0
        assert generate_fn.call_count == 3

    def test_invalid_response_model_type_raises(self):
        executor = _make_executor()
        generate_fn = MagicMock(return_value="ok")

        with pytest.raises(TypeError, match="Pydantic BaseModel subclass"):
            executor.execute(
                generate_fn=generate_fn,
                prompt="test",
                prompt_name="t",
                options=ResponseOptions(response_model=str),
                default_model="test-model",
            )

    def test_custom_response_format_preserved(self):
        executor = _make_executor()

        class Item(BaseModel):
            name: str

        custom_fmt = {"type": "json_object", "custom": True}
        generate_fn = MagicMock(return_value=json.dumps({"name": "widget"}))

        executor.execute(
            generate_fn=generate_fn,
            prompt="test",
            prompt_name="t",
            options=ResponseOptions(response_model=Item, response_format=custom_fmt),
            default_model="test-model",
        )

        call_kwargs = generate_fn.call_args.kwargs
        assert call_kwargs["response_format"] == custom_fmt

    def test_schema_appended_to_system_instructions(self):
        executor = _make_executor()

        class Item(BaseModel):
            name: str

        generate_fn = MagicMock(return_value=json.dumps({"name": "widget"}))

        executor.execute(
            generate_fn=generate_fn,
            prompt="test",
            prompt_name="t",
            options=ResponseOptions(
                response_model=Item,
                system_instructions="You are a cataloger.",
            ),
            default_model="test-model",
        )

        call_kwargs = generate_fn.call_args.kwargs
        sys_instr = call_kwargs["system_instructions"]
        assert "You are a cataloger." in sys_instr
        assert "json" in sys_instr.lower()


class TestExecuteAsync:
    def test_async_simple_call(self):
        executor = _make_executor()

        async def mock_generate(**kwargs):
            return "async response"

        result = asyncio.run(executor.execute_async(
            generate_fn=mock_generate,
            prompt="test",
            prompt_name="t",
            options=ResponseOptions(),
            default_model="test-model",
        ))

        assert result.status == "success"
        assert result.response == "async response"

    def test_async_condition_skip(self):
        executor = _make_executor(condition_result=(False, None, "trace"))

        async def mock_generate(**kwargs):
            return "should not run"

        result = asyncio.run(executor.execute_async(
            generate_fn=mock_generate,
            prompt="test",
            prompt_name="t",
            options=ResponseOptions(condition="False"),
            default_model="test-model",
        ))

        assert result.status == "skipped"

    def test_async_structured_output(self):
        executor = _make_executor()

        class Score(BaseModel):
            value: int

        async def mock_generate(**kwargs):
            return json.dumps({"value": 99})

        result = asyncio.run(executor.execute_async(
            generate_fn=mock_generate,
            prompt="score",
            prompt_name="score",
            options=ResponseOptions(response_model=Score),
            default_model="test-model",
        ))

        assert result.parsed is not None
        assert result.parsed.value == 99

    def test_async_structured_retry(self):
        executor = _make_executor()

        class Score(BaseModel):
            value: int

        call_count = 0

        async def mock_generate(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "bad"
            return json.dumps({"value": 5})

        result = asyncio.run(executor.execute_async(
            generate_fn=mock_generate,
            prompt="score",
            prompt_name="score",
            options=ResponseOptions(response_model=Score),
            default_model="test-model",
        ))

        assert result.parsed.value == 5
        assert call_count == 2

    def test_async_skip_condition_flag(self):
        eval_count = 0

        def counting_evaluator(condition, results):
            nonlocal eval_count
            eval_count += 1
            return (True, None, None)

        executor = ResponseExecutor(
            prompt_builder=lambda p, h=None, d=None, strict=False: (p, set()),
            condition_evaluator_fn=counting_evaluator,
            results_by_name_fn=lambda: {},
        )

        async def mock_generate(**kwargs):
            return "ok"

        asyncio.run(executor.execute_async(
            generate_fn=mock_generate,
            prompt="test",
            prompt_name="t",
            options=ResponseOptions(condition="True"),
            default_model="test-model",
            skip_condition=True,
        ))

        assert eval_count == 0


class TestExecutionResult:
    def test_defaults(self):
        result = ExecutionResult()
        assert result.response is None
        assert result.resolved_prompt == ""
        assert result.model == ""
        assert result.usage is None
        assert result.cost_usd == 0.0
        assert result.duration_ms == 0.0
        assert result.status == "success"
        assert result.condition_trace is None
        assert result.condition_error is None
        assert result.parsed is None
        assert result.parsing_errors is None
