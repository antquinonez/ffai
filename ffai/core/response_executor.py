# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Orchestration helper for prompt execution with conditions and structured output."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

from .execution_result import ExecutionResult
from .response_options import ResponseOptions
from .structured_output import StructuredOutputHandler

logger = logging.getLogger(__name__)

_CallableT = Callable[..., Any]


class ResponseExecutor:
    """Orchestrates prompt resolution, condition evaluation, and LLM dispatch.

    Constructed once in ``FFAI.__init__`` with callables bound to the FFAI
    instance.  The ``execute`` method handles the synchronous path used by
    ``generate_response``; ``execute_async`` handles the async path used by
    ``execute_graph``.

    Args:
        prompt_builder: Bound ``PromptBuilder.build_prompt`` method.
        condition_evaluator_fn: Callable taking ``(condition_str, results_by_name)``
            and returning ``(should_execute, error, trace)``.
        results_by_name_fn: Callable returning ``dict[str, dict[str, Any]]``
            mapping prompt names to their results.

    """

    def __init__(
        self,
        prompt_builder: Callable[..., tuple[str, set[str]]],
        condition_evaluator_fn: Callable[
            [str, dict[str, dict[str, Any]]], tuple[bool, str | None, str | None]
        ],
        results_by_name_fn: Callable[[], dict[str, dict[str, Any]]],
    ) -> None:
        self._prompt_builder = prompt_builder
        self._evaluate_condition = condition_evaluator_fn
        self._build_results_by_name = results_by_name_fn

    def execute(
        self,
        generate_fn: _CallableT,
        prompt: str,
        prompt_name: str | None,
        options: ResponseOptions,
        default_model: str,
        skip_condition: bool = False,
    ) -> ExecutionResult:
        """Synchronous execution path (used by ``generate_response``)."""
        used_model = options.model if options.model else default_model
        final_prompt, kwargs, condition_result = self._prepare(
            prompt, prompt_name, options, used_model, skip_condition,
        )

        if condition_result is not None:
            return condition_result

        if options.response_model is not None:
            return self._execute_structured_sync(
                generate_fn, final_prompt, used_model, kwargs, options.response_model,
            )

        call_kwargs: dict[str, Any] = {"prompt": final_prompt, "model": used_model, **kwargs}
        call_start = time.monotonic()
        response = generate_fn(**call_kwargs)
        call_duration_ms = (time.monotonic() - call_start) * 1000

        return ExecutionResult(
            response=response,
            resolved_prompt=final_prompt,
            model=used_model,
            duration_ms=round(call_duration_ms, 1),
            status="success",
        )

    async def execute_async(
        self,
        generate_fn: _CallableT,
        prompt: str,
        prompt_name: str | None,
        options: ResponseOptions,
        default_model: str,
        skip_condition: bool = False,
    ) -> ExecutionResult:
        """Async execution path (used by ``execute_graph``)."""
        used_model = options.model if options.model else default_model
        final_prompt, kwargs, condition_result = self._prepare(
            prompt, prompt_name, options, used_model, skip_condition,
        )

        if condition_result is not None:
            return condition_result

        if options.response_model is not None:
            return await self._execute_structured_async(
                generate_fn, final_prompt, used_model, kwargs, options.response_model,
            )

        call_kwargs: dict[str, Any] = {"prompt": final_prompt, "model": used_model, **kwargs}
        call_start = time.monotonic()
        response = await generate_fn(**call_kwargs)
        call_duration_ms = (time.monotonic() - call_start) * 1000

        return ExecutionResult(
            response=response,
            resolved_prompt=final_prompt,
            model=used_model,
            duration_ms=round(call_duration_ms, 1),
            status="success",
        )

    def _prepare(
        self,
        prompt: str,
        prompt_name: str | None,
        options: ResponseOptions,
        used_model: str,
        skip_condition: bool,
    ) -> tuple[str, dict[str, Any], ExecutionResult | None]:
        """Build prompt, evaluate condition, and prepare kwargs.

        Returns:
            Tuple of (final_prompt, kwargs, condition_result).
            If condition_result is not None, the caller should return it
            immediately (skip or failure).

        """
        dependencies = options.dependencies
        if dependencies:
            dependencies = list(set(dependencies))

        final_prompt, _interpolated_names = self._prompt_builder(
            prompt, options.history, dependencies, strict=options.strict,
        )

        if not skip_condition and options.condition is not None:
            results_by_name = self._build_results_by_name()
            should_execute, error, trace = self._evaluate_condition(
                options.condition, results_by_name
            )
            logger.info(
                f"Condition evaluation for '{prompt_name}': "
                f"should_execute={should_execute}"
            )
            if error is not None:
                logger.warning(f"Condition evaluation error for '{prompt_name}': {error}")
                return final_prompt, {}, ExecutionResult(
                    response=None,
                    resolved_prompt=final_prompt,
                    model=used_model,
                    status="failed",
                    condition_error=error,
                )
            if not should_execute:
                return final_prompt, {}, ExecutionResult(
                    response=None,
                    resolved_prompt=final_prompt,
                    model=used_model,
                    status="skipped",
                    condition_trace=trace,
                )

        kwargs: dict[str, Any] = {}

        if options.system_instructions is not None:
            kwargs["system_instructions"] = options.system_instructions

        if options.response_model is not None:
            from pydantic import BaseModel as _BaseModel

            if not (
                isinstance(options.response_model, type)
                and issubclass(options.response_model, _BaseModel)
            ):
                raise TypeError(
                    f"response_model must be a Pydantic BaseModel subclass, "
                    f"got {type(options.response_model)}"
                )

            so_handler = StructuredOutputHandler(max_retries=2)
            if options.response_format is None:
                kwargs["response_format"] = so_handler.build_response_format(options.response_model)
            else:
                kwargs["response_format"] = options.response_format
            schema_suffix = so_handler.build_system_suffix(options.response_model)
            current_system = kwargs.get("system_instructions") or options.system_instructions or ""
            kwargs["system_instructions"] = current_system + schema_suffix
            return final_prompt, kwargs, None

        if options.response_format is not None:
            kwargs["response_format"] = options.response_format

        return final_prompt, kwargs, None

    @staticmethod
    def _execute_structured_sync(
        generate_fn: _CallableT,
        final_prompt: str,
        used_model: str,
        kwargs: dict[str, Any],
        response_model: type,
    ) -> ExecutionResult:
        so_handler = StructuredOutputHandler(max_retries=2)
        max_attempts = so_handler.max_retries + 1
        all_errors: list[str] = []
        current_prompt = final_prompt
        best_result = None

        call_start = time.monotonic()
        for attempt in range(max_attempts):
            call_kwargs: dict[str, Any] = {"prompt": current_prompt, "model": used_model, **kwargs}
            response = generate_fn(**call_kwargs)
            best_result, current_prompt, all_errors, should_stop = (
                so_handler.process_attempt(
                    response, response_model, final_prompt,
                    attempt, all_errors, best_result,
                )
            )
            if should_stop:
                call_duration_ms = (time.monotonic() - call_start) * 1000
                return ExecutionResult(
                    response=response,
                    resolved_prompt=final_prompt,
                    model=used_model,
                    duration_ms=round(call_duration_ms, 1),
                    status="success",
                    parsed=best_result.parsed if best_result else None,
                    parsing_errors=(
                        best_result.parsing_errors
                        if best_result and best_result.parsing_errors
                        else None
                    ),
                )

        final_so = so_handler.finalize_retry(best_result, all_errors, max_attempts)
        call_duration_ms = (time.monotonic() - call_start) * 1000
        return ExecutionResult(
            response=final_so.raw_response,
            resolved_prompt=final_prompt,
            model=used_model,
            duration_ms=round(call_duration_ms, 1),
            status="success",
            parsed=final_so.parsed,
            parsing_errors=final_so.parsing_errors if final_so.parsing_errors else None,
        )

    @staticmethod
    async def _execute_structured_async(
        generate_fn: _CallableT,
        final_prompt: str,
        used_model: str,
        kwargs: dict[str, Any],
        response_model: type,
    ) -> ExecutionResult:
        so_handler = StructuredOutputHandler(max_retries=2)
        max_attempts = so_handler.max_retries + 1
        all_errors: list[str] = []
        current_prompt = final_prompt
        best_result = None

        call_start = time.monotonic()
        for attempt in range(max_attempts):
            call_kwargs: dict[str, Any] = {"prompt": current_prompt, "model": used_model, **kwargs}
            response = await generate_fn(**call_kwargs)
            best_result, current_prompt, all_errors, should_stop = (
                so_handler.process_attempt(
                    response, response_model, final_prompt,
                    attempt, all_errors, best_result,
                )
            )
            if should_stop:
                call_duration_ms = (time.monotonic() - call_start) * 1000
                return ExecutionResult(
                    response=response,
                    resolved_prompt=final_prompt,
                    model=used_model,
                    duration_ms=round(call_duration_ms, 1),
                    status="success",
                    parsed=best_result.parsed if best_result else None,
                    parsing_errors=(
                        best_result.parsing_errors
                        if best_result and best_result.parsing_errors
                        else None
                    ),
                )

        final_so = so_handler.finalize_retry(best_result, all_errors, max_attempts)
        call_duration_ms = (time.monotonic() - call_start) * 1000
        return ExecutionResult(
            response=final_so.raw_response,
            resolved_prompt=final_prompt,
            model=used_model,
            duration_ms=round(call_duration_ms, 1),
            status="success",
            parsed=final_so.parsed,
            parsing_errors=final_so.parsing_errors if final_so.parsing_errors else None,
        )
