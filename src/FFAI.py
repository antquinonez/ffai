# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Declarative context handling API wrapper for AI clients.

This module provides the FFAI class which wraps AI client implementations
and adds declarative context management, history tracking, and DataFrame
export capabilities.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import polars as pl

from .config import get_config
from .core.client_base import FFAIClientBase
from .core.condition_evaluator import ConditionEvaluator
from .core.graph import ExecutionGraph, build_execution_graph_with_edges
from .core.graph_execution_helpers import resolve_graph_prompt
from .core.history.ordered import OrderedPromptHistory
from .core.history.permanent import PermanentHistory
from .core.history_exporter import HistoryExporter
from .core.prompt_builder import PromptBuilder
from .core.prompt_utils import extract_json_field, interpolate_prompt
from .core.response_context import ResponseContext
from .core.response_result import ResponseResult
from .core.response_utils import clean_response as _clean_response_impl
from .core.response_utils import extract_json
from .core.usage import TokenUsage

__all__ = ["FFAI", "extract_json_field", "interpolate_prompt"]

logger = logging.getLogger(__name__)


class FFAI:
    """Declarative context handling wrapper for AI clients.

    This class wraps an AI client implementation and adds:
    - Named prompt management for declarative context assembly
    - Multiple history tracking mechanisms via ``ResponseContext``
    - DataFrame export capabilities
    - Automatic persistence of history data

    ``generate_response()`` returns a ``ResponseResult`` with the response
    text, resolved prompt, token usage, and cost estimate.

    Attributes:
        client: The underlying AI client instance.
        history: Raw interaction history.
        clean_history: Cleaned interaction history.
        prompt_attr_history: History indexed by prompt attributes (via ``ResponseContext``).
        ordered_history: Ordered prompt-response history.
        permanent_history: Chronological turn history.

    """

    def __init__(
        self,
        client: FFAIClientBase,
        persist_dir: str | None = None,
        persist_name: str | None = None,
        auto_persist: bool = False,
        shared_prompt_attr_history: list[dict[str, Any]] | None = None,
        history_lock: threading.Lock | None = None,
    ) -> None:
        """Initialize the FFAI wrapper.

        Args:
            client: AI client instance to wrap.
            persist_dir: Directory for persistence files. Uses config default if None.
            persist_name: Base name for persisted files.
            auto_persist: Whether to automatically persist DataFrames.
            shared_prompt_attr_history: Optional shared history list for parallel execution.
            history_lock: Optional lock for thread-safe history access.

        """
        logger.info("Initializing FFAI wrapper")

        config = get_config()
        self.persist_dir = persist_dir if persist_dir is not None else config.paths.ffai_data
        self.persist_name = persist_name
        self.auto_persist = auto_persist
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = client
        self._client_history_lock = threading.Lock()

        self.history: list[dict[str, Any]] = []
        self.clean_history: list[dict[str, Any]] = []

        self._context = ResponseContext(
            shared_prompt_attr_history=shared_prompt_attr_history,
            history_lock=history_lock,
        )

        self.permanent_history = PermanentHistory()
        self.ordered_history = OrderedPromptHistory()

        self._prompt_builder = PromptBuilder(self._context.prompt_attr_history)
        self._exporter = HistoryExporter(
            history=self.history,
            clean_history=self.clean_history,
            prompt_attr_history=self._context.prompt_attr_history,
            ordered_history=self.ordered_history,
            persist_dir=self.persist_dir,
            persist_name=self.persist_name,
            auto_persist=self.auto_persist,
        )

    @property
    def prompt_attr_history(self) -> list[dict[str, Any]]:
        """Shared prompt attribute history (delegates to ResponseContext)."""
        return self._context.prompt_attr_history

    @prompt_attr_history.setter
    def prompt_attr_history(self, value: list[dict[str, Any]]) -> None:
        """Allow external reassignment for backward compatibility."""
        self._context.prompt_attr_history = value

    @property
    def _history_lock(self) -> threading.Lock | None:
        """Backward-compatible access to the history lock."""
        return self._context.history_lock

    def set_client(self, client: FFAIClientBase) -> None:
        """Switch to a different AI client."""
        logger.info(f"Switching client to {client.__class__.__name__}")
        self.client = client

    def _extract_json(self, text: str) -> Any | None:
        """Extract JSON from text, handling markdown code blocks and JSON within first 20 chars."""
        return extract_json(text)

    def get_system_instructions(self) -> str | None:
        """Get system instructions from the client."""
        if hasattr(self.client, "system_instructions"):
            return self.client.system_instructions
        return None

    def _clean_response(self, response: Any) -> Any:
        """Process and validate the evaluation response."""
        return _clean_response_impl(response)

    def _build_prompt(
        self,
        prompt: str,
        history: list[str] | None = None,
        dependencies: Any | None = None,
        strict: bool = False,
    ) -> tuple[str, set[str]]:
        """Build the final prompt with history and variable interpolation."""
        result, interpolated = self._prompt_builder.build_prompt(
            prompt, history, dependencies, strict=strict
        )
        return result, interpolated

    def build_prompt(
        self,
        prompt: str,
        history: list[str] | None = None,
        dependencies: Any | None = None,
        strict: bool = False,
    ) -> tuple[str, set[str]]:
        """Public API for prompt building (delegates to PromptBuilder).

        Use this instead of the private ``_build_prompt()`` method.
        """
        return self._build_prompt(prompt, history, dependencies, strict=strict)

    def generate_response(
        self,
        prompt: str,
        model: str | None = None,
        prompt_name: str | None = None,
        history: list[str] | None = None,
        dependencies: list[str] | None = None,
        system_instructions: str | None = None,
        response_format: str | dict | None = None,
        response_model: type | None = None,
        condition: str | None = None,
        abort_condition: str | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> ResponseResult:
        """Generate response using the configured AI client.

        Args:
            prompt: The user prompt (may contain ``{{name.response}}`` patterns).
            model: Override model for this call.
            prompt_name: Logical name for the prompt (used for interpolation).
            history: List of prompt names to include in conversation context.
            dependencies: Prompt names this call depends on.
            system_instructions: Override system instructions for this call.
            response_format: Response format hint (e.g. ``{"type": "json_object"}``).
                Passed through to the underlying provider API.
            response_model: Pydantic BaseModel subclass for structured output.
                When set, the response is validated against the model schema
                with up to 3 total attempts on validation failure.
            condition: Expression evaluated before execution. If false, prompt
                is skipped with ``status="skipped"``.
            abort_condition: Stored for future DAG executor use.
            strict: If True, raise ValueError on unknown ``{{name.response}}``
                references instead of silently replacing with empty string.
            **kwargs: Additional provider-specific parameters (temperature,
                max_tokens, tools, tool_choice, etc.).

        Returns:
            ``ResponseResult`` with response, resolved prompt, usage, and cost.

        """
        logger.debug(
            "\n==================================================================================="
        )
        prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
        logger.info(
            f"Generating response for '{prompt_name}' ({len(prompt)} chars): '{prompt_preview}'"
        )
        logger.debug(f"Full prompt: '{prompt}'")
        logger.debug(f"Prompt_name: '{prompt_name}'")
        logger.debug(
            f"System instructions: '{system_instructions}'"
        ) if system_instructions else logger.debug("No system instructions provided")
        logger.debug(f"History: {history}") if history else logger.debug("No history provided")
        logger.debug(f"Dependencies: {dependencies}") if dependencies else logger.debug(
            "No dependencies provided"
        )

        used_model = model if model else self.client.model
        logger.debug(f"Using model: {used_model}")

        cleaned_response = None
        usage: TokenUsage | None = None
        cost_usd: float = 0.0
        resolved_prompt: str = prompt

        try:
            if dependencies:
                dependencies_set = set(dependencies)
                dependencies = list(dependencies_set)

            final_prompt, interpolated_names = self._build_prompt(prompt, history, dependencies, strict=strict)
            resolved_prompt = final_prompt
            logger.debug(f"final_prompt built: {final_prompt}")
            logger.debug(f"interpolated_names: {interpolated_names}")

            if condition is not None:
                results_by_name = self._build_results_by_name()
                should_execute, error, trace = self._evaluate_condition(
                    condition, results_by_name
                )
                logger.info(
                    f"Condition evaluation for '{prompt_name}': "
                    f"should_execute={should_execute}"
                )
                if error is not None:
                    logger.warning(f"Condition evaluation error for '{prompt_name}': {error}")
                    return ResponseResult(
                        response=None,
                        resolved_prompt=resolved_prompt,
                        model=used_model,
                        status="failed",
                        condition_error=error,
                    )
                if not should_execute:
                    interaction = {
                        "prompt": prompt,
                        "response": None,
                        "prompt_name": prompt_name,
                        "timestamp": time.time(),
                        "model": used_model,
                        "history": history,
                        "status": "skipped",
                        "condition_trace": trace,
                    }
                    self.history.append(interaction)
                    return ResponseResult(
                        response=None,
                        resolved_prompt=resolved_prompt,
                        model=used_model,
                        status="skipped",
                        condition_trace=trace,
                    )

            if system_instructions is not None:
                kwargs["system_instructions"] = system_instructions

            if response_model is not None:
                from pydantic import BaseModel as _BaseModel

                if not (isinstance(response_model, type) and issubclass(response_model, _BaseModel)):
                    raise TypeError(
                        f"response_model must be a Pydantic BaseModel subclass, got {type(response_model)}"
                    )

                from .core.structured_output import StructuredOutputHandler

                so_handler = StructuredOutputHandler(max_retries=2)
                if response_format is None:
                    response_format = so_handler.build_response_format(response_model)
                schema_suffix = so_handler.build_system_suffix(response_model)
                current_system = kwargs.get("system_instructions") or system_instructions or ""
                kwargs["system_instructions"] = current_system + schema_suffix

            if response_format is not None:
                kwargs["response_format"] = response_format

            saved_client_history = None
            new_client_messages: list[dict[str, Any]] = []
            should_suspend_client_history = history is not None or bool(interpolated_names)

            if should_suspend_client_history:
                with self._client_history_lock:
                    saved_client_history = self.client.get_conversation_history().copy()
                    self.client.set_conversation_history([])
                    reason = "history injection" if history else f"interpolation of {interpolated_names}"
                    logger.debug(
                        f"Suspended client conversation history: {reason}"
                    )

            so_result = None
            call_start = time.monotonic()
            try:
                if response_model is not None:
                    response, so_result = self._execute_structured(
                        so_handler, response_model, final_prompt, used_model, kwargs,
                        should_suspend_client_history, saved_client_history,
                    )
                else:
                    response = self.client.generate_response(
                        prompt=final_prompt, model=used_model, **kwargs
                    )
            finally:
                if should_suspend_client_history and saved_client_history is not None and so_result is None:
                    with self._client_history_lock:
                        new_client_messages = self.client.get_conversation_history()
                        combined = list(saved_client_history) + list(new_client_messages)
                        self.client.set_conversation_history(combined)
                        logger.debug(
                            f"Restored client conversation history (+{len(new_client_messages)} new messages)"
                        )
            call_duration_ms = (time.monotonic() - call_start) * 1000
            usage = getattr(self.client, "last_usage", None)
            cost_usd = getattr(self.client, "last_cost_usd", 0.0)

            logger.debug(f"Generated response: {response}")

            cleaned_response = self._clean_response(response)
            logger.debug(f"cleaned_response: {cleaned_response}")

            self.permanent_history.add_turn_user(prompt)
            self.permanent_history.add_turn_assistant(cleaned_response)

            interaction = {
                "prompt": prompt,
                "response": cleaned_response,
                "prompt_name": prompt_name,
                "timestamp": time.time(),
                "model": used_model,
                "history": history,
                "status": "success",
            }

            self.history.append(interaction)
            self.clean_history.append(interaction)

            self._context.record(prompt, cleaned_response, used_model, prompt_name, history)

            self.ordered_history.add_interaction(
                model=used_model,
                prompt=prompt,
                response=cleaned_response,
                prompt_name=prompt_name,
                history=history,
            )

            return ResponseResult(
                response=cleaned_response,
                resolved_prompt=resolved_prompt,
                usage=usage,
                cost_usd=cost_usd,
                model=used_model,
                duration_ms=round(call_duration_ms, 1),
                parsed=so_result.parsed if so_result else None,
                parsing_errors=so_result.parsing_errors if so_result and so_result.parsing_errors else None,
            )

        except Exception as e:
            logger.error(f"Problem with response generation: {e!s}")
            logger.error(f"Prompt: {prompt}")
            logger.error(f"Model: {used_model}")
            logger.error(f"Prompt name: {prompt_name}")
            logger.error(
                f"Response: {cleaned_response if cleaned_response is not None else 'No response generated'}"
            )
            logger.error(f"History: {history}")
            raise

    def clear_conversation(self) -> None:
        """Clear conversation in client but retain history."""
        self.client.clear_conversation()

    # ===========================================================================
    # Structured output helpers
    # ===========================================================================

    def _execute_structured(
        self,
        handler: Any,
        response_model: type,
        prompt: str,
        model: str,
        kwargs: dict[str, Any],
        should_suspend: bool,
        saved_history: list[dict[str, Any]] | None,
    ) -> tuple[str, Any]:
        """Execute LLM call with structured output validation retry loop.

        Args:
            handler: StructuredOutputHandler instance.
            response_model: Pydantic BaseModel subclass.
            prompt: The resolved prompt to send.
            model: Model identifier.
            kwargs: Additional kwargs for the client.
            should_suspend: Whether client history suspension is active.
            saved_history: Saved client history to restore.

        Returns:
            Tuple of (raw_response, StructuredResult).

        """
        max_attempts = handler.max_retries + 1
        all_errors, current_prompt, best_result = handler.prepare_retry_state(
            prompt, response_model
        )

        for attempt in range(max_attempts):
            response = self.client.generate_response(
                prompt=current_prompt, model=model, **kwargs
            )

            best_result, current_prompt, all_errors, should_stop = (
                handler.process_attempt(
                    response, response_model, prompt, attempt, all_errors, best_result
                )
            )

            if should_stop:
                if should_suspend and saved_history is not None:
                    with self._client_history_lock:
                        new_msgs = self.client.get_conversation_history()
                        combined = list(saved_history) + list(new_msgs)
                        self.client.set_conversation_history(combined)
                        logger.debug(
                            f"Restored client conversation history (+{len(new_msgs)} new messages)"
                        )
                return response, best_result

        if should_suspend and saved_history is not None:
            with self._client_history_lock:
                new_msgs = self.client.get_conversation_history()
                combined = list(saved_history) + list(new_msgs)
                self.client.set_conversation_history(combined)
                logger.debug(
                    f"Restored client conversation history (+{len(new_msgs)} new messages)"
                )

        final_result = handler.finalize_retry(best_result, all_errors, max_attempts)
        return final_result.raw_response, final_result

    # ===========================================================================
    # Condition & DAG helpers
    # ===========================================================================

    def _build_results_by_name(self) -> dict[str, dict[str, Any]]:
        """Build a name->result dict from history for condition evaluation."""
        results: dict[str, dict[str, Any]] = {}
        for entry in self.history:
            name = entry.get("prompt_name")
            if name:
                response = entry.get("response")
                results[name] = {
                    "status": entry.get("status", "success"),
                    "response": str(response) if response is not None else "",
                    "attempts": 1,
                    "error": "",
                    "has_response": response is not None and len(str(response).strip()) > 0,
                }
        return results

    def _evaluate_condition(
        self, condition: str, results_by_name: dict[str, dict[str, Any]]
    ) -> tuple[bool, str | None, str | None]:
        """Evaluate a condition expression against known results."""
        evaluator = ConditionEvaluator(results_by_name)
        return evaluator.evaluate_with_trace(condition)

    def validate_graph(
        self,
        prompts: list[dict[str, Any]],
    ) -> tuple[ExecutionGraph, list[str]]:
        """Validate a prompt dependency graph without executing.

        Args:
            prompts: List of dicts with keys: prompt_name, prompt, history, condition.

        Returns:
            Tuple of (ExecutionGraph, list of warning strings).

        Raises:
            ValueError: If a dependency cycle is detected.

        """
        specs: list[dict[str, Any]] = [
            {
                "sequence": i,
                "prompt_name": p.get("prompt_name", f"unnamed_{i}"),
                "prompt": p.get("prompt", ""),
                "history": p.get("history"),
                "condition": p.get("condition"),
            }
            for i, p in enumerate(prompts)
        ]
        graph = build_execution_graph_with_edges(specs)

        history_edges = {
            (e.from_seq, e.to_seq) for e in graph.edges if e.source == "history"
        }
        warnings = []
        for edge in graph.edges:
            if edge.source == "condition" and (edge.from_seq, edge.to_seq) not in history_edges:
                from_name = graph.nodes[edge.from_seq].get_prompt_name()
                to_name = graph.nodes[edge.to_seq].get_prompt_name()
                warnings.append(
                    f"Undeclared dependency: {to_name} conditions on {from_name} "
                    f"but does not list it in history. Edge: {edge.condition_text}"
                )

        return graph, warnings

    # ===========================================================================
    # Async DAG execution
    # ===========================================================================

    async def execute_graph(
        self,
        prompts: list[dict[str, Any]],
        max_concurrency: int = 10,
    ) -> Any:
        """Execute a prompt dependency graph with topological-parallel async calls.

        Prompts on the same DAG level run concurrently via ``asyncio.gather``.
        Levels execute sequentially.  Requires an ``AsyncFFAIClientBase`` client.

        Supports the same declarative features as ``generate_response()``:
        ``{{name.response}}`` interpolation, ``history=`` context injection,
        ``condition`` / ``abort_condition`` gating, ``response_model`` for
        structured output, and per-prompt ``model`` / ``system_instructions``
        overrides.

        Args:
            prompts: List of dicts with keys: prompt_name, prompt, history,
                condition, abort_condition, response_model, system_instructions,
                model.
            max_concurrency: Maximum concurrent API calls.

        Returns:
            ``GraphResult`` with per-prompt ``ResponseResult`` instances and
            aggregate counts.

        Raises:
            TypeError: If the client is not an ``AsyncFFAIClientBase``.
            ValueError: If a dependency cycle is detected.

        """
        from .core.async_client_base import AsyncFFAIClientBase
        from .core.async_executor import AsyncGraphExecutor

        if not isinstance(self.client, AsyncFFAIClientBase):
            raise TypeError(
                f"execute_graph requires an async client. "
                f"Got {type(self.client).__name__}. "
                f"Use AsyncFFLiteLLMClient instead."
            )

        async_client: AsyncFFAIClientBase = self.client
        ffai_ref = self

        async def run_prompt(**spec: Any) -> ResponseResult:
            prompt_text = spec.get("prompt", "")
            spec_model = spec.get("model")
            spec_system = spec.get("system_instructions")
            response_model = spec.get("response_model")

            used_model = spec_model if spec_model else async_client.model

            cloned = await async_client.clone()

            saved_history = cloned.get_conversation_history()
            cloned.set_conversation_history([])

            kwargs: dict[str, Any] = {}
            if spec_system is not None:
                kwargs["system_instructions"] = spec_system

            so_result = None
            call_start = time.monotonic()

            try:
                if response_model is not None:
                    from pydantic import BaseModel as _BaseModel

                    if not (
                        isinstance(response_model, type)
                        and issubclass(response_model, _BaseModel)
                    ):
                        raise TypeError(
                            f"response_model must be a Pydantic BaseModel subclass, "
                            f"got {type(response_model)}"
                        )

                    from .core.structured_output import StructuredOutputHandler

                    so_handler = StructuredOutputHandler(max_retries=2)
                    kwargs["response_format"] = so_handler.build_response_format(
                        response_model
                    )
                    schema_suffix = so_handler.build_system_suffix(response_model)
                    current_system = kwargs.get("system_instructions") or ""
                    kwargs["system_instructions"] = current_system + schema_suffix

                    max_attempts = so_handler.max_retries + 1
                    all_errors, current_prompt, _best = so_handler.prepare_retry_state(
                        prompt_text, response_model
                    )
                    so_result = None

                    for attempt in range(max_attempts):
                        response = await cloned.generate_response(
                            prompt=current_prompt, model=used_model, **kwargs
                        )

                        _best, current_prompt, all_errors, should_stop = (
                            so_handler.process_attempt(
                                response, response_model, prompt_text,
                                attempt, all_errors, _best,
                            )
                        )

                        if should_stop:
                            so_result = _best
                            break

                    if so_result is None:
                        so_result = so_handler.finalize_retry(
                            _best, all_errors, max_attempts
                        )

                    response = so_result.raw_response
                else:
                    response = await cloned.generate_response(
                        prompt=prompt_text, model=used_model, **kwargs
                    )
            finally:
                cloned.set_conversation_history(saved_history)

            call_duration_ms = (time.monotonic() - call_start) * 1000
            cleaned = ffai_ref._clean_response(response)
            usage = getattr(cloned, "last_usage", None)
            cost_usd = getattr(cloned, "last_cost_usd", 0.0)

            return ResponseResult(
                response=cleaned,
                resolved_prompt=prompt_text,
                model=used_model,
                status="success",
                usage=usage,
                cost_usd=cost_usd,
                duration_ms=round(call_duration_ms, 1),
                parsed=so_result.parsed if so_result else None,
                parsing_errors=(
                    so_result.parsing_errors
                    if so_result and so_result.parsing_errors
                    else None
                ),
            )

        executor = AsyncGraphExecutor(
            executor_fn=run_prompt,
            max_concurrency=max_concurrency,
            prompt_resolver=resolve_graph_prompt,
        )

        graph_result = await executor.execute(prompts)

        for name, result in graph_result.results.items():
            if result.status == "success" and result.response is not None:
                self.history.append(
                    {
                        "prompt": result.resolved_prompt,
                        "response": result.response,
                        "prompt_name": name,
                        "timestamp": time.time(),
                        "model": result.model,
                        "history": None,
                        "status": "success",
                    }
                )
                self.clean_history.append(
                    {
                        "prompt": result.resolved_prompt,
                        "response": result.response,
                        "prompt_name": name,
                        "timestamp": time.time(),
                        "model": result.model,
                        "history": None,
                        "status": "success",
                    }
                )

                self.permanent_history.add_turn_user(result.resolved_prompt)
                self.permanent_history.add_turn_assistant(result.response)

                self._context.record(
                    prompt=result.resolved_prompt,
                    response=result.response,
                    model=result.model,
                    prompt_name=name,
                )

                self.ordered_history.add_interaction(
                    model=result.model,
                    prompt=result.resolved_prompt,
                    response=result.response,
                    prompt_name=name,
                )

        return graph_result

    # ===========================================================================
    # History accessors
    # ===========================================================================

    def get_interaction_history(self) -> list[dict[str, Any]]:
        """Get complete history."""
        return self.history

    def get_clean_interaction_history(self) -> list[dict[str, Any]]:
        """Get complete clean history."""
        return self.clean_history

    def get_prompt_attr_history(self) -> list[dict[str, Any]]:
        """Get prompt attribute history."""
        return self.prompt_attr_history

    def get_all_interactions(self) -> list[Any]:
        """Get all interactions as dictionaries."""
        return self.ordered_history.get_all_interactions()

    def get_latest_interaction_by_prompt_name(self, prompt_name: str) -> dict[str, Any] | None:
        """Get most recent interaction for a prompt name."""
        matching = [e for e in self.history if e.get("prompt_name") == prompt_name]
        return matching[-1] if matching else None

    def get_last_n_interactions(self, n: int) -> list[dict[str, Any]]:
        """Get the last n interactions as dictionaries."""
        all_interactions = self.ordered_history.get_all_interactions()
        return [i.to_dict() for i in all_interactions[-n:]]

    def get_interaction(self, sequence_number: int) -> dict[str, Any] | None:
        """Get a specific interaction by sequence number."""
        all_interactions = self.ordered_history.get_all_interactions()
        interaction = next(
            (i for i in all_interactions if i.sequence_number == sequence_number), None
        )
        return interaction.to_dict() if interaction else None

    def get_model_interactions(self, model: str) -> list[dict[str, Any]]:
        """Get all interactions for a specific model."""
        all_interactions = self.ordered_history.get_all_interactions()
        return [i.to_dict() for i in all_interactions if i.model == model]

    def get_interactions_by_prompt_name(self, prompt_name: str) -> list[dict[str, Any]]:
        """Get all interactions for a specific prompt name."""
        return [
            i.to_dict() for i in self.ordered_history.get_interactions_by_prompt_name(prompt_name)
        ]

    def get_latest_interaction(self) -> dict[str, Any] | None:
        """Get the most recent interaction."""
        all_interactions = self.ordered_history.get_all_interactions()
        return all_interactions[-1].to_dict() if all_interactions else None

    def get_prompt_history(self) -> list[str]:
        """Get all prompts in order."""
        return [i.prompt for i in self.ordered_history.get_all_interactions()]

    def get_response_history(self) -> list[str]:
        """Get all responses in order."""
        return [i.response for i in self.ordered_history.get_all_interactions()]

    def get_model_usage_stats(self) -> dict[str, int]:
        """Get statistics on model usage."""
        usage_stats: dict[str, int] = {}
        for interaction in self.ordered_history.get_all_interactions():
            usage_stats[interaction.model] = usage_stats.get(interaction.model, 0) + 1
        return usage_stats

    def get_prompt_name_usage_stats(self) -> dict[str, int]:
        """Get statistics on prompt name usage."""
        return self.ordered_history.get_prompt_name_usage_stats()

    def get_prompt_dict(self) -> dict[str, list[dict[str, Any]]]:
        """Get the complete history as an ordered dictionary keyed by prompts."""
        return self.ordered_history.to_dict()

    def get_latest_responses_by_prompt_names(
        self, prompt_names: list[str]
    ) -> dict[str, dict[str, str]]:
        """Get the latest prompt and response for each specified prompt name."""
        return self.ordered_history.get_latest_responses_by_prompt_names(prompt_names)

    def get_formatted_responses(self, prompt_names: list[str]) -> str:
        """Get formatted string output of latest prompts and responses."""
        return self.ordered_history.get_formatted_responses(prompt_names)

    # ===========================================================================
    # Client conversation history access
    # ===========================================================================

    def get_client_conversation_history(self) -> list[dict[str, str]]:
        """Get the raw conversation history from the underlying client."""
        logger.info("Retrieving raw conversation history from client")
        try:
            if hasattr(self.client, "get_conversation_history"):
                history = self.client.get_conversation_history()
                logger.debug(f"Retrieved conversation history: {history}")
                return history
            else:
                logger.warning("Client does not support retrieving conversation history")
                return []
        except Exception as e:
            logger.error(f"Error retrieving conversation history: {e!s}")
            return []

    def set_client_conversation_history(self, history: list[dict[str, str]]) -> bool:
        """Set the raw conversation history in the underlying client."""
        logger.info(f"Setting raw conversation history in client: {history}")
        try:
            if hasattr(self.client, "set_conversation_history"):
                self.client.set_conversation_history(history)
                logger.debug("Successfully set conversation history")
                return True
            else:
                logger.warning("Client does not support setting conversation history")
                return False
        except Exception as e:
            logger.error(f"Error setting conversation history: {e!s}")
            return False

    def add_client_message(self, role: str, content: str, **kwargs: Any) -> bool:
        """Add a single message to the client's conversation history."""
        logger.info(
            f"Adding message to client conversation history: role={role}, content={content}"
        )
        try:
            history = self.get_client_conversation_history()
            message = {"role": role, "content": content, **kwargs}
            history.append(message)
            return self.set_client_conversation_history(history)
        except Exception as e:
            logger.error(f"Error adding message to conversation history: {e!s}")
            return False

    # ===========================================================================
    # DataFrame export (delegated to HistoryExporter)
    # ===========================================================================

    def _convert_unix_seconds_to_datetime(self, df: pl.DataFrame) -> pl.DataFrame:
        """Convert Unix timestamps in seconds to datetime."""
        return self._exporter._convert_unix_seconds_to_datetime(df)

    def history_to_dataframe(self) -> pl.DataFrame:
        """Convert the full interaction history to a polars DataFrame."""
        return self._exporter.history_to_dataframe()

    def clean_history_to_dataframe(self) -> pl.DataFrame:
        """Convert the clean interaction history to a polars DataFrame."""
        return self._exporter.clean_history_to_dataframe()

    def prompt_attr_history_to_dataframe(self) -> pl.DataFrame:
        """Convert the prompt attribute history to a polars DataFrame."""
        return self._exporter.prompt_attr_history_to_dataframe()

    def ordered_history_to_dataframe(self) -> pl.DataFrame:
        """Convert the ordered interaction history to a polars DataFrame."""
        return self._exporter.ordered_history_to_dataframe()

    def search_history(
        self,
        text: str | None = None,
        prompt_name: str | None = None,
        model: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> pl.DataFrame:
        """Search interaction history with flexible filtering options."""
        return self._exporter.search_history(
            text=text,
            prompt_name=prompt_name,
            model=model,
            start_time=start_time,
            end_time=end_time,
        )

    def get_model_stats_df(self) -> pl.DataFrame:
        """Get statistics on model usage as a DataFrame."""
        return self._exporter.get_model_stats_df(self.get_model_usage_stats())

    def get_prompt_name_stats_df(self) -> pl.DataFrame:
        """Get statistics on prompt name usage as a DataFrame."""
        return self._exporter.get_prompt_name_stats_df(self.get_prompt_name_usage_stats())

    def get_response_length_stats(self) -> pl.DataFrame:
        """Get statistics on response lengths by prompt name."""
        return self._exporter.get_response_length_stats()

    def interaction_counts_by_date(self) -> pl.DataFrame:
        """Get counts of interactions grouped by date."""
        return self._exporter.interaction_counts_by_date()

    def persist_all_histories(self) -> bool:
        """Persist all histories to Parquet files in the configured directory."""
        return self._exporter.persist_all_histories()
