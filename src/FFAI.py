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
from .core.execution_result import ExecutionResult
from .core.graph import ExecutionGraph, build_execution_graph_with_edges
from .core.graph_execution_helpers import resolve_graph_prompt
from .core.history.ordered import OrderedPromptHistory
from .core.history.permanent import PermanentHistory
from .core.history_exporter import HistoryExporter
from .core.prompt_builder import PromptBuilder
from .core.prompt_utils import extract_json_field, interpolate_prompt
from .core.response_context import ResponseContext
from .core.response_executor import ResponseExecutor
from .core.response_options import ResponseOptions
from .core.response_result import ResponseResult
from .core.response_utils import clean_response as _clean_response_impl
from .core.response_utils import extract_json

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
        self._executor = ResponseExecutor(
            prompt_builder=self._prompt_builder.build_prompt,
            condition_evaluator_fn=self._evaluate_condition,
            results_by_name_fn=self._build_results_by_name,
        )
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
        self._context.prompt_attr_history = value

    @property
    def _history_lock(self) -> threading.Lock | None:
        return self._context.history_lock

    def set_client(self, client: FFAIClientBase) -> None:
        """Switch to a different AI client."""
        logger.info(f"Switching client to {client.__class__.__name__}")
        self.client = client

    def _extract_json(self, text: str) -> Any | None:
        return extract_json(text)

    def get_system_instructions(self) -> str | None:
        if hasattr(self.client, "system_instructions"):
            return self.client.system_instructions
        return None

    def _clean_response(self, response: Any) -> Any:
        return _clean_response_impl(response)

    def build_prompt(
        self,
        prompt: str,
        history: list[str] | None = None,
        dependencies: Any | None = None,
        strict: bool = False,
    ) -> tuple[str, set[str]]:
        """Public API for prompt building (delegates to PromptBuilder)."""
        return self._prompt_builder.build_prompt(prompt, history, dependencies, strict=strict)

    # ===========================================================================
    # Core: generate_response
    # ===========================================================================

    def generate_response(
        self,
        prompt: str,
        prompt_name: str | None = None,
        history: list[str] | None = None,
        options: ResponseOptions | None = None,
        **kwargs: Any,
    ) -> ResponseResult:
        """Generate response using the configured AI client.

        Args:
            prompt: The user prompt (may contain ``{{name.response}}`` patterns).
            prompt_name: Logical name for the prompt (used for interpolation).
            history: List of prompt names to include in conversation context.
                Top-level takes precedence over ``options.history``.
            options: Configuration for this call.  Groups model override,
                dependencies, system_instructions, response_format,
                response_model, condition, abort_condition, and strict.
            **kwargs: Additional provider-specific parameters (temperature,
                max_tokens, tools, tool_choice, etc.).

        Returns:
            ``ResponseResult`` with response, resolved prompt, usage, and cost.

        """
        opts = options or ResponseOptions()
        if history is not None:
            opts = ResponseOptions(
                model=opts.model,
                system_instructions=opts.system_instructions,
                response_format=opts.response_format,
                response_model=opts.response_model,
                condition=opts.condition,
                abort_condition=opts.abort_condition,
                strict=opts.strict,
                history=history,
                dependencies=opts.dependencies,
            )
        used_model = opts.model or self.client.model

        logger.debug(
            "\n==================================================================================="
        )
        prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
        logger.info(
            f"Generating response for '{prompt_name}' ({len(prompt)} chars): '{prompt_preview}'"
        )
        logger.debug(f"Full prompt: '{prompt}'")
        logger.debug(f"Using model: {used_model}")

        should_suspend, saved_client_history = self._maybe_suspend_history(opts, prompt)

        base_fn = self.client.generate_response
        provider_kwargs = kwargs

        def dispatch(prompt: str, model: str, **extra: Any) -> str:
            merged = {**provider_kwargs, **extra}
            return base_fn(prompt=prompt, model=model, **merged)

        try:
            exec_result = self._executor.execute(
                generate_fn=dispatch,
                prompt=prompt,
                prompt_name=prompt_name,
                options=opts,
                default_model=used_model,
            )
        except Exception as e:
            logger.error(f"Problem with response generation: {e!s}")
            logger.error(f"Prompt: {prompt}")
            logger.error(f"Model: {used_model}")
            logger.error(f"Prompt name: {prompt_name}")
            raise
        finally:
            self._maybe_restore_history(should_suspend, saved_client_history)

        usage = getattr(self.client, "last_usage", None)
        cost_usd = getattr(self.client, "last_cost_usd", 0.0)
        exec_result.usage = usage
        exec_result.cost_usd = cost_usd

        self._record_interaction(prompt, exec_result, prompt_name, opts, used_model)

        return ResponseResult(
            response=self._clean_response(exec_result.response),
            resolved_prompt=exec_result.resolved_prompt,
            usage=exec_result.usage,
            cost_usd=exec_result.cost_usd,
            model=used_model,
            duration_ms=exec_result.duration_ms,
            status=exec_result.status,
            condition_trace=exec_result.condition_trace,
            condition_error=exec_result.condition_error,
            parsed=exec_result.parsed,
            parsing_errors=exec_result.parsing_errors,
        )

    # ===========================================================================
    # History suspension helpers
    # ===========================================================================

    def _maybe_suspend_history(
        self, options: ResponseOptions, prompt: str
    ) -> tuple[bool, list[dict[str, Any]] | None]:
        has_interpolation = "{{" in prompt and "}}" in prompt
        should_suspend = options.history is not None or has_interpolation
        if not should_suspend:
            return False, None
        with self._client_history_lock:
            saved = self.client.get_conversation_history().copy()
            self.client.set_conversation_history([])
        reason = "history injection" if options.history else "interpolation"
        logger.debug(f"Suspended client conversation history: {reason}")
        return True, saved

    def _maybe_restore_history(
        self, should_suspend: bool, saved: list[dict[str, Any]] | None
    ) -> None:
        if not should_suspend or saved is None:
            return
        with self._client_history_lock:
            new_msgs = self.client.get_conversation_history()
            combined = list(saved) + list(new_msgs)
            self.client.set_conversation_history(combined)
            logger.debug(
                f"Restored client conversation history (+{len(new_msgs)} new messages)"
            )

    def _record_interaction(
        self,
        prompt: str,
        result: ExecutionResult,
        prompt_name: str | None,
        options: ResponseOptions,
        used_model: str,
    ) -> None:
        cleaned = self._clean_response(result.response)

        self.permanent_history.add_turn_user(prompt)
        self.permanent_history.add_turn_assistant(cleaned)

        interaction: dict[str, Any] = {
            "prompt": prompt,
            "response": cleaned,
            "prompt_name": prompt_name,
            "timestamp": time.time(),
            "model": used_model,
            "history": options.history,
            "status": result.status,
        }

        self.history.append(interaction)
        self.clean_history.append(interaction)

        self._context.record(prompt, cleaned, used_model, prompt_name, options.history)

        self.ordered_history.add_interaction(
            model=used_model,
            prompt=prompt,
            response=cleaned,
            prompt_name=prompt_name,
            history=options.history,
        )

    # ===========================================================================
    # Condition & DAG helpers
    # ===========================================================================

    def _build_results_by_name(self) -> dict[str, dict[str, Any]]:
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
        from .core.condition_evaluator import ConditionEvaluator

        evaluator = ConditionEvaluator(results_by_name)
        return evaluator.evaluate_with_trace(condition)

    def validate_graph(
        self,
        prompts: list[dict[str, Any]],
    ) -> tuple[ExecutionGraph, list[str]]:
        """Validate a prompt dependency graph without executing."""
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
        """Execute a prompt dependency graph with topological-parallel async calls."""
        from .core.async_client_base import AsyncFFAIClientBase
        from .core.async_executor import AsyncGraphExecutor

        if not isinstance(self.client, AsyncFFAIClientBase):
            raise TypeError(
                f"execute_graph requires an async client. "
                f"Got {type(self.client).__name__}. "
                f"Use AsyncFFLiteLLMClient instead."
            )

        async_client: AsyncFFAIClientBase = self.client

        async def run_prompt(**spec: Any) -> ResponseResult:
            raw_opts = ResponseOptions.from_dict(spec)

            if raw_opts.history:
                opts = ResponseOptions(
                    model=raw_opts.model,
                    system_instructions=raw_opts.system_instructions,
                    response_format=raw_opts.response_format,
                    response_model=raw_opts.response_model,
                    condition=raw_opts.condition,
                    abort_condition=raw_opts.abort_condition,
                    strict=raw_opts.strict,
                    history=None,
                    dependencies=raw_opts.dependencies,
                )
            else:
                opts = raw_opts

            used_model = opts.model or async_client.model

            cloned = await async_client.clone()
            saved_history = cloned.get_conversation_history()
            cloned.set_conversation_history([])

            try:
                exec_result = await self._executor.execute_async(
                    generate_fn=cloned.generate_response,
                    prompt=spec.get("prompt", ""),
                    prompt_name=spec.get("prompt_name"),
                    options=opts,
                    default_model=used_model,
                    skip_condition=True,
                )
            finally:
                cloned.set_conversation_history(saved_history)

            usage = getattr(cloned, "last_usage", None)
            cost_usd = getattr(cloned, "last_cost_usd", 0.0)

            return ResponseResult(
                response=self._clean_response(exec_result.response),
                resolved_prompt=exec_result.resolved_prompt,
                model=used_model,
                status=exec_result.status,
                usage=usage,
                cost_usd=cost_usd,
                duration_ms=exec_result.duration_ms,
                parsed=exec_result.parsed,
                parsing_errors=exec_result.parsing_errors,
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
    # Client conversation history access
    # ===========================================================================

    def clear_conversation(self) -> None:
        """Clear conversation in client but retain history."""
        self.client.clear_conversation()

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
