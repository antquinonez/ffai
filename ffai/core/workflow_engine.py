# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from ..config import get_config
from .async_client_base import AsyncFFAIClientBase
from .async_executor import AsyncGraphExecutor
from .client_base import FFAIClientBase
from .conversation_manager import ConversationManager
from .graph import ExecutionGraph, build_execution_graph_with_edges
from .graph_execution_helpers import resolve_graph_prompt
from .history.recorder import HistoryRecorder
from .prompt_builder import PromptBuilder
from .response_executor import ResponseExecutor
from .response_options import ResponseOptions
from .response_result import ResponseResult

logger = logging.getLogger(__name__)


class WorkflowEngine:
    def __init__(
        self,
        client: FFAIClientBase,
        conversation: ConversationManager,
        recorder: HistoryRecorder,
        prompt_attr_history: list[dict[str, Any]],
        clean_response_fn: Callable[[Any], Any],
    ) -> None:
        self._client = client
        self._conversation = conversation
        self._recorder = recorder
        self._clean_response_fn = clean_response_fn

        self._prompt_builder = PromptBuilder(prompt_attr_history)
        self._executor = ResponseExecutor(
            prompt_builder=self._prompt_builder.build_prompt,
            condition_evaluator_fn=self._evaluate_condition,
            results_by_name_fn=self._build_results_by_name,
        )

    @property
    def client(self) -> FFAIClientBase:
        return self._client

    @client.setter
    def client(self, value: FFAIClientBase) -> None:
        self._client = value

    @property
    def executor(self) -> ResponseExecutor:
        return self._executor

    @property
    def recorder(self) -> HistoryRecorder:
        return self._recorder

    @property
    def clean_response_fn(self) -> Callable[[Any], Any]:
        return self._clean_response_fn

    def build_prompt(
        self,
        prompt: str,
        history: list[str] | None = None,
        dependencies: Any | None = None,
        strict: bool = False,
    ) -> tuple[str, set[str]]:
        return self._prompt_builder.build_prompt(prompt, history, dependencies, strict=strict)

    def generate_response(
        self,
        prompt: str,
        prompt_name: str | None = None,
        history: list[str] | None = None,
        options: ResponseOptions | None = None,
        **kwargs: Any,
    ) -> ResponseResult:
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
        used_model = opts.model or self._client.model

        logger.debug(
            "\n==================================================================================="
        )
        prompt_preview = prompt[:80] + "..." if len(prompt) > 80 else prompt
        logger.info(
            f"Generating response for '{prompt_name}' ({len(prompt)} chars): '{prompt_preview}'"
        )
        logger.debug(f"Full prompt: '{prompt}'")
        logger.debug(f"Using model: {used_model}")

        saved = None
        if self._conversation.should_suspend(prompt, opts.history):
            reason = "history injection" if opts.history else "interpolation"
            saved = self._conversation.suspend(reason=reason)

        base_fn = self._client.generate_response
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
            self._conversation.restore(saved)

        usage = getattr(self._client, "last_usage", None)
        cost_usd = getattr(self._client, "last_cost_usd", 0.0)
        exec_result.usage = usage
        exec_result.cost_usd = cost_usd

        cleaned = self._clean_response_fn(exec_result.response)
        self._recorder.record(
            prompt=prompt,
            response=cleaned,
            model=used_model,
            prompt_name=prompt_name,
            history=opts.history,
            status=exec_result.status,
            resolved_prompt=exec_result.resolved_prompt,
            usage=usage,
        )

        return ResponseResult(
            response=self._clean_response_fn(exec_result.response),
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

    def _build_results_by_name(self) -> dict[str, dict[str, Any]]:
        results: dict[str, dict[str, Any]] = {}
        for entry in self._recorder.history:
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
        from .condition_evaluator import ConditionEvaluator

        evaluator = ConditionEvaluator(results_by_name)
        return evaluator.evaluate_with_trace(condition)

    def validate_graph(
        self,
        prompts: list[dict[str, Any]],
    ) -> tuple[ExecutionGraph, list[str]]:
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

    async def execute_graph(
        self,
        prompts: list[dict[str, Any]],
        max_concurrency: int = 10,
    ) -> Any:
        if not isinstance(self._client, AsyncFFAIClientBase):
            raise TypeError(
                f"execute_graph requires an async client. "
                f"Got {type(self._client).__name__}. "
                f"Use AsyncFFLiteLLMClient instead."
            )

        async_client: AsyncFFAIClientBase = self._client

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
                response=self._clean_response_fn(exec_result.response),
                resolved_prompt=exec_result.resolved_prompt,
                model=used_model,
                status=exec_result.status,
                usage=usage,
                cost_usd=cost_usd,
                duration_ms=exec_result.duration_ms,
                parsed=exec_result.parsed,
                parsing_errors=exec_result.parsing_errors,
            )

        specs = [
            {**p, "sequence": p.get("sequence", i)}
            for i, p in enumerate(prompts)
        ]

        executor = AsyncGraphExecutor(
            executor_fn=run_prompt,
            max_concurrency=max_concurrency,
            prompt_resolver=resolve_graph_prompt,
        )

        graph_result = await executor.execute(specs)

        for name, result in graph_result.results.items():
            if result.status == "success" and result.response is not None:
                self._recorder.record(
                    prompt=result.resolved_prompt,
                    response=result.response,
                    model=result.model,
                    prompt_name=name,
                    status="success",
                    resolved_prompt=result.resolved_prompt,
                    usage=result.usage,
                )

        return graph_result

    async def execute_workflow(
        self,
        workflow: Any,
        variables: dict[str, str] | None = None,
        max_concurrency: int | None = None,
    ) -> Any:
        from ..workflow.executor import WorkflowExecutor
        from ..workflow.loader import load_workflow

        spec = load_workflow(workflow) if isinstance(workflow, str) else workflow

        executor = WorkflowExecutor(engine=self, spec=spec)
        return await executor.execute(
            variables=variables,
            max_concurrency=max_concurrency,
        )

    async def execute_workflow_file(
        self,
        path: str,
        variables: dict[str, str] | None = None,
        max_concurrency: int | None = None,
    ) -> Any:
        from ..workflow.loader import load_workflow_file

        spec = load_workflow_file(path)
        return await self.execute_workflow(
            spec, variables=variables, max_concurrency=max_concurrency,
        )

    def validate_workflow(
        self,
        workflow: Any,
    ) -> tuple[list[str], list[str]]:
        from ..workflow.executor import WorkflowExecutor
        from ..workflow.loader import WorkflowValidationError, load_workflow

        if isinstance(workflow, str):
            try:
                spec = load_workflow(workflow)
            except WorkflowValidationError as e:
                return [str(e)], []
        else:
            spec = workflow

        errors: list[str] = []
        warnings: list[str] = []

        executor = WorkflowExecutor(engine=self, spec=spec)
        try:
            specs = executor._build_specs({})
            _graph, graph_warnings = self.validate_graph(specs)
            warnings.extend(graph_warnings)
        except ValueError as e:
            errors.append(str(e))

        for step in spec.prompts:
            if step.client is not None and step.client.is_named_ref:
                name = step.client.name
                if name and name not in spec.clients:
                    config = get_config()
                    if not config.clients.get_client_type(name):
                        warnings.append(
                            f"Client '{name}' referenced by step '{step.name}' "
                            f"not found in workflow or config — will use default"
                        )

        return errors, warnings

    async def execute_workflow_csv(
        self,
        csv_text: str,
        *,
        variables: dict[str, str] | None = None,
        max_concurrency: int | None = None,
        delimiter: str = ",",
        clients: dict[str, dict[str, Any] | str] | None = None,
        defaults: dict[str, Any] | None = None,
        tools: dict[str, dict[str, Any]] | None = None,
        name: str = "unnamed",
        description: str = "",
    ) -> Any:
        from ..workflow.executor import WorkflowExecutor
        from ..workflow.tabular_csv import load_workflow_csv

        spec = load_workflow_csv(
            csv_text,
            name=name,
            description=description,
            defaults=defaults,
            clients=clients,
            tools=tools,
            delimiter=delimiter,
        )
        executor = WorkflowExecutor(engine=self, spec=spec)
        return await executor.execute(
            variables=variables, max_concurrency=max_concurrency
        )

    async def execute_workflow_csv_file(
        self,
        path: str,
        *,
        variables: dict[str, str] | None = None,
        max_concurrency: int | None = None,
        delimiter: str = ",",
        encoding: str = "utf-8",
        clients: dict[str, dict[str, Any] | str] | None = None,
        defaults: dict[str, Any] | None = None,
        tools: dict[str, dict[str, Any]] | None = None,
        name: str = "unnamed",
        description: str = "",
    ) -> Any:
        from ..workflow.executor import WorkflowExecutor
        from ..workflow.tabular_csv import load_workflow_csv_file

        spec = load_workflow_csv_file(
            path,
            name=name,
            description=description,
            defaults=defaults,
            clients=clients,
            tools=tools,
            delimiter=delimiter,
            encoding=encoding,
        )
        executor = WorkflowExecutor(engine=self, spec=spec)
        return await executor.execute(
            variables=variables, max_concurrency=max_concurrency
        )
