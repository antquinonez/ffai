from __future__ import annotations

import importlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..core.async_client_base import AsyncFFAIClientBase
from ..core.async_executor import AsyncGraphExecutor, GraphResult
from ..core.graph_execution_helpers import resolve_graph_prompt
from ..core.response_options import ResponseOptions
from ..core.response_result import ResponseResult
from ..tools.tool_registry import ToolDefinition, ToolRegistry
from .client_factory import ClientFactory
from .spec import WorkflowSpec

logger = logging.getLogger(__name__)

_VARIABLE_PATTERN = re.compile(r"(?<!\{)\{(\w+)\}(?!\})")


@dataclass
class WorkflowResult:
    results: dict[str, ResponseResult] = field(default_factory=dict)
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    aborted: bool = False
    aborted_count: int = 0
    spec_name: str = ""

    @classmethod
    def from_graph_result(cls, graph_result: GraphResult, spec_name: str) -> WorkflowResult:
        return cls(
            results=graph_result.results,
            success_count=graph_result.success_count,
            failed_count=graph_result.failed_count,
            skipped_count=graph_result.skipped_count,
            aborted=graph_result.aborted,
            aborted_count=graph_result.aborted_count,
            spec_name=spec_name,
        )


class WorkflowExecutor:
    def __init__(
        self,
        ffai: Any,
        spec: WorkflowSpec,
        client_factory: ClientFactory | None = None,
    ) -> None:
        self._ffai = ffai
        self._spec = spec

        if client_factory is not None:
            self._client_factory = client_factory
        else:
            self._client_factory = ClientFactory(
                ffai_client=ffai.client,
                workflow_clients=spec.clients,
                async_mode=True,
            )

        self._tool_registry = self._build_tool_registry(spec)

    async def execute(
        self,
        variables: dict[str, str] | None = None,
        max_concurrency: int | None = None,
    ) -> WorkflowResult:
        if not isinstance(self._ffai.client, AsyncFFAIClientBase):
            raise TypeError(
                "Workflow execution requires an async client. "
                "Use AsyncFFLiteLLMClient."
            )

        concurrency = max_concurrency or self._spec.defaults.max_concurrency

        step_clients = self._resolve_step_clients()

        specs = self._build_specs(variables or {})

        async def run_step(**spec: Any) -> ResponseResult:
            step_name = spec.get("prompt_name", "")
            client = step_clients.get(step_name, self._ffai.client)

            if not isinstance(client, AsyncFFAIClientBase):
                raise TypeError(
                    f"Client for step '{step_name}' is not async: "
                    f"{type(client).__name__}"
                )

            raw_opts = ResponseOptions.from_dict(spec)

            opts = raw_opts
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

            provider_kwargs = spec.get("_provider_kwargs", {})

            used_model = opts.model or client.model
            cloned = await client.clone()
            saved_history = cloned.get_conversation_history()
            cloned.set_conversation_history([])

            original_generate = cloned.generate_response

            async def generate_with_kwargs(
                prompt: str, **kw: Any
            ) -> str:
                merged = {**provider_kwargs, **kw}
                return await original_generate(prompt=prompt, **merged)

            try:
                exec_result = await self._ffai._executor.execute_async(
                    generate_fn=generate_with_kwargs,
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
                response=self._ffai._clean_response(exec_result.response),
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
            executor_fn=run_step,
            max_concurrency=concurrency,
            prompt_resolver=resolve_graph_prompt,
        )

        graph_result = await executor.execute(specs)

        for name, result in graph_result.results.items():
            if result.status == "success" and result.response is not None:
                self._ffai._recorder.record(
                    prompt=result.resolved_prompt,
                    response=result.response,
                    model=result.model,
                    prompt_name=name,
                    status="success",
                    resolved_prompt=result.resolved_prompt,
                    usage=result.usage,
                )

        return WorkflowResult.from_graph_result(graph_result, self._spec.name)

    def _resolve_step_clients(self) -> dict[str, AsyncFFAIClientBase]:
        step_clients: dict[str, AsyncFFAIClientBase] = {}
        for step in self._spec.prompts:
            client = self._client_factory.resolve(step.client)
            if not isinstance(client, AsyncFFAIClientBase):
                raise TypeError(
                    f"Client for step '{step.name}' is not async: "
                    f"{type(client).__name__}"
                )
            step_clients[step.name] = client
        return step_clients

    def _build_specs(
        self, variables: dict[str, str]
    ) -> list[dict[str, Any]]:
        specs: list[dict[str, Any]] = []

        for i, step in enumerate(self._spec.prompts):
            prompt = self._substitute_variables(step.prompt, variables)

            spec: dict[str, Any] = {
                "sequence": i,
                "prompt_name": step.name,
                "prompt": prompt,
            }

            if step.history is not None:
                spec["history"] = step.history

            if step.condition is not None:
                spec["condition"] = step.condition

            if step.abort_condition is not None:
                spec["abort_condition"] = step.abort_condition

            if step.system_instructions is not None:
                spec["system_instructions"] = step.system_instructions
            elif self._spec.defaults.system_instructions is not None:
                spec["system_instructions"] = self._spec.defaults.system_instructions

            if step.response_format is not None:
                spec["response_format"] = step.response_format

            if step.response_model is not None:
                spec["response_model"] = self._resolve_response_model(
                    step.response_model
                )

            if step.model is not None:
                spec["model"] = step.model

            if step.strict or self._spec.defaults.strict:
                spec["strict"] = True

            kwargs: dict[str, Any] = {}
            if step.max_tokens is not None:
                kwargs["max_tokens"] = step.max_tokens
            elif self._spec.defaults.max_tokens is not None:
                kwargs["max_tokens"] = self._spec.defaults.max_tokens

            if step.temperature is not None:
                kwargs["temperature"] = step.temperature
            elif self._spec.defaults.temperature is not None:
                kwargs["temperature"] = self._spec.defaults.temperature

            if step.tools:
                tool_schemas = self._tool_registry.get_tools_schema(step.tools)
                kwargs["tools"] = tool_schemas
                if step.tool_choice:
                    kwargs["tool_choice"] = step.tool_choice

            if kwargs:
                spec["_provider_kwargs"] = kwargs

            specs.append(spec)

        return specs

    @staticmethod
    def _substitute_variables(prompt: str, variables: dict[str, str]) -> str:
        if not variables:
            return prompt

        def _replacer(match: re.Match[str]) -> str:
            key = match.group(1)
            return variables.get(key, match.group(0))

        return _VARIABLE_PATTERN.sub(_replacer, prompt)

    @staticmethod
    def _build_tool_registry(spec: WorkflowSpec) -> ToolRegistry:
        registry = ToolRegistry()
        for _name, tdata in spec.tools.items():
            registry.register(ToolDefinition.from_dict(tdata))
        return registry

    @staticmethod
    def _resolve_response_model(dotted_path: str) -> type:
        module_path, class_name = dotted_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
