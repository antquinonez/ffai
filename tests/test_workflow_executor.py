from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ffai.core.async_client_base import AsyncFFAIClientBase
from ffai.workflow import (
    ClientFactory,
    WorkflowExecutor,
    WorkflowResult,
    WorkflowSpec,
    load_workflow,
)
from ffai.workflow.spec import ClientRef


def _make_spec(yaml_text: str) -> WorkflowSpec:
    return load_workflow(yaml_text)


class FakeAsyncClient(AsyncFFAIClientBase):
    def __init__(self, model: str = "test-model") -> None:
        self.model = model
        self.system_instructions = ""
        self._history: list[dict[str, Any]] = []
        self._last_usage = None
        self._last_cost_usd = 0.0
        self._mock_generate = AsyncMock(return_value="response")
        self._mock_clone = AsyncMock()

    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        return await self._mock_generate(prompt=prompt, **kwargs)

    async def clone(self) -> FakeAsyncClient:
        c = FakeAsyncClient(self.model)
        return await self._mock_clone() or c

    def clear_conversation(self) -> None:
        self._history.clear()

    def get_conversation_history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def set_conversation_history(self, history: list[dict[str, Any]]) -> None:
        self._history = list(history)


def _make_engine(client=None):
    engine = MagicMock()
    engine.client = client or FakeAsyncClient()
    engine.clean_response_fn = lambda x: x
    engine.recorder = MagicMock()
    engine.executor = MagicMock()
    return engine


class TestClientFactory:
    def test_none_ref_returns_default(self):
        default = FakeAsyncClient("default")
        factory = ClientFactory(ffai_client=default, async_mode=False)
        assert factory.resolve(None) is default

    def test_named_ref_from_workflow_clients(self):
        default = FakeAsyncClient("default")
        workflow_clients = {
            "my-client": ClientRef(
                type="litellm", model="gpt-4o", api_key_env="TEST_KEY"
            ),
        }
        factory = ClientFactory(
            ffai_client=default,
            workflow_clients=workflow_clients,
            async_mode=False,
        )
        client = factory.resolve(ClientRef(name="my-client"))
        assert client is not default
        assert client.model == "gpt-4o"

    def test_named_ref_caches(self):
        default = FakeAsyncClient("default")
        workflow_clients = {
            "cached": ClientRef(type="litellm", model="gpt-4o"),
        }
        factory = ClientFactory(
            ffai_client=default,
            workflow_clients=workflow_clients,
            async_mode=False,
        )
        ref = ClientRef(name="cached")
        first = factory.resolve(ref)
        second = factory.resolve(ref)
        assert first is second

    def test_inline_ref_creates_new_client(self):
        default = FakeAsyncClient("default")
        factory = ClientFactory(ffai_client=default, async_mode=False)
        ref = ClientRef(type="litellm", model="gpt-4o")
        client = factory.resolve(ref)
        assert client is not default

    def test_named_ref_falls_back_to_default(self):
        default = FakeAsyncClient("default")
        factory = ClientFactory(
            ffai_client=default,
            workflow_clients={},
            async_mode=False,
        )
        with patch("ffai.workflow.client_factory.get_config") as mock_cfg:
            mock_cfg.return_value.clients.get_client_type.return_value = None
            client = factory.resolve(ClientRef(name="unknown"))
        assert client is default


class TestSubstituteVariables:
    def test_substitutes_single_variable(self):
        result = WorkflowExecutor._substitute_variables(
            "Research {topic}.", {"topic": "AI safety"}
        )
        assert result == "Research AI safety."

    def test_preserves_double_brace_interpolation(self):
        result = WorkflowExecutor._substitute_variables(
            "Research {topic}. {{research.response}}", {"topic": "AI"}
        )
        assert result == "Research AI. {{research.response}}"

    def test_leaves_unset_variables_unchanged(self):
        result = WorkflowExecutor._substitute_variables(
            "Hello {name} and {place}", {"name": "World"}
        )
        assert result == "Hello World and {place}"

    def test_no_variables_returns_prompt_unchanged(self):
        prompt = "No variables {{here.response}}"
        result = WorkflowExecutor._substitute_variables(prompt, {})
        assert result == prompt

    def test_multiple_substitutions(self):
        result = WorkflowExecutor._substitute_variables(
            "{a} and {b}", {"a": "X", "b": "Y"}
        )
        assert result == "X and Y"


class TestBuildSpecs:
    def test_produces_valid_spec_dicts(self):
        spec = _make_spec("""
workflow:
  name: test
  defaults:
    max_concurrency: 3
  prompts:
    - name: step1
      prompt: "Hello {name}"
    - name: step2
      prompt: "Analyze {{step1.response}}"
      history: [step1]
      condition: '{{step1.status}} == "success"'
""")
        engine = _make_engine()
        executor = WorkflowExecutor(engine=engine, spec=spec)

        with patch.object(
            type(executor._client_factory),
            "resolve",
            return_value=engine.client,
        ):
            specs = executor._build_specs({"name": "World"})

        assert len(specs) == 2
        assert specs[0]["sequence"] == 0
        assert specs[0]["prompt_name"] == "step1"
        assert specs[0]["prompt"] == "Hello World"
        assert specs[1]["prompt_name"] == "step2"
        assert specs[1]["history"] == ["step1"]
        assert specs[1]["condition"] == '{{step1.status}} == "success"'

    def test_merges_defaults(self):
        spec = _make_spec("""
workflow:
  name: test
  defaults:
    system_instructions: "Be helpful"
    max_tokens: 2048
    temperature: 0.5
  prompts:
    - name: step1
      prompt: "Hello"
""")
        engine = _make_engine()
        executor = WorkflowExecutor(engine=engine, spec=spec)

        with patch.object(
            type(executor._client_factory),
            "resolve",
            return_value=engine.client,
        ):
            specs = executor._build_specs({})

        assert specs[0]["system_instructions"] == "Be helpful"
        assert specs[0]["_provider_kwargs"]["max_tokens"] == 2048
        assert specs[0]["_provider_kwargs"]["temperature"] == 0.5

    def test_step_overrides_defaults(self):
        spec = _make_spec("""
workflow:
  name: test
  defaults:
    system_instructions: "Default"
    temperature: 0.7
  prompts:
    - name: step1
      prompt: "Hello"
      system_instructions: "Custom"
      temperature: 0.3
""")
        engine = _make_engine()
        executor = WorkflowExecutor(engine=engine, spec=spec)

        with patch.object(
            type(executor._client_factory),
            "resolve",
            return_value=engine.client,
        ):
            specs = executor._build_specs({})

        assert specs[0]["system_instructions"] == "Custom"
        assert specs[0]["_provider_kwargs"]["temperature"] == 0.3

    def test_tools_produce_provider_kwargs(self):
        spec = _make_spec("""
workflow:
  name: test
  tools:
    search:
      description: "Search"
      parameters:
        type: object
        properties:
          query:
            type: string
  prompts:
    - name: step1
      prompt: "Search"
      tools: ["search"]
      tool_choice: "auto"
""")
        engine = _make_engine()
        executor = WorkflowExecutor(engine=engine, spec=spec)

        with patch.object(
            type(executor._client_factory),
            "resolve",
            return_value=engine.client,
        ):
            specs = executor._build_specs({})

        assert "tools" in specs[0]["_provider_kwargs"]
        assert specs[0]["_provider_kwargs"]["tool_choice"] == "auto"


class TestWorkflowResult:
    def test_from_graph_result(self):
        from ffai.core.async_executor import GraphResult
        from ffai.core.response_result import ResponseResult

        graph_result = GraphResult(
            results={"s1": ResponseResult(response="ok", status="success")},
            success_count=1,
            failed_count=0,
            skipped_count=0,
            aborted=False,
            aborted_count=0,
        )
        result = WorkflowResult.from_graph_result(graph_result, "my-workflow")
        assert result.spec_name == "my-workflow"
        assert result.success_count == 1
        assert "s1" in result.results


class TestWorkflowExecutorExecute:
    def test_requires_async_client(self):
        sync_client = MagicMock(spec=["model", "generate_response"])
        sync_client.model = "sync-model"
        engine = _make_engine(sync_client)

        spec = _make_spec("""
workflow:
  name: test
  prompts:
    - name: s1
      prompt: "hi"
""")
        executor = WorkflowExecutor(engine=engine, spec=spec)

        with pytest.raises(TypeError, match="async client"):
            asyncio.run(executor.execute())
