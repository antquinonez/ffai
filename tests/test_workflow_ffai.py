from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from ffai.core.async_client_base import AsyncFFAIClientBase
from ffai.FFAI import FFAI


class FakeAsyncClient(AsyncFFAIClientBase):
    def __init__(self, model: str = "test-model") -> None:
        self.model = model
        self.system_instructions = ""
        self._history: list[dict[str, Any]] = []
        self._last_usage = None
        self._last_cost_usd = 0.0
        self._mock_generate = AsyncMock(return_value="response from llm")

    async def generate_response(self, prompt: str, **kwargs: Any) -> str:
        return await self._mock_generate(prompt=prompt, **kwargs)

    async def clone(self) -> FakeAsyncClient:
        c = FakeAsyncClient(self.model)
        c._mock_generate = self._mock_generate
        return c

    def clear_conversation(self) -> None:
        self._history.clear()

    def get_conversation_history(self) -> list[dict[str, Any]]:
        return list(self._history)

    def set_conversation_history(self, history: list[dict[str, Any]]) -> None:
        self._history = list(history)


def _arun(coro):
    return asyncio.run(coro)


class TestExecuteWorkflow:
    def test_execute_from_yaml_string(self):
        client = FakeAsyncClient()
        ffai = FFAI(client)

        result = _arun(ffai.execute_workflow("""
workflow:
  name: basic
  prompts:
    - name: greet
      prompt: "Say hello"
"""))
        assert result.success_count == 1
        assert result.spec_name == "basic"
        assert "greet" in result.results
        assert result.results["greet"].status == "success"

    def test_execute_with_variables(self):
        client = FakeAsyncClient()
        ffai = FFAI(client)

        result = _arun(ffai.execute_workflow("""
workflow:
  name: var_test
  prompts:
    - name: q
      prompt: "Tell me about {topic}"
""", variables={"topic": "quantum computing"}))

        assert result.success_count == 1
        call_args = client._mock_generate.call_args
        assert "quantum computing" in call_args.kwargs.get("prompt", call_args[1].get("prompt", ""))

    def test_execute_workflow_file(self, tmp_path):
        client = FakeAsyncClient()
        ffai = FFAI(client)

        wf_file = tmp_path / "test.yaml"
        wf_file.write_text("""
workflow:
  name: file_test
  prompts:
    - name: step1
      prompt: "Hello"
""")
        result = _arun(ffai.execute_workflow_file(str(wf_file)))
        assert result.success_count == 1
        assert result.spec_name == "file_test"

    def test_execute_raises_on_non_async_client(self):
        sync_client = MagicMock(spec=["model", "generate_response"])
        sync_client.model = "sync-model"
        ffai = FFAI(sync_client)

        with pytest.raises(TypeError, match="async client"):
            _arun(ffai.execute_workflow("""
workflow:
  name: bad
  prompts:
    - name: s1
      prompt: "hi"
"""))

    def test_execute_file_not_found(self):
        client = FakeAsyncClient()
        ffai = FFAI(client)

        with pytest.raises(FileNotFoundError):
            _arun(ffai.execute_workflow_file("/nonexistent/workflow.yaml"))

    def test_invalid_yaml_raises_validation_error(self):
        from ffai.workflow import WorkflowValidationError

        client = FakeAsyncClient()
        ffai = FFAI(client)

        with pytest.raises(WorkflowValidationError):
            _arun(ffai.execute_workflow("workflow:\n  prompts: []"))


class TestValidateWorkflow:
    def test_valid_workflow_returns_no_errors(self):
        client = FakeAsyncClient()
        ffai = FFAI(client)

        errors, warnings = ffai.validate_workflow("""
workflow:
  name: valid
  prompts:
    - name: s1
      prompt: "Hello"
    - name: s2
      prompt: "World {{s1.response}}"
      history: [s1]
""")
        assert errors == []
        assert warnings == []

    def test_cycle_detected(self):
        client = FakeAsyncClient()
        ffai = FFAI(client)

        errors, warnings = ffai.validate_workflow("""
workflow:
  name: cyclic
  prompts:
    - name: a
      prompt: "A {{b.response}}"
      history: [b]
    - name: b
      prompt: "B {{a.response}}"
      history: [a]
""")
        assert len(errors) > 0
        assert any("cycle" in e.lower() for e in errors)

    def test_invalid_yaml_returns_parse_error(self):
        client = FakeAsyncClient()
        ffai = FFAI(client)

        errors, warnings = ffai.validate_workflow("workflow:\n  prompts: []")
        assert len(errors) > 0

    def test_unknown_client_produces_warning(self):
        client = FakeAsyncClient()
        ffai = FFAI(client)

        errors, warnings = ffai.validate_workflow("""
workflow:
  name: warn_test
  prompts:
    - name: s1
      prompt: "hi"
      client: totally_unknown_client
""")
        assert errors == []
        assert len(warnings) > 0
        assert any("totally_unknown_client" in w for w in warnings)
