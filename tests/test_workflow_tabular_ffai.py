from __future__ import annotations

import asyncio
import tempfile
from typing import Any
from unittest.mock import AsyncMock

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
        self._mock_generate = AsyncMock(return_value="4")

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


class TestExecuteWorkflowCsv:
    def test_single_step(self):
        client = FakeAsyncClient()
        ffai = FFAI(client)
        csv_text = "name,prompt\ngreet,What is 2+2? Answer with just the number."
        result = _arun(ffai.workflow.execute_workflow_csv(csv_text))
        assert result.success_count == 1
        assert "greet" in result.results

    def test_with_variables(self):
        client = FakeAsyncClient()
        ffai = FFAI(client)
        csv_text = "name,prompt\nstep,Name a {animal}."
        result = _arun(
            ffai.workflow.execute_workflow_csv(csv_text, variables={"animal": "color"})
        )
        assert result.success_count == 1
        call_args = client._mock_generate.call_args
        assert "color" in call_args.kwargs.get("prompt", call_args[1].get("prompt", ""))

    def test_with_defaults(self):
        client = FakeAsyncClient()
        ffai = FFAI(client)
        csv_text = "name,prompt\nstep,What is 3+4?"
        result = _arun(
            ffai.workflow.execute_workflow_csv(csv_text, defaults={"temperature": 0})
        )
        assert result.success_count == 1

    def test_invalid_csv_raises(self):
        client = FakeAsyncClient()
        ffai = FFAI(client)
        with pytest.raises(Exception, match="missing required"):
            _arun(ffai.workflow.execute_workflow_csv("name\nstep"))


class TestExecuteWorkflowCsvFile:
    def test_reads_file(self):
        client = FakeAsyncClient()
        ffai = FFAI(client)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("name,prompt\nstep,What is 5+6?\n")
            f.flush()
            result = _arun(ffai.workflow.execute_workflow_csv_file(f.name))
            assert result.success_count == 1
            assert "step" in result.results

    def test_file_not_found(self):
        client = FakeAsyncClient()
        ffai = FFAI(client)
        with pytest.raises(FileNotFoundError):
            _arun(ffai.workflow.execute_workflow_csv_file("/nonexistent.csv"))
