from __future__ import annotations

import tempfile

import pytest

from ffai.Clients.AsyncFFLiteLLMClient import AsyncFFLiteLLMClient
from ffai.FFAI import FFAI


def _make_ffai():
    client = AsyncFFLiteLLMClient(
        model_string="mistral/mistral-small-2503",
        temperature=0,
        max_tokens=50,
        system_instructions="Be concise. One word.",
    )
    return FFAI(client)


def _arun(coro):
    import asyncio

    return asyncio.run(coro)


class TestExecuteWorkflowCsv:
    def test_single_step(self):
        ffai = _make_ffai()
        csv_text = "name,prompt\ngreet,What is 2+2? Answer with just the number."
        result = _arun(ffai.execute_workflow_csv(csv_text))
        assert result.success_count == 1
        assert result.results["greet"].response.strip() == "4"

    def test_with_variables(self):
        ffai = _make_ffai()
        csv_text = "name,prompt\nstep,Name a {animal}."
        result = _arun(
            ffai.execute_workflow_csv(csv_text, variables={"animal": "color"})
        )
        assert result.success_count == 1

    def test_with_defaults(self):
        ffai = _make_ffai()
        csv_text = "name,prompt\nstep,What is 3+4? Just the number."
        result = _arun(
            ffai.execute_workflow_csv(csv_text, defaults={"temperature": 0})
        )
        assert result.success_count == 1

    def test_invalid_csv_raises(self):
        ffai = _make_ffai()
        with pytest.raises(Exception, match="missing required"):
            _arun(ffai.execute_workflow_csv("name\nstep"))


class TestExecuteWorkflowCsvFile:
    def test_reads_file(self):
        ffai = _make_ffai()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("name,prompt\nstep,What is 5+6? Just the number.\n")
            f.flush()
            result = _arun(ffai.execute_workflow_csv_file(f.name))
            assert result.success_count == 1
            assert result.results["step"].response.strip() == "11"

    def test_file_not_found(self):
        ffai = _make_ffai()
        with pytest.raises(FileNotFoundError):
            _arun(ffai.execute_workflow_csv_file("/nonexistent.csv"))
