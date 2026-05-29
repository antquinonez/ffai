# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT

from unittest.mock import MagicMock, patch

import pytest

from ffai.core.response_options import ResponseOptions

_ffai_mod = __import__("importlib").import_module("ffai.FFAI")


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.model = "test-model"
    client.get_conversation_history.return_value = []
    client.set_conversation_history = MagicMock()
    client.clear_conversation = MagicMock()
    client.last_usage = None
    client.last_cost_usd = 0.0
    return client


@pytest.fixture
def ffai(mock_client):
    with patch.object(_ffai_mod, "get_config"):
        from ffai.config import Config
        from ffai.FFAI import FFAI

        mock_config = MagicMock(spec=Config)
        mock_config.paths = MagicMock()
        mock_config.paths.ffai_data = "/tmp/ffai_test_data"
        mock_config.rag = MagicMock()
        mock_config.rag.enabled = False

        with patch.object(_ffai_mod, "get_config", return_value=mock_config):
            return FFAI(mock_client)


class TestConditionalExecution:
    def test_condition_true_executes(self, ffai, mock_client):
        mock_client.generate_response.return_value = "executed"

        ffai.history = [
            {"prompt": "go", "response": "ok", "prompt_name": "fetch", "status": "success"}
        ]

        result = ffai.generate_response(
            "process",
            prompt_name="process",
            options=ResponseOptions(condition='{{fetch.status}} == "success"'),
        )
        assert result.status == "success"
        assert result.response == "executed"
        mock_client.generate_response.assert_called_once()

    def test_condition_false_skips(self, ffai, mock_client):
        ffai.history = [
            {"prompt": "go", "response": "", "prompt_name": "fetch", "status": "failed"}
        ]

        result = ffai.generate_response(
            "process",
            prompt_name="process",
            options=ResponseOptions(condition='{{fetch.status}} == "success"'),
        )
        assert result.status == "skipped"
        assert result.response is None
        mock_client.generate_response.assert_not_called()

    def test_condition_trace_on_skip(self, ffai, mock_client):
        ffai.history = [
            {"prompt": "go", "response": "ok", "prompt_name": "fetch", "status": "failed"}
        ]

        result = ffai.generate_response(
            "process",
            prompt_name="process",
            options=ResponseOptions(condition='{{fetch.status}} == "success"'),
        )
        assert result.condition_trace is not None
        assert "failed" in result.condition_trace

    def test_unknown_prompt_name_returns_failed(self, ffai, mock_client):
        result = ffai.generate_response(
            "process",
            prompt_name="process",
            options=ResponseOptions(condition='{{nonexistent.status}} == "success"'),
        )
        assert result.status == "failed"
        assert result.condition_error is not None

    def test_no_condition_executes_normally(self, ffai, mock_client):
        mock_client.generate_response.return_value = "result"

        result = ffai.generate_response(
            prompt="hello",
            prompt_name="test",
        )
        assert result.status == "success"
        mock_client.generate_response.assert_called_once()


class TestBuildResultsByName:
    def test_builds_from_history(self, ffai):
        ffai.history = [
            {"prompt": "go", "response": "ok", "prompt_name": "step1", "status": "success"},
            {"prompt": "go2", "response": "ok2", "prompt_name": "step2", "status": "success"},
        ]
        results = ffai._build_results_by_name()
        assert "step1" in results
        assert "step2" in results
        assert results["step1"]["status"] == "success"

    def test_preserves_dict_response_name(self, ffai):
        ffai.history = [
            {"prompt": "analyze", "response": {"score": 8}, "prompt_name": "analysis", "status": "success"},
        ]
        results = ffai._build_results_by_name()
        assert "analysis" in results
        assert results["analysis"]["response"] == "{'score': 8}"

    def test_reads_actual_status(self, ffai):
        ffai.history = [
            {"prompt": "go", "response": None, "prompt_name": "step1", "status": "skipped"},
        ]
        results = ffai._build_results_by_name()
        assert results["step1"]["status"] == "skipped"


class TestValidateGraph:
    def test_valid_graph_no_warnings(self, ffai):
        graph, warnings = ffai.validate_graph([
            {"prompt_name": "a", "prompt": "first"},
            {"prompt_name": "b", "prompt": "second", "history": ["a"]},
        ])
        assert len(warnings) == 0
        assert graph.max_level == 1

    def test_cycle_raises(self, ffai):
        with pytest.raises(ValueError, match="Dependency cycle"):
            ffai.validate_graph([
                {"prompt_name": "a", "prompt": "first", "history": ["b"]},
                {"prompt_name": "b", "prompt": "second", "history": ["a"]},
            ])

    def test_undeclared_condition_warns(self, ffai):
        graph, warnings = ffai.validate_graph([
            {"prompt_name": "fetch", "prompt": "get data"},
            {"prompt_name": "process", "prompt": "process data", "condition": '{{fetch.status}} == "success"'},
        ])
        assert len(warnings) == 1
        assert "Undeclared dependency" in warnings[0]

    def test_declared_condition_no_warning(self, ffai):
        graph, warnings = ffai.validate_graph([
            {"prompt_name": "fetch", "prompt": "get data"},
            {"prompt_name": "process", "prompt": "process data", "history": ["fetch"], "condition": '{{fetch.status}} == "success"'},
        ])
        assert len(warnings) == 0


class TestStrictMode:
    def test_strict_raises_on_unknown_reference(self, ffai, mock_client):
        with pytest.raises(ValueError, match="Unknown prompt reference"):
            ffai.generate_response(
                "Based on {{nonexistent.response}}, elaborate",
                prompt_name="test",
                options=ResponseOptions(strict=True),
            )

    def test_non_strict_silently_replaces(self, ffai, mock_client):
        mock_client.generate_response.return_value = "result"

        result = ffai.generate_response(
            prompt="Based on {{nonexistent.response}}, elaborate",
            prompt_name="test",
        )
        assert result.status == "success"
        call_args = mock_client.generate_response.call_args
        resolved = call_args.kwargs.get("prompt") or call_args[1].get("prompt") or call_args[0][0]
        assert "{{nonexistent.response}}" not in resolved


class TestResponseResultExtended:
    def test_default_status(self):
        from ffai.core.response_result import ResponseResult

        result = ResponseResult(response="ok")
        assert result.status == "success"
        assert result.condition_trace is None
        assert result.condition_error is None
        assert result.parsed is None
        assert result.parsing_errors is None

    def test_skipped_status(self):
        from ffai.core.response_result import ResponseResult

        result = ResponseResult(response=None, status="skipped", condition_trace='"failed" == "success"')
        assert result.status == "skipped"
        assert result.condition_trace is not None
