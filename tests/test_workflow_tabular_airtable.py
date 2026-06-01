from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from ffai.workflow.tabular import TabularLoadError
from ffai.workflow.tabular_airtable import (
    _get_api_key,
    _records_to_rows,
    load_workflow_airtable,
)


class TestGetApiKey:
    def test_explicit_key(self):
        assert _get_api_key("my_key") == "my_key"

    def test_from_env(self):
        with patch.dict(os.environ, {"AIRTABLE_API_KEY": "env_key"}):
            assert _get_api_key(None) == "env_key"

    def test_custom_env_var(self):
        with patch.dict(os.environ, {"MY_KEY": "custom_key"}):
            assert _get_api_key(None, env_var="MY_KEY") == "custom_key"

    def test_missing_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(TabularLoadError, match="AIRTABLE_API_KEY"):
                _get_api_key(None)

    def test_explicit_overrides_env(self):
        with patch.dict(os.environ, {"AIRTABLE_API_KEY": "env_key"}):
            assert _get_api_key("explicit") == "explicit"


class TestRecordsToRows:
    def test_basic_records(self):
        records = [
            {"id": "rec1", "fields": {"name": "a", "prompt": "Hello"}},
            {"id": "rec2", "fields": {"name": "b", "prompt": "World"}},
        ]
        rows = _records_to_rows(records)
        assert len(rows) == 2
        assert rows[0] == {"name": "a", "prompt": "Hello"}
        assert rows[1] == {"name": "b", "prompt": "World"}

    def test_empty_fields_skipped(self):
        records = [
            {"id": "rec1", "fields": {"name": "a", "prompt": "Hello"}},
            {"id": "rec2", "fields": {}},
            {"id": "rec3", "fields": {"name": "c", "prompt": "Go"}},
        ]
        rows = _records_to_rows(records)
        assert len(rows) == 2
        assert rows[1]["name"] == "c"

    def test_missing_fields_key(self):
        records = [{"id": "rec1"}, {"id": "rec2", "fields": {"name": "a", "prompt": "Go"}}]
        rows = _records_to_rows(records)
        assert len(rows) == 1
        assert rows[0]["name"] == "a"


class TestLoadWorkflowAirtable:
    @patch("pyairtable.api.Api")
    def test_basic_load(self, mock_api_cls):
        mock_table = MagicMock()
        mock_table.all.return_value = [
            {"id": "rec1", "fields": {"name": "topic", "prompt": "Name a discovery."}},
            {
                "id": "rec2",
                "fields": {
                    "name": "explain",
                    "prompt": "Explain {{topic.response}}.",
                    "history": "topic",
                },
            },
        ]
        mock_api_cls.return_value.table.return_value = mock_table

        spec = load_workflow_airtable(
            "appTestBase",
            "Workflow Steps",
            api_key="test_key",
            name="research",
        )

        mock_api_cls.assert_called_once_with("test_key")
        mock_api_cls.return_value.table.assert_called_once_with(
            "appTestBase", "Workflow Steps"
        )
        assert spec.name == "research"
        assert len(spec.prompts) == 2
        assert spec.prompts[0].name == "topic"
        assert spec.prompts[1].history == ["topic"]

    @patch("pyairtable.api.Api")
    def test_with_view(self, mock_api_cls):
        mock_table = MagicMock()
        mock_table.all.return_value = [
            {"id": "rec1", "fields": {"name": "a", "prompt": "Go"}},
        ]
        mock_api_cls.return_value.table.return_value = mock_table

        load_workflow_airtable(
            "appBase", "Steps", api_key="key", view="Active steps"
        )

        mock_table.all.assert_called_once_with(view="Active steps")

    @patch("pyairtable.api.Api")
    def test_without_view(self, mock_api_cls):
        mock_table = MagicMock()
        mock_table.all.return_value = [
            {"id": "rec1", "fields": {"name": "a", "prompt": "Go"}},
        ]
        mock_api_cls.return_value.table.return_value = mock_table

        load_workflow_airtable("appBase", "Steps", api_key="key")
        mock_table.all.assert_called_once_with()

    @patch("pyairtable.api.Api")
    def test_with_clients(self, mock_api_cls):
        mock_table = MagicMock()
        mock_table.all.return_value = [
            {"id": "rec1", "fields": {"name": "a", "prompt": "Go", "client": "reviewer"}},
        ]
        mock_api_cls.return_value.table.return_value = mock_table

        spec = load_workflow_airtable(
            "appBase",
            "Steps",
            api_key="key",
            clients={"reviewer": {"type": "litellm", "model": "gpt-4o"}},
        )
        assert "reviewer" in spec.clients
        assert spec.prompts[0].client is not None
        assert spec.prompts[0].client.name == "reviewer"

    @patch("pyairtable.api.Api")
    def test_with_defaults(self, mock_api_cls):
        mock_table = MagicMock()
        mock_table.all.return_value = [
            {"id": "rec1", "fields": {"name": "a", "prompt": "Go"}},
        ]
        mock_api_cls.return_value.table.return_value = mock_table

        spec = load_workflow_airtable(
            "appBase",
            "Steps",
            api_key="key",
            defaults={"temperature": 0.5, "max_tokens": 200},
        )
        assert spec.defaults.temperature == 0.5
        assert spec.defaults.max_tokens == 200

    @patch("pyairtable.api.Api")
    def test_empty_table_raises(self, mock_api_cls):
        mock_table = MagicMock()
        mock_table.all.return_value = []
        mock_api_cls.return_value.table.return_value = mock_table

        with pytest.raises(TabularLoadError, match="contains no records"):
            load_workflow_airtable("appBase", "Empty", api_key="key")

    @patch("pyairtable.api.Api")
    def test_description(self, mock_api_cls):
        mock_table = MagicMock()
        mock_table.all.return_value = [
            {"id": "rec1", "fields": {"name": "a", "prompt": "Go"}},
        ]
        mock_api_cls.return_value.table.return_value = mock_table

        spec = load_workflow_airtable(
            "appBase", "Steps", api_key="key", description="From Airtable"
        )
        assert spec.description == "From Airtable"

    def test_missing_api_key_raises(self):
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(TabularLoadError, match="AIRTABLE_API_KEY"):
                load_workflow_airtable("appBase", "Steps")

    def test_missing_pyairtable_raises(self):
        with patch.dict("sys.modules", {"pyairtable.api": None, "pyairtable": None}):
            with pytest.raises(TabularLoadError, match="pyairtable is required"):
                load_workflow_airtable(
                    "appBase", "Steps", api_key="key"
                )
