from __future__ import annotations

import tempfile

import pytest

from ffai.workflow.tabular import TabularLoadError
from ffai.workflow.tabular_csv import (
    _parse_comment_metadata,
    load_workflow_csv,
    load_workflow_csv_file,
)


class TestParseCommentMetadata:
    def test_no_comments(self):
        lines = ["name,prompt", "a,Hello"]
        result = _parse_comment_metadata(lines)
        assert result.meta == {}
        assert result.remaining_lines == lines

    def test_workflow_name(self):
        lines = ["# workflow: my_pipeline", "name,prompt"]
        result = _parse_comment_metadata(lines)
        assert result.meta["name"] == "my_pipeline"

    def test_description(self):
        lines = ["# description: A test", "name,prompt"]
        result = _parse_comment_metadata(lines)
        assert result.meta["description"] == "A test"

    def test_default_temperature(self):
        lines = ["# default_temperature: 0.5", "name,prompt"]
        result = _parse_comment_metadata(lines)
        assert result.defaults["temperature"] == 0.5

    def test_default_max_tokens(self):
        lines = ["# default_max_tokens: 200", "name,prompt"]
        result = _parse_comment_metadata(lines)
        assert result.defaults["max_tokens"] == 200

    def test_client_definition(self):
        lines = [
            '# client.reviewer: {"type":"litellm","model":"gpt-4o"}',
            "name,prompt",
        ]
        result = _parse_comment_metadata(lines)
        assert result.clients["reviewer"] == {"type": "litellm", "model": "gpt-4o"}

    def test_mixed_comments_and_data(self):
        lines = [
            "# workflow: test",
            "# default_temperature: 0",
            "name,prompt",
            "a,Hello",
        ]
        result = _parse_comment_metadata(lines)
        assert result.meta["name"] == "test"
        assert result.defaults["temperature"] == 0.0
        assert result.remaining_lines == ["name,prompt", "a,Hello"]


class TestLoadWorkflowCsv:
    def test_minimal_csv(self):
        csv_text = "name,prompt\ntopic,Name a discovery."
        spec = load_workflow_csv(csv_text)
        assert len(spec.prompts) == 1
        assert spec.prompts[0].name == "topic"

    def test_multistep_csv(self):
        csv_text = (
            "name,prompt,history\n"
            'topic,"Name a discovery.",\n'
            "explain,Explain {{{{topic.response}}}},topic"
        )
        spec = load_workflow_csv(csv_text)
        assert len(spec.prompts) == 2
        assert spec.prompts[1].history == ["topic"]

    def test_csv_with_quoted_prompts(self):
        csv_text = (
            'name,prompt\n'
            'greet,"What is 2+2? Answer with just the number."'
        )
        spec = load_workflow_csv(csv_text)
        assert "2+2" in spec.prompts[0].prompt

    def test_csv_with_comment_metadata(self):
        csv_text = (
            "# workflow: research\n"
            "# default_temperature: 0\n"
            "name,prompt\n"
            "topic,Name a discovery."
        )
        spec = load_workflow_csv(csv_text)
        assert spec.name == "research"
        assert spec.defaults.temperature == 0.0

    def test_function_params_override_comments(self):
        csv_text = (
            "# workflow: comment_name\n"
            "# default_temperature: 0\n"
            "name,prompt\n"
            "topic,Go"
        )
        spec = load_workflow_csv(
            csv_text, name="func_name", defaults={"temperature": 0.7}
        )
        assert spec.name == "func_name"
        assert spec.defaults.temperature == 0.7

    def test_function_defaults_merge_with_comments(self):
        csv_text = (
            "# default_temperature: 0\n"
            "name,prompt\n"
            "topic,Go"
        )
        spec = load_workflow_csv(csv_text, defaults={"max_tokens": 200})
        assert spec.defaults.temperature == 0.0
        assert spec.defaults.max_tokens == 200

    def test_tsv_delimiter(self):
        tsv_text = "name\tprompt\nstep\tHello"
        spec = load_workflow_csv(tsv_text, delimiter="\t")
        assert len(spec.prompts) == 1
        assert spec.prompts[0].name == "step"

    def test_empty_csv_raises(self):
        with pytest.raises(TabularLoadError, match="no data rows"):
            load_workflow_csv("name,prompt\n")

    def test_csv_with_clients_param(self):
        csv_text = (
            "name,prompt,client\n"
            "step,Go,reviewer"
        )
        spec = load_workflow_csv(
            csv_text,
            clients={"reviewer": {"type": "litellm", "model": "gpt-4o"}},
        )
        assert "reviewer" in spec.clients
        assert spec.prompts[0].client is not None
        assert spec.prompts[0].client.name == "reviewer"

    def test_header_normalization_in_csv(self):
        csv_text = (
            "Step Name,Question\n"
            "a,Hello"
        )
        spec = load_workflow_csv(csv_text)
        assert spec.prompts[0].name == "a"
        assert spec.prompts[0].prompt == "Hello"


class TestLoadWorkflowCsvFile:
    def test_reads_file(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("name,prompt\nstep,Hello\n")
            f.flush()
            spec = load_workflow_csv_file(f.name)
            assert spec.prompts[0].name == "step"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            load_workflow_csv_file("/nonexistent/path.csv")

    def test_file_with_metadata(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write("# workflow: file_test\n")
            f.write("# description: From file\n")
            f.write("name,prompt\n")
            f.write("step,Go\n")
            f.flush()
            spec = load_workflow_csv_file(f.name)
            assert spec.name == "file_test"
            assert spec.description == "From file"
