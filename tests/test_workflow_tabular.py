from __future__ import annotations

import pytest

from ffai.workflow.tabular import (
    TabularLoadError,
    _canonical_column,
    _coerce_value,
    _normalize_header,
    _parse_client_ref,
    load_workflow_rows,
)


class TestNormalizeHeader:
    def test_lowercase(self):
        assert _normalize_header("Name") == "name"

    def test_spaces_to_underscores(self):
        assert _normalize_header("max tokens") == "max_tokens"

    def test_hyphens_to_underscores(self):
        assert _normalize_header("step-name") == "step_name"

    def test_leading_trailing_stripped(self):
        assert _normalize_header("  name  ") == "name"

    def test_mixed(self):
        assert _normalize_header("  Step Name ") == "step_name"


class TestCanonicalColumn:
    def test_alias_step(self):
        assert _canonical_column("step") == "name"

    def test_alias_prompt_text(self):
        assert _canonical_column("prompt_text") == "prompt"

    def test_alias_depends_on(self):
        assert _canonical_column("depends_on") == "history"

    def test_alias_max_tokens(self):
        assert _canonical_column("max tokens") == "max_tokens"

    def test_passthrough_unknown(self):
        assert _canonical_column("custom_field") == "custom_field"

    def test_case_insensitive(self):
        assert _canonical_column("Step Name") == "name"


class TestCoerceValue:
    def test_temperature_float(self):
        assert _coerce_value("temperature", "0.7") == 0.7

    def test_temperature_int_string(self):
        assert _coerce_value("temperature", "0") == 0.0

    def test_max_tokens_int(self):
        assert _coerce_value("max_tokens", "100") == 100

    def test_strict_true_values(self):
        for v in ("true", "True", "yes", "YES", "1"):
            assert _coerce_value("strict", v) is True

    def test_strict_false_values(self):
        for v in ("false", "no", "0", "maybe"):
            assert _coerce_value("strict", v) is False

    def test_history_comma_separated(self):
        result = _coerce_value("history", "topic,explain")
        assert result == ["topic", "explain"]

    def test_history_single(self):
        result = _coerce_value("history", "topic")
        assert result == ["topic"]

    def test_history_empty(self):
        assert _coerce_value("history", "") is None

    def test_history_none(self):
        assert _coerce_value("history", None) is None

    def test_tools_comma_separated(self):
        result = _coerce_value("tools", "search,lookup")
        assert result == ["search", "lookup"]

    def test_response_format_json(self):
        result = _coerce_value("response_format", '{"type":"json_object"}')
        assert result == {"type": "json_object"}

    def test_response_format_string(self):
        assert _coerce_value("response_format", "text") == "text"

    def test_response_format_empty(self):
        assert _coerce_value("response_format", "") is None

    def test_blank_string_becomes_none(self):
        assert _coerce_value("model", "") is None

    def test_blank_string_with_spaces(self):
        assert _coerce_value("condition", "   ") is None

    def test_none_passthrough(self):
        assert _coerce_value("model", None) is None

    def test_non_blank_string_kept(self):
        assert _coerce_value("model", "gpt-4o") == "gpt-4o"


class TestParseClientRef:
    def test_none(self):
        assert _parse_client_ref(None) is None

    def test_empty_string(self):
        assert _parse_client_ref("") is None

    def test_string_becomes_named_ref(self):
        ref = _parse_client_ref("openai_reviewer")
        assert ref is not None
        assert ref.name == "openai_reviewer"
        assert ref.is_named_ref is True

    def test_dict(self):
        ref = _parse_client_ref({"type": "litellm", "model": "gpt-4o"})
        assert ref is not None
        assert ref.type == "litellm"
        assert ref.model == "gpt-4o"


class TestLoadWorkflowRowsBasic:
    def test_minimal_single_step(self):
        spec = load_workflow_rows(
            [{"name": "greet", "prompt": "Hello"}],
            name="test",
        )
        assert spec.name == "test"
        assert len(spec.prompts) == 1
        assert spec.prompts[0].name == "greet"
        assert spec.prompts[0].prompt == "Hello"

    def test_multiple_steps(self):
        spec = load_workflow_rows([
            {"name": "a", "prompt": "Step A"},
            {"name": "b", "prompt": "Step B", "history": "a"},
        ])
        assert len(spec.prompts) == 2
        assert spec.prompts[1].history == ["a"]

    def test_with_clients(self):
        spec = load_workflow_rows(
            [{"name": "step", "prompt": "Go", "client": "reviewer"}],
            clients={"reviewer": {"type": "litellm", "model": "gpt-4o"}},
        )
        assert "reviewer" in spec.clients
        assert spec.prompts[0].client is not None
        assert spec.prompts[0].client.name == "reviewer"

    def test_with_defaults(self):
        spec = load_workflow_rows(
            [{"name": "step", "prompt": "Go"}],
            defaults={"temperature": 0.5, "max_tokens": 200},
        )
        assert spec.defaults.temperature == 0.5
        assert spec.defaults.max_tokens == 200

    def test_with_tools(self):
        spec = load_workflow_rows(
            [{"name": "step", "prompt": "Go", "tools": "search"}],
            tools={"search": {"description": "Search tool", "parameters": {}}},
        )
        assert "search" in spec.tools
        assert spec.tools["search"]["name"] == "search"

    def test_name_default(self):
        spec = load_workflow_rows([{"name": "step", "prompt": "Go"}])
        assert spec.name == "unnamed"

    def test_description(self):
        spec = load_workflow_rows(
            [{"name": "step", "prompt": "Go"}],
            description="A test workflow",
        )
        assert spec.description == "A test workflow"


class TestLoadWorkflowRowsHeaderNormalization:
    def test_alias_headers(self):
        spec = load_workflow_rows([
            {"Step": "a", "Question": "Hello"},
            {"step_name": "b", "depends_on": "a", "prompt_text": "World"},
        ])
        assert spec.prompts[0].name == "a"
        assert spec.prompts[0].prompt == "Hello"
        assert spec.prompts[1].name == "b"
        assert spec.prompts[1].prompt == "World"
        assert spec.prompts[1].history == ["a"]

    def test_max_tokens_header_variants(self):
        spec = load_workflow_rows([
            {"name": "a", "prompt": "Go", "max tokens": "100"},
        ])
        assert spec.prompts[0].max_tokens == 100


class TestLoadWorkflowRowsErrors:
    def test_empty_rows(self):
        with pytest.raises(TabularLoadError, match="No rows provided"):
            load_workflow_rows([])

    def test_missing_name(self):
        with pytest.raises(TabularLoadError, match="missing required field 'name'"):
            load_workflow_rows([{"prompt": "Hello"}])

    def test_missing_prompt(self):
        with pytest.raises(TabularLoadError, match="missing required field 'prompt'"):
            load_workflow_rows([{"name": "a"}])

    def test_duplicate_names(self):
        with pytest.raises(TabularLoadError, match="Duplicate"):
            load_workflow_rows([
                {"name": "a", "prompt": "One"},
                {"name": "a", "prompt": "Two"},
            ])

    def test_bad_history_reference(self):
        with pytest.raises(TabularLoadError, match="unknown prompt"):
            load_workflow_rows([
                {"name": "a", "prompt": "Go", "history": "nonexistent"},
            ])

    def test_self_reference(self):
        with pytest.raises(TabularLoadError, match="references itself"):
            load_workflow_rows([
                {"name": "a", "prompt": "Go", "history": "a"},
            ])


class TestLoadWorkflowRowsTypeCoercion:
    def test_temperature_coerced(self):
        spec = load_workflow_rows([
            {"name": "a", "prompt": "Go", "temperature": "0.7"},
        ])
        assert spec.prompts[0].temperature == 0.7

    def test_max_tokens_coerced(self):
        spec = load_workflow_rows([
            {"name": "a", "prompt": "Go", "max_tokens": "100"},
        ])
        assert spec.prompts[0].max_tokens == 100

    def test_strict_coerced(self):
        spec = load_workflow_rows([
            {"name": "a", "prompt": "Go", "strict": "true"},
        ])
        assert spec.prompts[0].strict is True

    def test_history_comma_separated(self):
        spec = load_workflow_rows([
            {"name": "a", "prompt": "Go"},
            {"name": "b", "prompt": "Go", "history": "a"},
            {"name": "c", "prompt": "Go", "history": "a,b"},
        ])
        assert spec.prompts[2].history == ["a", "b"]

    def test_response_format_json(self):
        spec = load_workflow_rows([
            {"name": "a", "prompt": "Go", "response_format": '{"type":"json_object"}'},
        ])
        assert spec.prompts[0].response_format == {"type": "json_object"}

    def test_blank_optional_fields_are_none(self):
        spec = load_workflow_rows([
            {"name": "a", "prompt": "Go", "model": "", "condition": "   "},
        ])
        assert spec.prompts[0].model is None
        assert spec.prompts[0].condition is None


class TestLoadWorkflowRowsEquivalence:
    def test_produces_same_spec_as_yaml(self):
        from ffai.workflow import load_workflow

        yaml_text = """
workflow:
  name: research
  description: A pipeline
  defaults:
    temperature: 0
    max_tokens: 100
  clients:
    reviewer:
      type: litellm
      model: gpt-4o
      api_key_env: OPENAI_API_KEY
  prompts:
    - name: topic
      prompt: "Name a discovery."
    - name: explain
      prompt: "Explain {{topic.response}}."
      history: [topic]
      client: reviewer
    - name: opinion
      prompt: "Why does {{topic.response}} matter?"
      history: [topic]
"""

        yaml_spec = load_workflow(yaml_text)
        row_spec = load_workflow_rows(
            [
                {"name": "topic", "prompt": "Name a discovery."},
                {
                    "name": "explain",
                    "prompt": "Explain {{topic.response}}.",
                    "history": "topic",
                    "client": "reviewer",
                },
                {
                    "name": "opinion",
                    "prompt": "Why does {{topic.response}} matter?",
                    "history": "topic",
                },
            ],
            name="research",
            description="A pipeline",
            defaults={"temperature": 0, "max_tokens": 100},
            clients={
                "reviewer": {
                    "type": "litellm",
                    "model": "gpt-4o",
                    "api_key_env": "OPENAI_API_KEY",
                },
            },
        )

        assert yaml_spec.name == row_spec.name
        assert yaml_spec.description == row_spec.description
        assert len(yaml_spec.prompts) == len(row_spec.prompts)

        for yp, rp in zip(yaml_spec.prompts, row_spec.prompts):
            assert yp.name == rp.name
            assert yp.prompt == rp.prompt
            assert yp.history == rp.history

        assert yaml_spec.defaults.temperature == row_spec.defaults.temperature
        assert yaml_spec.defaults.max_tokens == row_spec.defaults.max_tokens
        assert set(yaml_spec.clients.keys()) == set(row_spec.clients.keys())
