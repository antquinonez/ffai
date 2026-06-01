import pytest

from ffai.workflow import (
    ClientRef,
    PromptStep,
    WorkflowDefaults,
    WorkflowSpec,
    WorkflowValidationError,
    load_workflow,
    load_workflow_file,
)


class TestClientRef:
    def test_from_string_creates_named_ref(self):
        ref = ClientRef.from_dict("litellm-mistral-small")
        assert ref.name == "litellm-mistral-small"
        assert ref.type is None
        assert ref.model is None
        assert ref.is_named_ref is True

    def test_from_dict_creates_inline_ref(self):
        ref = ClientRef.from_dict({
            "type": "litellm",
            "provider_prefix": "anthropic/",
            "model": "claude-3-5-sonnet-20241022",
            "api_key_env": "ANTHROPIC_API_KEY",
            "fallbacks": ["openai/gpt-4o"],
        })
        assert ref.name is None
        assert ref.type == "litellm"
        assert ref.provider_prefix == "anthropic/"
        assert ref.model == "claude-3-5-sonnet-20241022"
        assert ref.api_key_env == "ANTHROPIC_API_KEY"
        assert ref.fallbacks == ["openai/gpt-4o"]
        assert ref.is_named_ref is False

    def test_to_dict_round_trip_named(self):
        original = ClientRef.from_dict("my-client")
        restored = ClientRef.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.is_named_ref == original.is_named_ref

    def test_to_dict_round_trip_inline(self):
        original = ClientRef.from_dict({
            "type": "litellm",
            "model": "gpt-4o",
            "fallbacks": ["mistral/mistral-small"],
        })
        restored = ClientRef.from_dict(original.to_dict())
        assert restored.type == "litellm"
        assert restored.model == "gpt-4o"
        assert restored.fallbacks == ["mistral/mistral-small"]

    def test_to_dict_omits_empty_fields(self):
        ref = ClientRef(name="x")
        d = ref.to_dict()
        assert d == {"name": "x"}

    def test_is_named_ref_false_when_type_set(self):
        ref = ClientRef(name="x", type="litellm")
        assert ref.is_named_ref is False

    def test_is_named_ref_false_when_model_set(self):
        ref = ClientRef(name="x", model="gpt-4o")
        assert ref.is_named_ref is False


class TestPromptStep:
    def test_from_dict_minimal(self):
        step = PromptStep.from_dict({"name": "greet", "prompt": "Say hello"})
        assert step.name == "greet"
        assert step.prompt == "Say hello"
        assert step.client is None
        assert step.model is None
        assert step.history is None
        assert step.condition is None
        assert step.strict is False

    def test_from_dict_full(self):
        step = PromptStep.from_dict({
            "name": "analyze",
            "prompt": "Analyze {{research.response}}",
            "client": "researcher",
            "model": "mistral/mistral-large-latest",
            "history": ["research"],
            "condition": '{{research.status}} == "success"',
            "abort_condition": '{{research.status}} == "failed"',
            "system_instructions": "Be thorough",
            "response_format": "json_object",
            "response_model": "mypackage.models.Analysis",
            "strict": True,
            "tools": ["search", "calculate"],
            "tool_choice": "auto",
            "max_tokens": 8000,
            "temperature": 0.3,
        })
        assert step.name == "analyze"
        assert step.client is not None
        assert step.client.name == "researcher"
        assert step.model == "mistral/mistral-large-latest"
        assert step.history == ["research"]
        assert step.condition == '{{research.status}} == "success"'
        assert step.abort_condition == '{{research.status}} == "failed"'
        assert step.system_instructions == "Be thorough"
        assert step.response_format == "json_object"
        assert step.response_model == "mypackage.models.Analysis"
        assert step.strict is True
        assert step.tools == ["search", "calculate"]
        assert step.tool_choice == "auto"
        assert step.max_tokens == 8000
        assert step.temperature == 0.3

    def test_to_dict_omits_defaults(self):
        step = PromptStep(name="x", prompt="hi")
        d = step.to_dict()
        assert d == {"name": "x", "prompt": "hi"}

    def test_to_dict_includes_strict_true(self):
        step = PromptStep(name="x", prompt="hi", strict=True)
        d = step.to_dict()
        assert d["strict"] is True

    def test_to_dict_omits_strict_false(self):
        step = PromptStep(name="x", prompt="hi", strict=False)
        d = step.to_dict()
        assert "strict" not in d

    def test_round_trip(self):
        original = PromptStep.from_dict({
            "name": "step1",
            "prompt": "Hello",
            "client": "my-client",
            "history": ["prev"],
            "condition": "x == 1",
            "strict": True,
        })
        restored = PromptStep.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.prompt == original.prompt
        assert restored.client is not None
        assert original.client is not None
        assert restored.client.name == original.client.name
        assert restored.history == original.history
        assert restored.condition == original.condition
        assert restored.strict == original.strict

    def test_from_dict_client_as_inline_dict(self):
        step = PromptStep.from_dict({
            "name": "s1",
            "prompt": "p",
            "client": {"type": "litellm", "model": "gpt-4o"},
        })
        assert step.client is not None
        assert step.client.type == "litellm"
        assert step.client.model == "gpt-4o"
        assert step.client.is_named_ref is False


class TestWorkflowDefaults:
    def test_from_dict_empty(self):
        defaults = WorkflowDefaults.from_dict({})
        assert defaults.client is None
        assert defaults.max_concurrency == 10
        assert defaults.strict is False

    def test_from_dict_with_client(self):
        defaults = WorkflowDefaults.from_dict({"client": "my-client"})
        assert defaults.client is not None
        assert defaults.client.name == "my-client"

    def test_from_dict_with_overrides(self):
        defaults = WorkflowDefaults.from_dict({
            "max_concurrency": 3,
            "strict": True,
            "system_instructions": "Be helpful",
            "max_tokens": 2048,
            "temperature": 0.5,
        })
        assert defaults.max_concurrency == 3
        assert defaults.strict is True
        assert defaults.system_instructions == "Be helpful"
        assert defaults.max_tokens == 2048
        assert defaults.temperature == 0.5


class TestWorkflowSpec:
    def _minimal_yaml_dict(self):
        return {
            "workflow": {
                "name": "test",
                "prompts": [
                    {"name": "step1", "prompt": "Hello"},
                ],
            },
        }

    def test_from_dict_minimal(self):
        spec = WorkflowSpec.from_dict(self._minimal_yaml_dict())
        assert spec.name == "test"
        assert len(spec.prompts) == 1
        assert spec.prompts[0].name == "step1"

    def test_from_dict_full(self):
        data = {
            "workflow": {
                "name": "research",
                "description": "A research pipeline",
                "defaults": {
                    "client": "litellm-mistral-small",
                    "max_concurrency": 3,
                },
                "clients": {
                    "researcher": {
                        "type": "litellm",
                        "model": "gpt-4o",
                        "api_key_env": "OPENAI_API_KEY",
                    },
                },
                "tools": {
                    "search": {
                        "description": "Search",
                        "parameters": {"type": "object"},
                    },
                },
                "prompts": [
                    {"name": "fetch", "prompt": "Research {topic}"},
                    {
                        "name": "analyze",
                        "prompt": "Analyze {{fetch.response}}",
                        "client": "researcher",
                        "history": ["fetch"],
                        "tools": ["search"],
                    },
                ],
            },
        }
        spec = WorkflowSpec.from_dict(data)
        assert spec.name == "research"
        assert spec.description == "A research pipeline"
        assert spec.defaults.client is not None
        assert spec.defaults.client.name == "litellm-mistral-small"
        assert spec.defaults.max_concurrency == 3
        assert "researcher" in spec.clients
        assert spec.clients["researcher"].model == "gpt-4o"
        assert "search" in spec.tools
        assert spec.tools["search"]["name"] == "search"
        assert len(spec.prompts) == 2
        assert spec.prompts[1].client is not None
        assert spec.prompts[1].client.name == "researcher"
        assert spec.prompts[1].tools == ["search"]

    def test_to_dict_round_trip(self):
        data = self._minimal_yaml_dict()
        spec = WorkflowSpec.from_dict(data)
        restored = WorkflowSpec.from_dict(spec.to_dict())
        assert restored.name == spec.name
        assert len(restored.prompts) == len(spec.prompts)
        assert restored.prompts[0].name == spec.prompts[0].name

    def test_get_client_names(self):
        spec = WorkflowSpec.from_dict({
            "workflow": {
                "name": "t",
                "defaults": {"client": "default-client"},
                "prompts": [
                    {"name": "s1", "prompt": "p", "client": "client-a"},
                    {"name": "s2", "prompt": "p"},
                    {"name": "s3", "prompt": "p", "client": "client-a"},
                ],
            },
        })
        assert spec.get_client_names() == {"default-client", "client-a"}

    def test_get_tool_names(self):
        spec = WorkflowSpec.from_dict({
            "workflow": {
                "name": "t",
                "tools": {
                    "search": {"description": "Search"},
                    "calc": {"description": "Calc"},
                },
                "prompts": [
                    {"name": "s1", "prompt": "p", "tools": ["search", "calc"]},
                    {"name": "s2", "prompt": "p", "tools": ["search"]},
                ],
            },
        })
        assert spec.get_tool_names() == {"search", "calc"}

    def test_from_dict_wraps_unwrapped(self):
        spec = WorkflowSpec.from_dict({
            "name": "unwrapped",
            "prompts": [{"name": "s1", "prompt": "p"}],
        })
        assert spec.name == "unwrapped"


class TestLoadWorkflow:
    def test_minimal_valid(self):
        spec = load_workflow("""
workflow:
  name: simple
  prompts:
    - name: greet
      prompt: "Say hello"
""")
        assert spec.name == "simple"
        assert len(spec.prompts) == 1
        assert spec.prompts[0].name == "greet"

    def test_missing_prompts_key(self):
        with pytest.raises(WorkflowValidationError, match="prompts"):
            load_workflow("workflow:\n  name: no_prompts")

    def test_empty_prompts(self):
        with pytest.raises(WorkflowValidationError, match="at least one prompt"):
            load_workflow("workflow:\n  prompts: []")

    def test_duplicate_prompt_names(self):
        with pytest.raises(WorkflowValidationError, match="Duplicate"):
            load_workflow("""
workflow:
  prompts:
    - name: dup
      prompt: "first"
    - name: dup
      prompt: "second"
""")

    def test_unknown_history_reference(self):
        with pytest.raises(WorkflowValidationError, match="unknown prompt"):
            load_workflow("""
workflow:
  prompts:
    - name: s1
      prompt: "p"
      history: ["nonexistent"]
""")

    def test_self_referencing_history(self):
        with pytest.raises(WorkflowValidationError, match="itself"):
            load_workflow("""
workflow:
  prompts:
    - name: s1
      prompt: "p"
      history: ["s1"]
""")

    def test_undefined_tool_reference(self):
        with pytest.raises(WorkflowValidationError, match="undefined tool"):
            load_workflow("""
workflow:
  prompts:
    - name: s1
      prompt: "p"
      tools: ["missing_tool"]
""")

    def test_valid_tool_reference_passes(self):
        spec = load_workflow("""
workflow:
  tools:
    search:
      description: "Search"
  prompts:
    - name: s1
      prompt: "p"
      tools: ["search"]
""")
        assert spec.prompts[0].tools == ["search"]

    def test_non_mapping_yaml(self):
        with pytest.raises(WorkflowValidationError, match="mapping"):
            load_workflow("- just\n- a\n- list")

    def test_multiple_errors_reported(self):
        with pytest.raises(WorkflowValidationError, match="2 error"):
            load_workflow("""
workflow:
  prompts:
    - name: dup
      prompt: "first"
    - name: dup
      prompt: "second"
    - name: s3
      prompt: "p"
      history: ["missing"]
""")


class TestLoadWorkflowFile:
    def test_loads_valid_file(self, tmp_path):
        f = tmp_path / "test.yaml"
        f.write_text("""
workflow:
  name: from_file
  prompts:
    - name: step1
      prompt: "Hello"
""")
        spec = load_workflow_file(f)
        assert spec.name == "from_file"

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_workflow_file("/nonexistent/path.yaml")

    def test_invalid_file_raises_validation_error(self, tmp_path):
        f = tmp_path / "bad.yaml"
        f.write_text("workflow:\n  name: bad\n  prompts: []")
        with pytest.raises(WorkflowValidationError):
            load_workflow_file(f)
