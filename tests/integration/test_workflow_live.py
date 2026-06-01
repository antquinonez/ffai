import os

import pytest

from ffai.Clients.AsyncFFLiteLLMClient import AsyncFFLiteLLMClient
from ffai.FFAI import FFAI

pytestmark = [pytest.mark.integration, pytest.mark.workflow]

MISTRAL_KEY_ENV = "MISTRAL_API_KEY"
OPENAI_KEY_ENV = "OPENAI_API_KEY"


def _get_key(env_var: str) -> str:
    key = os.getenv(env_var)
    if not key:
        pytest.skip(f"{env_var} not set")
    return key


def _make_async_mistral() -> AsyncFFLiteLLMClient:
    return AsyncFFLiteLLMClient(
        model_string="mistral/mistral-small-2503",
        api_key=_get_key(MISTRAL_KEY_ENV),
        temperature=0,
        max_tokens=50,
        system_instructions="Be concise. Answer in one short sentence.",
    )


def _make_async_openai() -> AsyncFFLiteLLMClient:
    return AsyncFFLiteLLMClient(
        model_string="openai/gpt-4o-mini",
        api_key=_get_key(OPENAI_KEY_ENV),
        temperature=0,
        max_tokens=50,
        system_instructions="Be concise. Answer in one short sentence.",
    )


class TestWorkflowDefaultClientOnly:
    """All steps use the default Mistral client — no per-step overrides."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.ffai = FFAI(_make_async_mistral())

    @pytest.mark.asyncio
    async def test_single_step_succeeds(self):
        result = await self.ffai.execute_workflow("""
workflow:
  name: single
  prompts:
    - name: greet
      prompt: "What is 2+2? Answer with just the number."
""")
        assert result.success_count == 1
        assert result.failed_count == 0
        assert "4" in result.results["greet"].response

    @pytest.mark.asyncio
    async def test_sequential_chain_with_interpolation(self):
        result = await self.ffai.execute_workflow("""
workflow:
  name: chain
  prompts:
    - name: lang
      prompt: "Name one programming language. Just the name."
    - name: creator
      prompt: "Who created {{lang.response}}? Just the name."
      history: [lang]
""")
        assert result.success_count == 2
        assert result.results["lang"].status == "success"
        assert result.results["creator"].status == "success"
        assert len(result.results["creator"].response.strip()) > 0

    @pytest.mark.asyncio
    async def test_variable_substitution(self):
        result = await self.ffai.execute_workflow("""
workflow:
  name: vars
  prompts:
    - name: topic_q
      prompt: "What is {topic}? Answer in one sentence."
""", variables={"topic": "machine learning"})
        assert result.success_count == 1
        response_lower = result.results["topic_q"].response.lower()
        assert "machine learning" in response_lower or "learn" in response_lower

    @pytest.mark.asyncio
    async def test_results_recorded_in_history(self):
        await self.ffai.execute_workflow("""
workflow:
  name: history_check
  prompts:
    - name: wf_hist
      prompt: "Say the word: workflow_history_test"
""")
        latest = self.ffai.get_latest_interaction_by_prompt_name("wf_hist")
        assert latest is not None
        assert "workflow_history_test" in latest["response"].lower()

    @pytest.mark.asyncio
    async def test_usage_tracked_per_step(self):
        result = await self.ffai.execute_workflow("""
workflow:
  name: usage
  prompts:
    - name: u1
      prompt: "Say: hello"
    - name: u2
      prompt: "Say: world"
""")
        assert result.success_count == 2
        for name in ("u1", "u2"):
            r = result.results[name]
            assert r.usage is not None
            assert r.usage.total_tokens > 0
            assert r.cost_usd >= 0
            assert r.duration_ms > 0


class TestWorkflowPerStepClient:
    """Steps route to different providers: Mistral default, OpenAI override."""

    @pytest.fixture(autouse=True)
    def setup(self):
        _get_key(MISTRAL_KEY_ENV)
        _get_key(OPENAI_KEY_ENV)
        self.ffai = FFAI(_make_async_mistral())

    @pytest.mark.asyncio
    async def test_default_mistral_override_openai(self):
        result = await self.ffai.execute_workflow("""
workflow:
  name: multi_client
  defaults:
    client: litellm-mistral-small
  clients:
    openai_reviewer:
      type: litellm
      provider_prefix: "openai/"
      model: "gpt-4o-mini"
      api_key_env: "OPENAI_API_KEY"
  prompts:
    - name: mistral_step
      prompt: "What is 3+4? Answer with just the number."
    - name: openai_step
      prompt: "What is 5+6? Answer with just the number."
      client: openai_reviewer
""")
        assert result.success_count == 2
        assert "7" in result.results["mistral_step"].response
        assert "11" in result.results["openai_step"].response

    @pytest.mark.asyncio
    async def test_different_clients_with_interpolation(self):
        result = await self.ffai.execute_workflow("""
workflow:
  name: cross_client_chain
  clients:
    openai_reviewer:
      type: litellm
      provider_prefix: "openai/"
      model: "gpt-4o-mini"
      api_key_env: "OPENAI_API_KEY"
  prompts:
    - name: mistral_fact
      prompt: "Name the capital of France. One word only."
    - name: openai_elaborate
      prompt: "In one sentence, describe {{mistral_fact.response}}."
      history: [mistral_fact]
      client: openai_reviewer
""")
        assert result.success_count == 2
        fact = result.results["mistral_fact"].response.lower()
        assert "paris" in fact
        assert result.results["openai_elaborate"].status == "success"
        assert len(result.results["openai_elaborate"].response.strip()) > 0

    @pytest.mark.asyncio
    async def test_three_steps_two_clients(self):
        result = await self.ffai.execute_workflow("""
workflow:
  name: three_two
  clients:
    openai_reviewer:
      type: litellm
      provider_prefix: "openai/"
      model: "gpt-4o-mini"
      api_key_env: "OPENAI_API_KEY"
  prompts:
    - name: step_a
      prompt: "What is 1+1? Just the number."
    - name: step_b
      prompt: "What is 2+2? Just the number."
      client: openai_reviewer
    - name: step_c
      prompt: "Summarize: A says {{step_a.response}}, B says {{step_b.response}}."
      history: [step_a, step_b]
""")
        assert result.success_count == 3
        assert "2" in result.results["step_a"].response
        assert "4" in result.results["step_b"].response
        assert result.results["step_c"].status == "success"


class TestWorkflowConditionalExecution:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ffai = FFAI(_make_async_mistral())

    @pytest.mark.asyncio
    async def test_condition_true_executes_step(self):
        result = await self.ffai.execute_workflow("""
workflow:
  name: cond_true
  prompts:
    - name: always
      prompt: "Say the word: yes"
    - name: conditional
      prompt: "What day comes after Monday? One word."
      condition: '{{always.status}} == "success"'
""")
        assert result.results["conditional"].status == "success"
        assert "tuesday" in result.results["conditional"].response.lower()

    @pytest.mark.asyncio
    async def test_condition_false_skips_step(self):
        result = await self.ffai.execute_workflow("""
workflow:
  name: cond_false
  prompts:
    - name: always
      prompt: "Say the word: yes"
    - name: skipped
      prompt: "This should not execute."
      condition: '{{always.response}} == "IMPOSSIBLE_VALUE_XYZ_12345"'
""")
        assert result.results["skipped"].status == "skipped"
        assert result.skipped_count == 1

    @pytest.mark.asyncio
    async def test_parallel_steps_with_different_conditions(self):
        result = await self.ffai.execute_workflow("""
workflow:
  name: parallel_cond
  prompts:
    - name: base
      prompt: "Say: hello"
    - name: run_me
      prompt: "What is 1+1? Just the number."
      condition: '{{base.status}} == "success"'
    - name: skip_me
      prompt: "This should be skipped."
      condition: '{{base.status}} == "failed"'
""")
        assert result.success_count == 2
        assert result.results["run_me"].status == "success"
        assert result.results["skip_me"].status == "skipped"


class TestWorkflowFromFile:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ffai = FFAI(_make_async_mistral())

    @pytest.mark.asyncio
    async def test_execute_from_yaml_file(self, tmp_path):
        wf = tmp_path / "pipeline.yaml"
        wf.write_text("""
workflow:
  name: file_pipeline
  prompts:
    - name: from_file
      prompt: "What is the capital of Japan? One word."
""")
        result = await self.ffai.execute_workflow_file(str(wf))
        assert result.spec_name == "file_pipeline"
        assert result.success_count == 1
        assert "tokyo" in result.results["from_file"].response.lower()

    @pytest.mark.asyncio
    async def test_execute_from_file_with_variables(self, tmp_path):
        wf = tmp_path / "vars.yaml"
        wf.write_text("""
workflow:
  name: file_vars
  prompts:
    - name: ask
      prompt: "What color is a {color}? One word."
""")
        result = await self.ffai.execute_workflow_file(
            str(wf), variables={"color": "banana"}
        )
        assert result.success_count == 1
        assert "yellow" in result.results["ask"].response.lower()


class TestWorkflowValidationLive:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.ffai = FFAI(_make_async_mistral())

    def test_valid_workflow_no_errors(self):
        errors, warnings = self.ffai.validate_workflow("""
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

    def test_cycle_detected(self):
        errors, warnings = self.ffai.validate_workflow("""
workflow:
  name: cyclic
  prompts:
    - name: a
      prompt: "{{b.response}}"
      history: [b]
    - name: b
      prompt: "{{a.response}}"
      history: [a]
""")
        assert len(errors) >= 1
        assert any("cycle" in e.lower() for e in errors)
