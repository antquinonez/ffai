import pytest

from src.core.client_base import FFAIClientBase
from src.core.response_options import ResponseOptions
from src.FFAI import FFAI

pytestmark = pytest.mark.integration


class TestNamedPromptInterpolation:
    def test_single_reference_resolves(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.generate_response(
            "What is the capital of France?",
            prompt_name="capital_q",
        )
        result = ffai.generate_response(
            "Say the word '{{capital_q.response}}' and nothing else.",
            prompt_name="echo",
        )
        assert "paris" in result.response.lower()

    def test_chained_references(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.generate_response(
            "Name one primary color.",
            prompt_name="color",
        )
        result = ffai.generate_response(
            "Is '{{color.response}}' a warm color or cool color? Answer in one word.",
            prompt_name="warmth",
        )
        assert result.response.strip().lower() in (
            "warm",
            "cool",
            "a warm",
            "a cool",
        )

    def test_reference_in_multi_step_context(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.generate_response(
            "Name a programming language.",
            prompt_name="lang",
        )
        ffai.generate_response(
            "Who created {{lang.response}}?",
            prompt_name="creator",
        )
        result = ffai.generate_response(
            "In one sentence: {{lang.response}} was created by {{creator.response}}.",
            prompt_name="summary",
        )
        assert len(result.response.strip()) > 0
        summary_interaction = ffai.get_latest_interaction_by_prompt_name("summary")
        assert summary_interaction is not None
        assert "{{lang" not in summary_interaction.get("resolved_prompt", "")
        assert "{{creator" not in summary_interaction.get("resolved_prompt", "")


class TestNamedPromptHistory:
    def test_history_records_named_prompts(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.generate_response("Say: alpha", prompt_name="first")
        ffai.generate_response("Say: beta", prompt_name="second")

        history = ffai.get_prompt_attr_history()
        assert len(history) == 2
        names = [h.get("prompt_name") for h in history]
        assert "first" in names
        assert "second" in names

    def test_ordered_history_preserves_sequence(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.generate_response("Say: one", prompt_name="step1")
        ffai.generate_response("Say: two", prompt_name="step2")

        interactions = ffai.get_all_interactions()
        assert len(interactions) == 2
        assert interactions[0]["prompt_name"] == "step1"
        assert interactions[1]["prompt_name"] == "step2"

    def test_get_latest_by_name(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.generate_response("Say: first attempt", prompt_name="retry_me")
        ffai.generate_response("Say: second attempt", prompt_name="retry_me")

        latest = ffai.get_latest_interaction_by_prompt_name("retry_me")
        assert latest is not None
        assert "second" in latest["response"].lower()


class TestNamedPromptUsageAndCost:
    def test_each_named_call_tracks_usage(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        r1 = ffai.generate_response("Say: hello", prompt_name="greet")
        r2 = ffai.generate_response("Say: world", prompt_name="place")

        assert r1.usage is not None
        assert r1.usage.total_tokens > 0
        assert r2.usage is not None
        assert r2.usage.total_tokens > 0
        assert r1.cost_usd >= 0
        assert r2.cost_usd >= 0

    def test_interpolated_call_has_higher_input_tokens(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        r1 = ffai.generate_response(
            "List three animals.",
            prompt_name="animals",
        )
        r2 = ffai.generate_response(
            "Which of {{animals.response}} is the fastest? Name only that animal.",
            prompt_name="fastest",
        )
        assert r2.usage is not None
        assert r2.usage.input_tokens > 0
        resolved = ffai.get_latest_interaction_by_prompt_name("fastest")
        assert resolved is not None
        assert len(resolved.get("resolved_prompt", "")) > len("Which of  is the fastest?")


class TestResponseOptionsLive:
    def test_model_override_in_options(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        result = ffai.generate_response(
            "Say hello",
            prompt_name="greet",
            options=ResponseOptions(model=integration_client.model),
        )
        assert len(result.response.strip()) > 0
        assert result.model == integration_client.model

    def test_system_instructions_per_call(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        result = ffai.generate_response(
            "What is your name?",
            prompt_name="identity",
            options=ResponseOptions(
                system_instructions="You are a bot named BattleTest. Always say your name is BattleTest.",
            ),
        )
        assert "battletest" in result.response.lower()

    def test_condition_true_executes(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.generate_response("Say: elephant", prompt_name="animal")
        result = ffai.generate_response(
            "What sound does an elephant make?",
            prompt_name="sound",
            options=ResponseOptions(
                condition="'{{animal.response}}' != ''",
                history=["animal"],
            ),
        )
        assert result.status == "success"
        assert len(result.response.strip()) > 0

    def test_condition_false_skips(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.generate_response("Say: cat", prompt_name="animal")
        result = ffai.generate_response(
            "What sound does a dog make?",
            prompt_name="sound",
            options=ResponseOptions(
                condition="'{{animal.response}}' == 'unicorn'",
                history=["animal"],
            ),
        )
        assert result.status == "skipped"
        assert result.response is None
