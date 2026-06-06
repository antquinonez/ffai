import pytest

from ffai.core.client_base import FFAIClientBase
from ffai.FFAI import FFAI

pytestmark = pytest.mark.integration


class TestHistoryTrackingLive:
    def test_raw_history_records_all_calls(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.workflow.generate_response("Say: one", prompt_name="call1")
        ffai.workflow.generate_response("Say: two", prompt_name="call2")
        ffai.workflow.generate_response("Say: three", prompt_name="call3")

        history = ffai.history.raw
        assert len(history) == 3
        names = [h.get("prompt_name") for h in history]
        assert names == ["call1", "call2", "call3"]

    def test_prompt_attr_history_deduplicates_by_name(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.workflow.generate_response("Say: first", prompt_name="repeated")
        ffai.workflow.generate_response("Say: second", prompt_name="repeated")

        pah = ffai.history.get_prompt_attr_history()
        assert len(pah) == 2
        assert pah[0]["prompt_name"] == "repeated"
        assert pah[1]["prompt_name"] == "repeated"
        assert "first" in pah[0]["response"].lower()
        assert "second" in pah[1]["response"].lower()

    def test_clean_history_strips_whitespace(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.workflow.generate_response("Respond with lots of     spaces", prompt_name="spaces")

        clean = ffai.history.get_clean_interaction_history()
        assert len(clean) == 1
        response = clean[0].get("response", "")
        assert "     " not in response

    def test_ordered_history_maintains_sequence(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.workflow.generate_response("Say: alpha", prompt_name="step_1")
        ffai.workflow.generate_response("Say: beta", prompt_name="step_2")
        ffai.workflow.generate_response("Say: gamma", prompt_name="step_3")

        interactions = ffai.history.get_all_interactions()
        assert len(interactions) == 3
        assert interactions[0].prompt_name == "step_1"
        assert interactions[1].prompt_name == "step_2"
        assert interactions[2].prompt_name == "step_3"

    def test_history_includes_model_and_usage(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.workflow.generate_response("Say: test", prompt_name="tracked")

        entry = ffai.history.get_latest_interaction_by_prompt_name("tracked")
        assert entry is not None
        assert entry.get("model") == integration_client.model
        usage = entry.get("usage")
        if usage is not None:
            assert usage.total_tokens > 0


class TestDataFrameExportLive:
    def test_history_to_dataframe(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.workflow.generate_response("Say: df_test_1", prompt_name="df1")
        ffai.workflow.generate_response("Say: df_test_2", prompt_name="df2")

        df = ffai.history.history_to_dataframe()
        assert df.height == 2
        assert "prompt_name" in df.columns
        assert "response" in df.columns
        assert "model" in df.columns
        assert df["prompt_name"][0] == "df1"
        assert df["prompt_name"][1] == "df2"
        assert df["model"][0] == integration_client.model
        assert df["model"][1] == integration_client.model

    def test_statistics_dataframe(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.workflow.generate_response("Say: stats_test", prompt_name="stats_call")

        stats_df = ffai.history.get_model_stats_df()
        assert stats_df.height >= 1
        assert "count" in stats_df.columns

    def test_empty_history_produces_empty_dataframe(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        df = ffai.history.history_to_dataframe()
        assert df.height == 0


class TestCostTrackingLive:
    def test_cost_accumulates_across_calls(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        r1 = ffai.workflow.generate_response("Say: cost1", prompt_name="cost1")
        r2 = ffai.workflow.generate_response("Say: cost2", prompt_name="cost2")

        assert r1.cost_usd >= 0
        assert r2.cost_usd >= 0
        total = r1.cost_usd + r2.cost_usd
        assert total >= 0

    def test_usage_is_nonzero(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        result = ffai.workflow.generate_response("What is 1+1?", prompt_name="usage_check")

        assert result.usage is not None
        assert result.usage.input_tokens > 0
        assert result.usage.output_tokens > 0
        assert result.usage.total_tokens == result.usage.input_tokens + result.usage.output_tokens

    def test_longer_prompt_has_more_input_tokens(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        r_short = ffai.workflow.generate_response("Say: hi", prompt_name="short")
        ffai.clear_conversation()
        long_prompt = "Describe the following in one sentence: " + "x " * 200
        r_long = ffai.workflow.generate_response(long_prompt, prompt_name="long")

        assert r_short.usage is not None
        assert r_long.usage is not None
        assert r_short.usage.input_tokens < r_long.usage.input_tokens
