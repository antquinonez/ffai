import polars as pl
import pytest

from ffai.core.client_base import FFAIClientBase
from ffai.FFAI import FFAI

pytestmark = pytest.mark.integration


class TestPersistRoundTripLive:
    def test_persist_and_reload_all_histories(self, integration_client: FFAIClientBase, tmp_path):
        ffai = FFAI(integration_client)
        ffai.generate_response("Say: hello", prompt_name="greet")
        ffai.generate_response("Say: goodbye", prompt_name="farewell")

        ffai._exporter.persist_dir = str(tmp_path)
        ffai._exporter.persist_name = "live_test"
        assert ffai.persist_all_histories() is True

        reloaded = pl.read_parquet(tmp_path / "live_test_ordered.parquet")
        assert reloaded.height == 2
        assert reloaded["prompt_name"][0] == "greet"
        assert reloaded["prompt_name"][1] == "farewell"
        assert reloaded["model"][0] == integration_client.model

        history = pl.read_parquet(tmp_path / "live_test_history.parquet")
        assert history.height == 2
        assert "usage" in history.columns
        assert isinstance(history["usage"][0], str)

    def test_persist_contains_usage_and_history_fields(self, integration_client: FFAIClientBase, tmp_path):
        ffai = FFAI(integration_client)
        r = ffai.generate_response("Say: test", prompt_name="dep_check")

        ffai._exporter.persist_dir = str(tmp_path)
        ffai._exporter.persist_name = "fields"
        assert ffai.persist_all_histories() is True

        reloaded = pl.read_parquet(tmp_path / "fields_history.parquet")
        assert reloaded.height == 1
        assert "usage" in reloaded.columns
        assert isinstance(reloaded["usage"][0], str)
        assert "history" in reloaded.columns

    def test_search_after_persist_reload(self, integration_client: FFAIClientBase, tmp_path):
        ffai = FFAI(integration_client)
        ffai.generate_response("What is Python?", prompt_name="python_q")
        ffai.generate_response("What is Rust?", prompt_name="rust_q")

        found = ffai.search_history(prompt_name="python_q")
        assert found.height == 1
        assert found["prompt_name"][0] == "python_q"

        found_rust = ffai.search_history(prompt_name="rust_q")
        assert found_rust.height == 1
        assert found_rust["prompt_name"][0] == "rust_q"


class TestSearchHistoryLive:
    def test_search_by_text_finds_match(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.generate_response("Explain quantum computing briefly", prompt_name="quantum")
        ffai.generate_response("Explain baking bread briefly", prompt_name="bread")

        found = ffai.search_history(text="quantum")
        assert found.height >= 1

    def test_search_by_time_range(self, integration_client: FFAIClientBase):
        import time

        ffai = FFAI(integration_client)
        before = time.time()
        ffai.generate_response("Say: alpha", prompt_name="time_a")
        ffai.generate_response("Say: beta", prompt_name="time_b")
        after = time.time()

        found = ffai.search_history(start_time=before, end_time=after + 1)
        assert found.height == 2

    def test_search_no_match_returns_empty(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.generate_response("Say: xyz", prompt_name="xyz")

        found = ffai.search_history(text="zzzzzz_nonexistent_keyword")
        assert found.height == 0


class TestCrossPromptAnalyticsLive:
    def test_response_length_stats_with_real_data(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.generate_response("Say: short", prompt_name="short_call")
        ffai.generate_response(
            "Write a three-sentence explanation of neural networks.",
            prompt_name="long_call",
        )

        stats = ffai.get_response_length_stats()
        assert stats.height == 2
        assert "mean_length" in stats.columns
        assert "count" in stats.columns

    def test_interaction_counts_by_date(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.generate_response("Say: date_test_1", prompt_name="d1")
        ffai.generate_response("Say: date_test_2", prompt_name="d2")

        counts = ffai.interaction_counts_by_date()
        assert counts.height >= 1
        assert "date" in counts.columns
        assert "len" in counts.columns
        assert counts["len"].sum() == 2

    def test_model_stats_matches_call_count(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.generate_response("Say: stat1", prompt_name="s1")
        ffai.generate_response("Say: stat2", prompt_name="s2")
        ffai.generate_response("Say: stat3", prompt_name="s3")

        stats = ffai.get_model_stats_df()
        assert stats.height >= 1
        row = stats.filter(pl.col("model") == integration_client.model)
        assert row["count"][0] == 3

    def test_prompt_name_stats_counts(self, integration_client: FFAIClientBase):
        ffai = FFAI(integration_client)
        ffai.generate_response("Say: pns1", prompt_name="repeat_me")
        ffai.generate_response("Say: pns2", prompt_name="repeat_me")
        ffai.generate_response("Say: pns3", prompt_name="once")

        stats = ffai.get_prompt_name_stats_df()
        assert stats.height == 2
        repeat_row = stats.filter(pl.col("prompt_name") == "repeat_me")
        assert repeat_row["count"][0] == 2
        once_row = stats.filter(pl.col("prompt_name") == "once")
        assert once_row["count"][0] == 1
