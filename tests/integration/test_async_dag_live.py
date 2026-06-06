import os

import pytest

from ffai.Clients.AsyncFFLiteLLMClient import AsyncFFLiteLLMClient
from ffai.FFAI import FFAI

pytestmark = pytest.mark.integration


def _get_mistral_api_key():
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        pytest.skip("MISTRAL_API_KEY not set")
    return key


def _make_async_client() -> AsyncFFLiteLLMClient:
    api_key = _get_mistral_api_key()
    return AsyncFFLiteLLMClient(
        model_string="mistral/mistral-small-2503",
        api_key=api_key,
        temperature=0,
        max_tokens=50,
        system_instructions="Be concise. Answer in one short sentence.",
    )


class TestAsyncDAGBasicExecution:
    @pytest.fixture(autouse=True)
    def setup_ffai(self):
        self.ffai = FFAI(_make_async_client())

    @pytest.mark.asyncio
    async def test_sequential_prompts_execute(self):
        result = await self.ffai.workflow.execute_graph([
            {"prompt_name": "q1", "prompt": "What is 2+2? Answer with just the number."},
            {"prompt_name": "q2", "prompt": "What is 3+3? Answer with just the number."},
        ])
        assert result.success_count == 2
        assert result.failed_count == 0
        assert "q1" in result.results
        assert "q2" in result.results
        assert "4" in result.results["q1"].response
        assert "6" in result.results["q2"].response

    @pytest.mark.asyncio
    async def test_dag_with_interpolation(self):
        result = await self.ffai.workflow.execute_graph([
            {"prompt_name": "language", "prompt": "Name one programming language."},
            {
                "prompt_name": "use_case",
                "prompt": "What is {{language.response}} mainly used for? Answer in 3 words.",
                "history": ["language"],
            },
        ])
        assert result.success_count == 2
        use_case = result.results["use_case"]
        assert use_case.status == "success"
        assert len(use_case.response.strip()) > 0

    @pytest.mark.asyncio
    async def test_fan_out_parallel_execution(self):
        result = await self.ffai.workflow.execute_graph([
            {"prompt_name": "base", "prompt": "Name a color."},
            {
                "prompt_name": "shade",
                "prompt": "Is {{base.response}} a light or dark color? One word.",
                "history": ["base"],
            },
            {
                "prompt_name": "hex",
                "prompt": "What is a common hex code for {{base.response}}? Just the hex code.",
                "history": ["base"],
            },
        ])
        assert result.success_count == 3
        assert result.results["shade"].status == "success"
        assert result.results["hex"].status == "success"

    @pytest.mark.asyncio
    async def test_diamond_dependency(self):
        result = await self.ffai.workflow.execute_graph([
            {"prompt_name": "root", "prompt": "Name a fruit."},
            {
                "prompt_name": "color",
                "prompt": "What color is {{root.response}}?",
                "history": ["root"],
            },
            {
                "prompt_name": "taste",
                "prompt": "Is {{root.response}} sweet or sour?",
                "history": ["root"],
            },
            {
                "prompt_name": "summary",
                "prompt": "{{root.response}} is {{color.response}} and {{taste.response}}. Summarize.",
                "history": ["root", "color", "taste"],
            },
        ])
        assert result.success_count == 4
        summary = result.results["summary"]
        assert summary.status == "success"
        assert len(summary.response.strip()) > 0


class TestAsyncDAGConditionExecution:
    @pytest.fixture(autouse=True)
    def setup_ffai(self):
        self.ffai = FFAI(_make_async_client())

    @pytest.mark.asyncio
    async def test_condition_true_executes(self):
        result = await self.ffai.workflow.execute_graph([
            {"prompt_name": "check", "prompt": "Say the word: yes"},
            {
                "prompt_name": "follow_up",
                "prompt": "What day comes after Monday?",
                "history": ["check"],
                "condition": "'{{check.response}}' != ''",
            },
        ])
        assert result.results["follow_up"].status == "success"
        assert "tuesday" in result.results["follow_up"].response.lower()

    @pytest.mark.asyncio
    async def test_condition_false_skips(self):
        result = await self.ffai.workflow.execute_graph([
            {"prompt_name": "check", "prompt": "Say the word: yes"},
            {
                "prompt_name": "skip_me",
                "prompt": "This should not execute.",
                "history": ["check"],
                "condition": "'{{check.response}}' == 'IMPOSSIBLE_VALUE_XYZ'",
            },
        ])
        assert result.results["skip_me"].status == "skipped"
        assert result.skipped_count == 1

    @pytest.mark.asyncio
    async def test_abort_condition_stops_later_nodes(self):
        result = await self.ffai.workflow.execute_graph([
            {"prompt_name": "trigger", "prompt": "Say: abort"},
            {
                "prompt_name": "after_abort",
                "prompt": "This should be skipped due to abort.",
            },
        ])
        assert result.results["trigger"].status == "success"


class TestAsyncDAGUsageTracking:
    @pytest.fixture(autouse=True)
    def setup_ffai(self):
        self.ffai = FFAI(_make_async_client())

    @pytest.mark.asyncio
    async def test_each_node_tracks_usage(self):
        result = await self.ffai.workflow.execute_graph([
            {"prompt_name": "q1", "prompt": "Say: hello"},
            {"prompt_name": "q2", "prompt": "Say: world"},
        ])
        for name in ("q1", "q2"):
            r = result.results[name]
            assert r.usage is not None
            assert r.usage.total_tokens > 0
            assert r.cost_usd >= 0

    @pytest.mark.asyncio
    async def test_results_recorded_in_history(self):
        await self.ffai.workflow.execute_graph([
            {"prompt_name": "g1", "prompt": "Say: dag_test"},
        ])
        latest = self.ffai.history.get_latest_interaction_by_prompt_name("g1")
        assert latest is not None
        assert "dag_test" in latest.get("response", "").lower()


class TestAsyncDAGValidation:
    @pytest.fixture(autouse=True)
    def setup_ffai(self):
        self.ffai = FFAI(_make_async_client())

    def test_validate_graph_no_cycle(self):
        graph, warnings = self.ffai.workflow.validate_graph([
            {"prompt_name": "a", "prompt": "prompt a"},
            {"prompt_name": "b", "prompt": "prompt b", "history": ["a"]},
        ])
        assert len(graph.nodes) == 2

    def test_validate_graph_detects_cycle(self):
        with pytest.raises(ValueError, match="[Cc]ycle"):
            self.ffai.workflow.validate_graph([
                {"prompt_name": "a", "prompt": "prompt a", "history": ["b"]},
                {"prompt_name": "b", "prompt": "prompt b", "history": ["a"]},
            ])

    def test_sync_client_raises_type_error(self):
        from ffai.Clients.FFLiteLLMClient import FFLiteLLMClient

        sync_client = FFLiteLLMClient(
            model_string="mistral/mistral-small-2503",
            api_key=_get_mistral_api_key(),
        )
        ffai = FFAI(sync_client)
        with pytest.raises(TypeError, match="async client"):
            import asyncio
            asyncio.run(ffai.workflow.execute_graph([{"prompt_name": "x", "prompt": "test"}]))
