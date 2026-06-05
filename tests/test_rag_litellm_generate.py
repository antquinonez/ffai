from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from ffai.core.usage import TokenUsage
from ffai.rag.litellm_generate import litellm_generate_fn
from ffai.rag.types import GenerationResult


def _mock_response(
    content: str = "answer",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = content
    resp.usage = MagicMock()
    resp.usage.prompt_tokens = prompt_tokens
    resp.usage.completion_tokens = completion_tokens
    resp.usage.total_tokens = prompt_tokens + completion_tokens
    return resp


class TestLiteLLMGenerateFn:
    def test_returns_callable(self):
        fn = litellm_generate_fn(model="test-model")
        assert callable(fn)

    @patch("litellm.completion_cost", return_value=0.01)
    @patch("litellm.completion")
    def test_happy_path_populates_all_fields(self, mock_completion, mock_cost):
        mock_completion.return_value = _mock_response("hello world", 20, 15)
        fn = litellm_generate_fn(model="test-model", api_key="sk-test")
        result = fn("prompt")

        assert isinstance(result, GenerationResult)
        assert result.text == "hello world"
        assert isinstance(result.usage, TokenUsage)
        assert result.usage.input_tokens == 20
        assert result.usage.output_tokens == 15
        assert result.cost_usd == 0.01
        assert result.duration_ms is not None
        assert result.duration_ms > 0

    @patch("litellm.completion_cost", return_value=0.0)
    @patch("litellm.completion")
    def test_passes_model_and_params_to_completion(self, mock_completion, mock_cost):
        mock_completion.return_value = _mock_response()
        fn = litellm_generate_fn(
            model="gpt-4",
            api_key="sk-abc",
            temperature=0.7,
            max_tokens=2048,
        )
        fn("test prompt")

        mock_completion.assert_called_once()
        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "gpt-4"
        assert call_kwargs["api_key"] == "sk-abc"
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs["messages"] == [{"role": "user", "content": "test prompt"}]

    @patch("litellm.completion_cost", return_value=0.0)
    @patch("litellm.completion")
    def test_none_usage_defaults_to_token_usage_zeros(self, mock_completion, mock_cost):
        resp = _mock_response()
        resp.usage = None
        mock_completion.return_value = resp
        fn = litellm_generate_fn(model="test-model")
        result = fn("prompt")

        assert isinstance(result.usage, TokenUsage)
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0
        assert result.usage.total_tokens == 0

    @patch("litellm.completion_cost", side_effect=Exception("no pricing data"))
    @patch("litellm.completion")
    def test_completion_cost_exception_defaults_to_zero(self, mock_completion, mock_cost):
        mock_completion.return_value = _mock_response()
        fn = litellm_generate_fn(model="test-model")
        result = fn("prompt")

        assert result.cost_usd == 0.0

    @patch("litellm.completion_cost", return_value=None)
    @patch("litellm.completion")
    def test_completion_cost_none_defaults_to_zero(self, mock_completion, mock_cost):
        mock_completion.return_value = _mock_response()
        fn = litellm_generate_fn(model="test-model")
        result = fn("prompt")

        assert result.cost_usd == 0.0

    @patch("litellm.completion_cost", return_value=0.0)
    @patch("litellm.completion")
    def test_empty_content_returns_empty_string(self, mock_completion, mock_cost):
        resp = _mock_response()
        resp.choices[0].message.content = None
        mock_completion.return_value = resp
        fn = litellm_generate_fn(model="test-model")
        result = fn("prompt")

        assert result.text == ""

    @patch("litellm.completion_cost", return_value=0.0)
    @patch("litellm.completion")
    def test_extra_kwargs_forwarded_to_completion(self, mock_completion, mock_cost):
        mock_completion.return_value = _mock_response()
        fn = litellm_generate_fn(model="test-model", top_p=0.9)
        fn("prompt")

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["top_p"] == 0.9

    @patch("litellm.completion_cost", return_value=0.0)
    @patch("litellm.completion")
    def test_conflicting_kwargs_filtered(self, mock_completion, mock_cost):
        mock_completion.return_value = _mock_response()
        fn = litellm_generate_fn(model="original", messages=[{"role": "user", "content": "bad"}])
        fn("actual prompt")

        call_kwargs = mock_completion.call_args[1]
        assert call_kwargs["model"] == "original"
        assert call_kwargs["messages"] == [{"role": "user", "content": "actual prompt"}]

    @patch("litellm.completion_cost", return_value=0.0)
    @patch("litellm.completion")
    def test_duration_is_positive(self, mock_completion, mock_cost):
        mock_completion.return_value = _mock_response()
        fn = litellm_generate_fn(model="test-model")
        result = fn("prompt")

        assert result.duration_ms is not None
        assert result.duration_ms > 0

    @patch("litellm.completion_cost", return_value=0.0)
    @patch("litellm.completion")
    def test_no_api_key_omits_from_params(self, mock_completion, mock_cost):
        mock_completion.return_value = _mock_response()
        fn = litellm_generate_fn(model="test-model")
        fn("prompt")

        call_kwargs = mock_completion.call_args[1]
        assert "api_key" not in call_kwargs


class TestLiteLLMGenerateFnFieldMapping:
    @patch("litellm.completion_cost", return_value=0.0)
    @patch("litellm.completion")
    def test_prompt_tokens_mapped_to_input_tokens(self, mock_completion, mock_cost):
        mock_completion.return_value = _mock_response(prompt_tokens=42, completion_tokens=13)
        fn = litellm_generate_fn(model="test-model")
        result = fn("prompt")

        assert isinstance(result.usage, TokenUsage)
        assert result.usage.input_tokens == 42

    @patch("litellm.completion_cost", return_value=0.0)
    @patch("litellm.completion")
    def test_completion_tokens_mapped_to_output_tokens(self, mock_completion, mock_cost):
        mock_completion.return_value = _mock_response(prompt_tokens=42, completion_tokens=13)
        fn = litellm_generate_fn(model="test-model")
        result = fn("prompt")

        assert isinstance(result.usage, TokenUsage)
        assert result.usage.output_tokens == 13

    @patch("litellm.completion_cost", return_value=0.0)
    @patch("litellm.completion")
    def test_total_tokens_computed(self, mock_completion, mock_cost):
        mock_completion.return_value = _mock_response(prompt_tokens=42, completion_tokens=13)
        fn = litellm_generate_fn(model="test-model")
        result = fn("prompt")

        assert isinstance(result.usage, TokenUsage)
        assert result.usage.total_tokens == 55


class TestLiteLLMGenerateFnWithRAG:
    @patch("litellm.completion_cost", return_value=0.003)
    @patch("litellm.completion")
    def test_works_with_rag_query(self, mock_completion, mock_cost):
        from ffai.rag.rag import RAG
        from ffai.rag.types import SearchHit

        mock_completion.return_value = _mock_response("Python is a language.", 30, 20)

        embed = MagicMock(
            spec=["aembed", "aembed_single", "model", "provider", "is_local", "cache_stats", "clear_cache"],
        )
        embed.model = "mistral/mistral-embed"
        embed.aembed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        embed.aembed_single = AsyncMock(return_value=[0.1, 0.2, 0.3])

        store = MagicMock()
        store.asearch = AsyncMock(return_value=[
            SearchHit(content="Python docs", score=0.9, source="python.txt"),
        ])

        rag = RAG(embed=embed, store=store)
        fn = litellm_generate_fn(model="test-model", api_key="sk-test")
        result = rag.query("What is Python?", generate_fn=fn)

        assert result.answer == "Python is a language."
        assert result.cost_usd == 0.003
        assert isinstance(result.usage, TokenUsage)
        assert result.usage.input_tokens == 30
        assert result.usage.output_tokens == 20
        assert result.duration_ms is not None
        assert result.duration_ms > 0
