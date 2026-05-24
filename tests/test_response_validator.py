from unittest.mock import MagicMock

from src.agent.response_validator import (
    ResponseValidator,
    ValidationResult,
    _build_validation_prompt,
    _parse_validation_response,
)
from src.core.client_base import FFAIClientBase
from src.core.response_result import ResponseResult


def _make_mock_client(responses):
    client = MagicMock(spec=FFAIClientBase)
    client.model = "test-model"
    raw_responses = []
    for r in responses:
        if isinstance(r, str):
            raw_responses.append(ResponseResult(response=r))
        else:
            raw_responses.append(r)
    client.generate_response = MagicMock(side_effect=raw_responses)
    return client


class TestBuildValidationPrompt:
    def test_contains_criteria_and_response(self):
        prompt = _build_validation_prompt("Must be JSON", '{"a": 1}')
        assert "Must be JSON" in prompt
        assert '{"a": 1}' in prompt
        assert "PASS" in prompt
        assert "FAIL" in prompt


class TestParseValidationResponse:
    def test_pass_exact(self):
        passed, critique = _parse_validation_response("PASS")
        assert passed is True
        assert critique is None

    def test_pass_with_whitespace(self):
        passed, critique = _parse_validation_response("  PASS  ")
        assert passed is True

    def test_pass_case_insensitive(self):
        passed, critique = _parse_validation_response("pass")
        assert passed is True

    def test_fail_with_reason(self):
        passed, critique = _parse_validation_response("FAIL: missing required field")
        assert passed is False
        assert critique == "missing required field"

    def test_fail_multiline_reason(self):
        passed, critique = _parse_validation_response("FAIL:\nline1\nline2")
        assert passed is False
        assert "line1" in critique

    def test_pass_embedded_in_text(self):
        passed, critique = _parse_validation_response("The response looks good. PASS")
        assert passed is True

    def test_neither_pass_nor_fail(self):
        passed, critique = _parse_validation_response("I am not sure about this")
        assert passed is False
        assert critique == "I am not sure about this"


class TestResponseValidator:
    def test_validate_pass_first_try(self):
        client = _make_mock_client(["PASS"])
        validator = ResponseValidator(client)

        result = validator.validate(
            response='{"score": 5}',
            criteria="Must be valid JSON with a score",
        )

        assert result.passed is True
        assert result.attempts == 1
        assert result.critique is None

    def test_validate_fail_no_retries(self):
        client = _make_mock_client(["FAIL: not valid"])
        validator = ResponseValidator(client)

        result = validator.validate(
            response="not json",
            criteria="Must be JSON",
            max_retries=0,
        )

        assert result.passed is False
        assert result.attempts == 1
        assert "not valid" in result.critique

    def test_validate_fail_with_retry_via_re_execute(self):
        call_count = [0]

        def re_execute_fn(prompt):
            call_count[0] += 1
            return ResponseResult(response="improved response")

        client = _make_mock_client(["FAIL: bad", "PASS"])
        validator = ResponseValidator(client)

        result = validator.validate(
            response="bad response",
            criteria="Must be good",
            max_retries=1,
            re_execute_fn=re_execute_fn,
        )

        assert result.passed is True
        assert result.attempts == 2
        assert call_count[0] == 1

    def test_validate_llm_call_fails(self):
        client = MagicMock(spec=FFAIClientBase)
        client.generate_response = MagicMock(side_effect=RuntimeError("API down"))

        validator = ResponseValidator(client)

        result = validator.validate(
            response="test",
            criteria="Must pass",
            max_retries=0,
        )

        assert result.passed is None
        assert result.attempts == 1
        assert "API down" in result.critique

    def test_validate_passes_model_and_temperature(self):
        client = _make_mock_client(["PASS"])
        validator = ResponseValidator(client, model="gpt-4", temperature=0.2)

        validator.validate(response="test", criteria="good?")

        client.generate_response.assert_called_once()
        call_kwargs = client.generate_response.call_args
        assert call_kwargs.kwargs.get("model") == "gpt-4"
        assert call_kwargs.kwargs.get("temperature") == 0.2

    def test_validate_exhausts_retries(self):
        client = _make_mock_client(["FAIL: bad1", "FAIL: bad2", "FAIL: bad3"])
        validator = ResponseValidator(client)

        result = validator.validate(
            response="bad",
            criteria="Must be good",
            max_retries=2,
        )

        assert result.passed is False
        assert result.attempts == 3

    def test_re_execute_fn_with_string_response(self):
        def re_execute_fn(prompt):
            return "string response"

        client = _make_mock_client(["FAIL: bad", "PASS"])
        validator = ResponseValidator(client)

        result = validator.validate(
            response="bad",
            criteria="good",
            max_retries=1,
            re_execute_fn=re_execute_fn,
        )

        assert result.passed is True

    def test_re_execute_fn_failure_continues(self):
        def re_execute_fn(prompt):
            raise RuntimeError("re-execute failed")

        client = _make_mock_client(["FAIL: bad", "PASS"])
        validator = ResponseValidator(client)

        result = validator.validate(
            response="bad",
            criteria="good",
            max_retries=1,
            re_execute_fn=re_execute_fn,
        )

        assert result.passed is True
        assert result.attempts == 2


class TestValidationResult:
    def test_pass_result(self):
        result = ValidationResult(passed=True, attempts=1)
        assert result.passed is True
        assert result.critique is None

    def test_fail_result(self):
        result = ValidationResult(passed=False, attempts=3, critique="bad")
        assert result.passed is False
        assert result.critique == "bad"

    def test_none_result(self):
        result = ValidationResult(passed=None, attempts=1, critique="API error")
        assert result.passed is None
