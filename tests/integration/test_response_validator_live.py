import pytest

from ffai.agent.response_validator import ResponseValidator, ValidationResult
from ffai.core.client_base import FFAIClientBase

pytestmark = pytest.mark.integration


class TestResponseValidatorPass:
    def test_obviously_correct_response_passes(self, integration_client: FFAIClientBase):
        validator = ResponseValidator(integration_client)
        result = validator.validate(
            response="Paris is the capital of France.",
            criteria="The response must state that Paris is the capital of France.",
        )
        assert isinstance(result, ValidationResult)
        assert result.passed is True
        assert result.attempts >= 1

    def test_factual_response_passes_factual_criteria(self, integration_client: FFAIClientBase):
        validator = ResponseValidator(integration_client)
        result = validator.validate(
            response="Water boils at 100 degrees Celsius at standard atmospheric pressure.",
            criteria="The response must mention that water boils at 100 degrees Celsius.",
        )
        assert result.passed is True


class TestResponseValidatorFail:
    def test_wrong_answer_fails(self, integration_client: FFAIClientBase):
        validator = ResponseValidator(integration_client)
        result = validator.validate(
            response="The capital of France is London.",
            criteria="The response must correctly state that Paris is the capital of France.",
            max_retries=0,
        )
        assert isinstance(result, ValidationResult)
        assert result.passed is False
        assert result.critique is not None
        assert len(result.critique.strip()) > 0

    def test_empty_response_fails(self, integration_client: FFAIClientBase):
        validator = ResponseValidator(integration_client)
        result = validator.validate(
            response="",
            criteria="The response must contain at least one sentence about Python.",
            max_retries=0,
        )
        assert result.passed is False


class TestResponseValidatorRetry:
    def test_retry_with_re_execute_fn(self, integration_client: FFAIClientBase):
        call_count = 0

        def re_execute(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "I have no idea."
            return "Python is a high-level programming language known for readability."

        validator = ResponseValidator(integration_client)
        result = validator.validate(
            response="I don't know anything about programming.",
            criteria="The response must mention Python as a programming language.",
            max_retries=2,
            re_execute_fn=re_execute,
        )
        assert isinstance(result, ValidationResult)
        assert result.attempts >= 1
        assert call_count >= 1

    def test_max_retries_limits_attempts(self, integration_client: FFAIClientBase):
        attempts_seen = []

        def re_execute(prompt: str) -> str:
            attempts_seen.append(prompt)
            return "Nope still wrong"

        validator = ResponseValidator(integration_client)
        result = validator.validate(
            response="Wrong answer",
            criteria="The response must say 'unicorn'.",
            max_retries=1,
            re_execute_fn=re_execute,
        )
        assert result.passed is False
        assert result.attempts <= 3


class TestResponseValidatorAttempts:
    def test_single_attempt_on_pass(self, integration_client: FFAIClientBase):
        validator = ResponseValidator(integration_client)
        result = validator.validate(
            response="Two plus two equals four.",
            criteria="The response must state that 2+2=4.",
        )
        assert result.passed is True
        assert result.attempts == 1

    def test_attempts_increments_on_each_try(self, integration_client: FFAIClientBase):
        validator = ResponseValidator(integration_client)
        result = validator.validate(
            response="Banana",
            criteria="The response must be about quantum physics with equations.",
            max_retries=0,
        )
        assert result.attempts >= 1
