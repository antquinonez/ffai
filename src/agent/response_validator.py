# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""LLM-as-judge response validation with automatic retry.

Validates LLM responses by asking a second LLM call to judge PASS/FAIL.
On failure, the original prompt is augmented with the rejection reason
and re-executed up to ``max_retries`` times.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from ..core.client_base import FFAIClientBase
from ..core.response_result import ResponseResult

logger = logging.getLogger(__name__)

_PASS_PATTERN = re.compile(r"^PASS\s*$", re.IGNORECASE)
_FAIL_PATTERN = re.compile(r"FAIL\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


@dataclass
class ValidationResult:
    """Result of a response validation attempt.

    Attributes:
        passed: True if PASS, False if FAIL, None if validation itself failed.
        attempts: Number of validation attempts made.
        critique: Failure reason from the validator, or None if passed.

    """

    passed: bool | None
    attempts: int = 0
    critique: str | None = None


def _build_validation_prompt(criteria: str, response: str) -> str:
    """Build the prompt sent to the validator LLM.

    Args:
        criteria: The validation criteria to evaluate against.
        response: The response text to evaluate.

    Returns:
        Formatted validation prompt.

    """
    return (
        "You are a response validator. Evaluate the response against the criteria below.\n"
        'Reply with exactly "PASS" if acceptable, or "FAIL: <reason>" if not.\n\n'
        f"Criteria: {criteria}\n\n"
        f"Response to evaluate:\n{response}"
    )


def _parse_validation_response(text: str) -> tuple[bool | None, str | None]:
    """Parse a PASS/FAIL response from the validator.

    Args:
        text: Raw validator response.

    Returns:
        Tuple of (passed, critique).  passed is True/False/None.

    """
    text = text.strip()
    if _PASS_PATTERN.match(text):
        return True, None
    fail_match = _FAIL_PATTERN.match(text)
    if fail_match:
        return False, fail_match.group(1).strip()
    if "PASS" in text.upper():
        return True, None
    return False, text


class ResponseValidator:
    """Validates LLM responses using a second LLM as judge.

    Usage:
        validator = ResponseValidator(client)
        result = validator.validate(
            response="The answer is 42",
            criteria="Must provide a numerical answer",
        )
        if result.passed:
            print("Response is valid")

    Args:
        client: FFAIClientBase used for validation LLM calls.
        model: Model to use for validation (defaults to client model).
        temperature: Temperature for validation calls (low for consistency).

    """

    def __init__(
        self,
        client: FFAIClientBase,
        model: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        self.client = client
        self.model = model
        self.temperature = temperature

    def validate(
        self,
        response: str,
        criteria: str,
        max_retries: int = 2,
        re_execute_fn: Any | None = None,
    ) -> ValidationResult:
        """Validate a response against criteria.

        Args:
            response: The response text to validate.
            criteria: Validation criteria description.
            max_retries: Maximum re-execution attempts on failure.
            re_execute_fn: Optional callable that accepts an augmented prompt
                and returns a new response string. If provided, failed
                validations trigger re-execution with the rejection reason.

        Returns:
            ValidationResult with pass/fail status and critique.

        """
        best_response = response
        last_critique: str | None = None

        for attempt in range(1, max_retries + 2):
            validation_prompt = _build_validation_prompt(criteria, best_response)

            try:
                raw_result = self.client.generate_response(
                    prompt=validation_prompt,
                    model=self.model,
                    temperature=self.temperature,
                )
                val_text = (
                    raw_result.response.strip()
                    if isinstance(raw_result, ResponseResult)
                    else str(raw_result).strip()
                )
            except Exception as e:
                logger.warning(f"Validation LLM call failed on attempt {attempt}: {e}")
                last_critique = f"Validation call failed: {e}"
                if attempt > max_retries:
                    return ValidationResult(
                        passed=None,
                        attempts=attempt,
                        critique=last_critique,
                    )
                continue

            passed, critique = _parse_validation_response(val_text)

            if passed:
                logger.info(f"Validation passed on attempt {attempt}/{max_retries + 1}")
                return ValidationResult(passed=True, attempts=attempt)

            last_critique = critique
            logger.info(
                f"Validation failed on attempt {attempt}/{max_retries + 1}: "
                f"{(critique or '')[:100]}"
            )

            if attempt > max_retries:
                break

            if re_execute_fn is not None:
                augmented_prompt = (
                    f"{criteria}\n\n"
                    f"[Previous attempt produced this response, which was rejected:]\n"
                    f"{best_response}\n\n"
                    f"[Rejection reason:]\n"
                    f"{last_critique}\n\n"
                    f"Please try again, addressing the rejection reason."
                )
                try:
                    new_response = re_execute_fn(augmented_prompt)
                    if isinstance(new_response, ResponseResult):
                        best_response = new_response.response
                    elif isinstance(new_response, str):
                        best_response = new_response
                except Exception as e:
                    logger.warning(f"Re-execution failed on attempt {attempt}: {e}")
                    continue

        logger.warning(f"Validation failed after {max_retries + 1} attempt(s)")
        return ValidationResult(
            passed=False,
            attempts=max_retries + 1,
            critique=last_critique,
        )
