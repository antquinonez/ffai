# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Pydantic-validated structured output for LLM responses.

Provides ``StructuredOutputHandler`` which converts Pydantic models into
JSON Schema instructions, validates LLM output against the schema, and
supports retry with error feedback on validation failure.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .response_utils import extract_json

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


@dataclass
class StructuredResult:
    """Outcome of a structured-output validation attempt.

    Attributes:
        parsed: Validated Pydantic model instance, or None on failure.
        raw_response: The original LLM response string.
        attempts: Number of validation attempts made.
        parsing_errors: List of error strings from failed validations.

    """

    parsed: BaseModel | None
    raw_response: str
    attempts: int
    parsing_errors: list[str] = field(default_factory=list)


class StructuredOutputHandler:
    """Validates LLM responses against Pydantic models.

    Args:
        max_retries: Maximum number of re-prompt attempts after validation
            failure. Total attempts = max_retries + 1.

    """

    def __init__(self, max_retries: int = 2) -> None:
        self.max_retries = max_retries

    def build_response_format(self, model: type[BaseModel]) -> dict[str, Any]:
        """Convert a Pydantic model to a JSON Schema response_format dict.

        Args:
            model: A Pydantic BaseModel subclass.

        Returns:
            Dict compatible with ``litellm.completion(response_format=...)``.

        """
        schema = model.model_json_schema()
        return {"type": "json_object", "schema": schema}

    def build_system_suffix(self, model: type[BaseModel]) -> str:
        """Generate instruction text describing the expected JSON schema.

        Intended to be appended to system instructions so the model knows
        the exact output shape.

        Args:
            model: A Pydantic BaseModel subclass.

        Returns:
            Instruction string with embedded JSON Schema.

        """
        schema = model.model_json_schema()
        return (
            "\n\nYou MUST respond with valid JSON matching this schema:\n"
            f"{json.dumps(schema, indent=2)}\n"
            "Do not include any text outside the JSON object."
        )

    def validate(self, response: str, model: type[T]) -> StructuredResult:
        """Parse and validate an LLM response against a Pydantic model.

        Uses ``json_repair`` for fault-tolerant JSON parsing, then Pydantic
        validation for type safety.

        Args:
            response: Raw LLM response text.
            model: Pydantic BaseModel subclass to validate against.

        Returns:
            ``StructuredResult`` with ``parsed`` set on success, or
            ``parsing_errors`` populated on failure.

        """
        errors: list[str] = []

        parsed_json = extract_json(response)

        if parsed_json is None:
            return StructuredResult(
                parsed=None,
                raw_response=response,
                attempts=1,
                parsing_errors=["No JSON found in response"],
            )

        if isinstance(parsed_json, list):
            errors.append(
                "Response is a JSON array; expected a JSON object matching the schema"
            )
            return StructuredResult(
                parsed=None,
                raw_response=response,
                attempts=1,
                parsing_errors=errors,
            )

        try:
            instance = model.model_validate(parsed_json)
            return StructuredResult(
                parsed=instance,
                raw_response=response,
                attempts=1,
            )
        except ValidationError as e:
            errors.append(str(e))
            return StructuredResult(
                parsed=None,
                raw_response=response,
                attempts=1,
                parsing_errors=errors,
            )

    def prepare_retry_state(
        self, prompt: str, response_model: type[BaseModel]
    ) -> tuple[list[str], str, StructuredResult | None]:
        """Initialize retry state for a structured output loop.

        Args:
            prompt: The original resolved prompt.
            response_model: Pydantic BaseModel subclass.

        Returns:
            Tuple of (all_errors, current_prompt, best_result).

        """
        return [], prompt, None

    def process_attempt(
        self,
        response: str,
        response_model: type[BaseModel],
        prompt: str,
        attempt: int,
        all_errors: list[str],
        best_result: StructuredResult | None,
    ) -> tuple[StructuredResult | None, str, list[str], bool]:
        """Process a single structured output attempt.

        Args:
            response: The raw LLM response.
            response_model: Pydantic BaseModel subclass.
            prompt: The original resolved prompt (for feedback).
            attempt: Zero-indexed attempt number.
            all_errors: Accumulated error list.
            best_result: Best result so far.

        Returns:
            Tuple of (updated_best_result, next_prompt, updated_errors, should_stop).

        """
        result = self.validate(response, response_model)
        result.attempts = attempt + 1

        if result.parsed is not None:
            return result, prompt, all_errors, True

        all_errors.extend(result.parsing_errors)
        feedback = self.build_retry_feedback(result.parsing_errors)
        next_prompt = prompt + feedback
        logger.info(
            f"Structured output attempt {attempt + 1}/{self.max_retries + 1} failed validation, retrying"
        )
        return result, next_prompt, all_errors, False

    def finalize_retry(
        self,
        best_result: StructuredResult | None,
        all_errors: list[str],
        max_attempts: int,
    ) -> StructuredResult:
        """Finalize after all retry attempts exhausted.

        Args:
            best_result: Last validation result.
            all_errors: All accumulated errors.
            max_attempts: Total attempts made.

        Returns:
            Final ``StructuredResult``.

        """
        if best_result is not None:
            best_result.parsing_errors = all_errors
            best_result.attempts = max_attempts
            return best_result

        return StructuredResult(
            parsed=None, raw_response="", attempts=max_attempts, parsing_errors=all_errors
        )

    def build_retry_feedback(self, errors: list[str]) -> str:
        """Build a prompt suffix telling the LLM what went wrong.

        Args:
            errors: Validation error strings from the previous attempt.

        Returns:
            Feedback text to append to the retry prompt.

        """
        error_text = "\n".join(errors)
        return (
            "\n\nYour previous response failed validation:\n"
            f"{error_text}\n\n"
            "Please fix the errors and respond with valid JSON."
        )
