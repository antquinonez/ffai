# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

import json

from pydantic import BaseModel, Field

from ffai.core.structured_output import StructuredOutputHandler, StructuredResult


class Sentiment(BaseModel):
    label: str
    confidence: float


class NestedModel(BaseModel):
    name: str
    address: dict[str, str]


class StrictModel(BaseModel):
    score: int = Field(ge=0, le=100)
    tags: list[str]


class TestStructuredResult:
    def test_defaults(self):
        result = StructuredResult(parsed=None, raw_response="text", attempts=1)
        assert result.parsed is None
        assert result.raw_response == "text"
        assert result.attempts == 1
        assert result.parsing_errors == []

    def test_with_parsed(self):
        instance = Sentiment(label="positive", confidence=0.9)
        result = StructuredResult(parsed=instance, raw_response="{}", attempts=1)
        assert result.parsed is not None
        parsed = result.parsed
        assert isinstance(parsed, Sentiment)
        assert parsed.label == "positive"


class TestBuildResponseFormat:
    def test_returns_model_class(self):
        handler = StructuredOutputHandler()
        fmt = handler.build_response_format(Sentiment)
        assert fmt is Sentiment

    def test_returns_subclass_of_base_model(self):
        handler = StructuredOutputHandler()
        from pydantic import BaseModel as _BM
        fmt = handler.build_response_format(Sentiment)
        assert issubclass(fmt, _BM)

    def test_identity_preserves_model_attributes(self):
        handler = StructuredOutputHandler()
        fmt = handler.build_response_format(Sentiment)
        schema = fmt.model_json_schema()
        assert "properties" in schema
        assert "label" in schema["properties"]
        assert "confidence" in schema["properties"]
        assert "required" in schema
        assert "label" in schema["required"]


class TestBuildSystemSuffix:
    def test_contains_schema(self):
        handler = StructuredOutputHandler()
        suffix = handler.build_system_suffix(Sentiment)
        assert "json" in suffix.lower()
        assert "label" in suffix
        assert "confidence" in suffix

    def test_includes_instruction(self):
        handler = StructuredOutputHandler()
        suffix = handler.build_system_suffix(Sentiment)
        assert "MUST respond" in suffix

    def test_valid_json_output(self):
        handler = StructuredOutputHandler()
        suffix = handler.build_system_suffix(Sentiment)
        schema_start = suffix.index("{")
        schema_end = suffix.rindex("}") + 1
        schema_json = suffix[schema_start:schema_end]
        parsed = json.loads(schema_json)
        assert "properties" in parsed


class TestValidate:
    def test_valid_json_returns_parsed(self):
        handler = StructuredOutputHandler()
        response = json.dumps({"label": "positive", "confidence": 0.95})
        result = handler.validate(response, Sentiment)
        assert result.parsed is not None
        parsed = result.parsed
        assert isinstance(parsed, Sentiment)
        assert parsed.label == "positive"
        assert parsed.confidence == 0.95
        assert result.parsing_errors == []

    def test_valid_json_in_markdown(self):
        handler = StructuredOutputHandler()
        response = '```json\n{"label": "negative", "confidence": 0.1}\n```'
        result = handler.validate(response, Sentiment)
        assert result.parsed is not None
        parsed = result.parsed
        assert isinstance(parsed, Sentiment)
        assert parsed.label == "negative"

    def test_invalid_json_returns_error(self):
        handler = StructuredOutputHandler()
        result = handler.validate("plain text response", Sentiment)
        assert result.parsed is None
        assert len(result.parsing_errors) > 0
        assert "No JSON" in result.parsing_errors[0]

    def test_valid_json_wrong_schema_returns_error(self):
        handler = StructuredOutputHandler()
        response = json.dumps({"wrong_key": "value"})
        result = handler.validate(response, Sentiment)
        assert result.parsed is None
        assert len(result.parsing_errors) > 0

    def test_extra_fields_still_pass(self):
        handler = StructuredOutputHandler()
        response = json.dumps({"label": "neutral", "confidence": 0.5, "extra": "ignored"})
        result = handler.validate(response, Sentiment)
        assert result.parsed is not None
        parsed = result.parsed
        assert isinstance(parsed, Sentiment)
        assert parsed.label == "neutral"

    def test_nested_model(self):
        handler = StructuredOutputHandler()
        response = json.dumps({"name": "Alice", "address": {"city": "NYC", "zip": "10001"}})
        result = handler.validate(response, NestedModel)
        assert result.parsed is not None
        parsed = result.parsed
        assert isinstance(parsed, NestedModel)
        assert parsed.name == "Alice"
        assert parsed.address["city"] == "NYC"

    def test_wrong_types_converted(self):
        handler = StructuredOutputHandler()
        response = json.dumps({"label": "positive", "confidence": "0.95"})
        result = handler.validate(response, Sentiment)
        assert result.parsed is not None
        parsed = result.parsed
        assert isinstance(parsed, Sentiment)
        assert isinstance(parsed.confidence, float)

    def test_array_response_returns_error(self):
        handler = StructuredOutputHandler()
        response = json.dumps([{"label": "positive", "confidence": 0.9}])
        result = handler.validate(response, Sentiment)
        assert result.parsed is None
        assert any("array" in e.lower() for e in result.parsing_errors)

    def test_strict_model_validation_failure(self):
        handler = StructuredOutputHandler()
        response = json.dumps({"score": 150, "tags": ["a"]})
        result = handler.validate(response, StrictModel)
        assert result.parsed is None
        assert len(result.parsing_errors) > 0

    def test_strict_model_validation_success(self):
        handler = StructuredOutputHandler()
        response = json.dumps({"score": 50, "tags": ["a", "b"]})
        result = handler.validate(response, StrictModel)
        assert result.parsed is not None
        parsed = result.parsed
        assert isinstance(parsed, StrictModel)
        assert parsed.score == 50

    def test_attempts_always_one(self):
        handler = StructuredOutputHandler()
        result = handler.validate("{}", Sentiment)
        assert result.attempts == 1

    def test_raw_response_preserved(self):
        handler = StructuredOutputHandler()
        response = '{"label": "positive", "confidence": 0.9}'
        result = handler.validate(response, Sentiment)
        assert result.raw_response == response


class TestBuildRetryFeedback:
    def test_contains_errors(self):
        handler = StructuredOutputHandler()
        feedback = handler.build_retry_feedback(["Missing field: label"])
        assert "Missing field: label" in feedback
        assert "fix" in feedback.lower()

    def test_multiple_errors(self):
        handler = StructuredOutputHandler()
        errors = ["Error 1", "Error 2"]
        feedback = handler.build_retry_feedback(errors)
        assert "Error 1" in feedback
        assert "Error 2" in feedback


class TestHandlerMaxRetries:
    def test_default_max_retries(self):
        handler = StructuredOutputHandler()
        assert handler.max_retries == 2

    def test_custom_max_retries(self):
        handler = StructuredOutputHandler(max_retries=5)
        assert handler.max_retries == 5
