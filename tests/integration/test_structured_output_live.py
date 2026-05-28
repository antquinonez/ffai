# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT

"""Integration tests for structured output (response_model) against live APIs.

Tests that ``FFAI.generate_response(response_model=...)`` works end-to-end
with real LLM providers, verifying that the Pydantic model passed through
LiteLLM's ``response_format`` is correctly translated to provider-specific
structured output parameters.

Requires enabled clients in ``tests/integration/test_config.yaml``.
"""

import json

import pytest
from pydantic import BaseModel

from ffai.core.response_options import ResponseOptions
from ffai.FFAI import FFAI

pytestmark = pytest.mark.integration


class Sentiment(BaseModel):
    label: str
    confidence: float


class Score(BaseModel):
    value: int
    explanation: str


class Item(BaseModel):
    name: str
    price: float


class TestStructuredOutputLiteLLM:
    """Tests for structured output via LiteLLM-backed clients."""

    def test_sentiment_analysis(self, integration_client):
        ffai = FFAI(integration_client)
        result = ffai.generate_response(
            "The food was absolutely amazing and the service was terrible!",
            options=ResponseOptions(response_model=Sentiment),
        )
        assert result.parsed is not None
        assert isinstance(result.parsed, Sentiment)
        assert result.parsed.label in ("positive", "negative", "mixed", "neutral")
        assert 0.0 <= result.parsed.confidence <= 1.0

    def test_numeric_score(self, integration_client):
        ffai = FFAI(integration_client)
        result = ffai.generate_response(
            "Rate the movie 'The Matrix' on a scale of 0-100.",
            options=ResponseOptions(response_model=Score),
        )
        assert result.parsed is not None
        assert isinstance(result.parsed, Score)
        assert 0 <= result.parsed.value <= 100
        assert len(result.parsed.explanation) > 0

    def test_item_extraction(self, integration_client):
        ffai = FFAI(integration_client)
        result = ffai.generate_response(
            "Extract the product: a wireless mouse that costs $29.99.",
            options=ResponseOptions(response_model=Item),
        )
        assert result.parsed is not None
        assert isinstance(result.parsed, Item)
        assert "mouse" in result.parsed.name.lower()
        assert result.parsed.price > 0

    def test_raw_response_contains_structured_data(self, integration_client):
        ffai = FFAI(integration_client)
        result = ffai.generate_response(
            "Is the sky blue?",
            options=ResponseOptions(response_model=Sentiment),
        )
        assert result.response is not None
        if isinstance(result.response, dict):
            assert "label" in result.response
            assert "confidence" in result.response
        else:
            parsed_json = json.loads(result.response)
            assert "label" in parsed_json
            assert "confidence" in parsed_json

    def test_parsing_errors_none_on_success(self, integration_client):
        ffai = FFAI(integration_client)
        result = ffai.generate_response(
            "Is water wet?",
            options=ResponseOptions(response_model=Sentiment),
        )
        assert result.parsed is not None
        assert result.parsing_errors is None

    def test_custom_response_format_preserved(self, integration_client):
        ffai = FFAI(integration_client)
        custom_fmt = {"type": "json_object"}
        result = ffai.generate_response(
            "Name a fruit and its color.",
            options=ResponseOptions(response_model=Item, response_format=custom_fmt),
        )
        assert result.parsed is not None
        assert isinstance(result.parsed, Item)


class TestStructuredOutputFFMistralSmall:
    """Tests for structured output via the native Mistral SDK client."""

    def test_sentiment_analysis(self, ffmistralsmall_client):
        ffai = FFAI(ffmistralsmall_client)
        result = ffai.generate_response(
            "The concert was incredible, best night of my life!",
            options=ResponseOptions(response_model=Sentiment),
        )
        assert result.parsed is not None
        assert isinstance(result.parsed, Sentiment)
        assert result.parsed.label in ("positive", "negative", "mixed", "neutral")
        assert 0.0 <= result.parsed.confidence <= 1.0

    def test_numeric_score(self, ffmistralsmall_client):
        ffai = FFAI(ffmistralsmall_client)
        result = ffai.generate_response(
            "Rate Python as a programming language from 0-100.",
            options=ResponseOptions(response_model=Score),
        )
        assert result.parsed is not None
        assert isinstance(result.parsed, Score)
        assert 0 <= result.parsed.value <= 100
