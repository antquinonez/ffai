# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Response cleaning and JSON extraction utilities.

Uses ``json_repair`` for fault-tolerant parsing of LLM output that may
contain trailing commas, unquoted keys, or other common JSON mistakes.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from json_repair import loads as json_repair_loads

logger = logging.getLogger(__name__)

_MARKDOWN_PATTERN = re.compile(r"```(?:json)?\s*(?P<content>[\s\S]*?)\s*```")
_THINK_TAG_PATTERN = re.compile(r"<think[\s\S]*?</think\s*>")
_JSON_LIKE_PATTERN = re.compile(r"^\s*[\[{]")


def _clean_text(text: str) -> str:
    return text.strip().replace("\ufeff", "")


def _extract_from_markdown(text: str) -> str | None:
    if match := _MARKDOWN_PATTERN.search(text):
        return _clean_text(match.group("content"))
    return None


def extract_json(text: str) -> Any | None:
    """Extract JSON from text using fault-tolerant ``json_repair``.

    Handles markdown code blocks, trailing commas, unquoted keys, and
    other common LLM JSON mistakes.

    Args:
        text: Response text that may contain JSON.

    Returns:
        Parsed JSON object or None if no valid JSON found.

    """
    text = _clean_text(text)
    if not text:
        return None

    markdown_content = _extract_from_markdown(text)
    if markdown_content:
        try:
            return json_repair_loads(markdown_content)
        except Exception:
            pass

    if _JSON_LIKE_PATTERN.match(text):
        try:
            return json_repair_loads(text)
        except Exception:
            pass

    return None


def clean_response(response: Any) -> Any:
    """Process and validate a response, removing think tags and extracting JSON.

    Args:
        response: The raw response from the AI client.

    Returns:
        Cleaned response. If JSON is detected, returns the parsed object.
        Otherwise returns the cleaned string with think tags removed.

    """
    if not isinstance(response, str):
        return response

    response = _THINK_TAG_PATTERN.sub("", response)

    cleaned_json = extract_json(response)

    if cleaned_json is not None:
        if isinstance(cleaned_json, dict):
            for key, value in cleaned_json.items():
                if isinstance(value, str):
                    cleaned_json[key] = _THINK_TAG_PATTERN.sub("", value)
        return cleaned_json
    return response
