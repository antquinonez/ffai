# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Synchronous LiteLLM-backed AI client implementing FFAIClientBase contract.

Delegates all shared logic to ``BaseLiteLLMClient`` and provides only the
synchronous ``completion()`` call and ``clone()`` factory.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from litellm import completion

from ..core.client_base import FFAIClientBase
from ..retry_utils import get_configured_retry_decorator
from .BaseLiteLLMClient import BaseLiteLLMClient

logger = logging.getLogger(__name__)


class FFLiteLLMClient(BaseLiteLLMClient, FFAIClientBase):
    """LiteLLM-backed AI client implementing FFAIClientBase.

    This client wraps LiteLLM's completion() function while maintaining
    the FFAIClientBase contract for compatibility with FFAI wrapper.

    Key features:
    - Internal conversation history management
    - Clone pattern for parallel execution
    - Model string routing (e.g., "azure/mistral-small-2503")
    - Retry and fallback support

    Args:
        model_string: LiteLLM model identifier (e.g., "openai/gpt-4", "azure/my-deployment")
        config: Optional configuration dictionary
        api_key: API key (overrides env var)
        api_base: API base URL (overrides env var)
        system_instructions: System prompt
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        fallbacks: List of fallback model strings
        retry_config: Retry configuration

    Example:
        >>> client = FFLiteLLMClient(model_string="azure/mistral-small-2503")
        >>> response = client.generate_response("Hello!")
        >>>
        >>> # With fallbacks
        >>> client = FFLiteLLMClient(
        ...     model_string="anthropic/claude-3-opus",
        ...     fallbacks=["openai/gpt-4", "azure/gpt-4"]
        ... )

    """

    def generate_response(
        self,
        prompt: str,
        model: str | None = None,
        system_instructions: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the AI model with retry and fallback logic.

        Retries are handled by ``retry_utils.get_configured_retry_decorator``
        on the inner ``_call_primary`` method. If the primary model (and all
        its retries) fail, fallback models are tried once each.

        Args:
            prompt: The user prompt
            model: Override model (appends to provider prefix)
            system_instructions: Override system instructions
            temperature: Override temperature
            max_tokens: Override max tokens
            **kwargs: Additional LiteLLM parameters

        Returns:
            The generated response text

        Raises:
            ValueError: If prompt is empty
            RuntimeError: If all models (including fallbacks) fail

        """
        api_params, model_string = self._prepare_generate_params(
            prompt, model, system_instructions, temperature, max_tokens, **kwargs
        )

        logger.debug(
            f"Calling LiteLLM with model={model_string}, temperature={api_params.get('temperature')}"
        )

        try:
            with self._trace_llm_call(model_string):
                return self._call_primary(api_params, model_string, prompt)
        except Exception as e:
            if self._fallbacks:
                logger.warning(f"Primary model {model_string} failed, trying fallbacks")
                return self._try_fallbacks(api_params, str(e))
            raise

    @get_configured_retry_decorator()
    def _call_primary(
        self, api_params: dict[str, Any], model_string: str, prompt: str
    ) -> str:
        """Execute a single LiteLLM completion call (retried by decorator).

        Args:
            api_params: Parameters dict for ``litellm.completion()``.
            model_string: Model identifier for logging.
            prompt: Original user prompt (used for history).

        Returns:
            The assistant response text.

        Raises:
            Exception: Re-raised from ``completion()`` after retries exhausted.

        """
        response = completion(**api_params)
        return self._record_response(prompt, response, model_string)

    def _try_fallbacks(
        self,
        original_params: dict[str, Any],
        original_error: str,
    ) -> str:
        """Try fallback models if primary fails."""
        for fallback_model in self._fallbacks:
            try:
                logger.info(f"Trying fallback model: {fallback_model}")
                params = original_params.copy()
                params["model"] = fallback_model
                response = completion(**params)
                return self._record_fallback_response(response, fallback_model)
            except Exception as e:
                logger.warning(f"Fallback model {fallback_model} failed: {e}")
                continue

        raise RuntimeError(f"All models failed. Primary error: {original_error}")

    def clone(self) -> FFLiteLLMClient:
        """Create a fresh clone of this client with empty history.

        Used for thread-safe parallel execution where each thread
        needs an isolated client instance with the same configuration.

        Returns:
            New FFLiteLLMClient with same config, empty history.

        """
        logger.debug(f"Cloning client with model_string={self._model_string}")
        clone = FFLiteLLMClient(
            model_string=self._model_string,
            config=copy.deepcopy(self._config),
            api_key=self.api_key,
            api_base=self.api_base,
            api_version=self.api_version,
            system_instructions=self.system_instructions,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            fallbacks=copy.copy(self._fallbacks) if self._fallbacks else None,
            retry_config=copy.copy(self._retry_config),
            **copy.deepcopy(self._extra_kwargs),
        )
        clone._reset_usage()
        return clone
