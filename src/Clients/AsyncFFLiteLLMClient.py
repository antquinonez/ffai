# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Async LiteLLM-backed AI client implementing AsyncFFAIClientBase contract.

Mirrors ``FFLiteLLMClient`` but uses ``litellm.acompletion()`` for async
I/O.  Shares all non-I/O logic via ``BaseLiteLLMClient``.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from litellm import acompletion

from ..core.async_client_base import AsyncFFAIClientBase
from ..retry_utils import get_configured_retry_decorator
from .BaseLiteLLMClient import BaseLiteLLMClient

logger = logging.getLogger(__name__)


class AsyncFFLiteLLMClient(BaseLiteLLMClient, AsyncFFAIClientBase):
    """Async LiteLLM-backed AI client implementing AsyncFFAIClientBase.

    Key features:
    - Internal conversation history management
    - Clone pattern for parallel execution
    - Model string routing (e.g., "azure/mistral-small-2503")
    - Retry and fallback support

    Args:
        model_string: LiteLLM model identifier.
        config: Optional configuration dictionary.
        api_key: API key (overrides env var).
        api_base: API base URL (overrides env var).
        system_instructions: System prompt.
        temperature: Sampling temperature (0-2).
        max_tokens: Maximum tokens to generate.
        fallbacks: List of fallback model strings.
        retry_config: Retry configuration.

    """

    async def generate_response(
        self,
        prompt: str,
        model: str | None = None,
        system_instructions: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        api_params, model_string = self._prepare_generate_params(
            prompt, model, system_instructions, temperature, max_tokens, **kwargs
        )

        logger.debug(
            f"Calling LiteLLM async with model={model_string}, temperature={api_params.get('temperature')}"
        )

        try:
            with self._trace_llm_call(model_string):
                return await self._call_primary(api_params, model_string, prompt)
        except Exception as e:
            if self._fallbacks:
                logger.warning(f"Primary model {model_string} failed, trying fallbacks")
                return await self._try_fallbacks(api_params, str(e))
            raise

    @get_configured_retry_decorator()
    async def _call_primary(
        self, api_params: dict[str, Any], model_string: str, prompt: str
    ) -> str:
        response = await acompletion(**api_params)
        return self._record_response(prompt, response, model_string)

    async def _try_fallbacks(
        self,
        original_params: dict[str, Any],
        original_error: str,
    ) -> str:
        for fallback_model in self._fallbacks:
            try:
                logger.info(f"Trying fallback model: {fallback_model}")
                params = original_params.copy()
                params["model"] = fallback_model
                response = await acompletion(**params)
                return self._record_fallback_response(response, fallback_model)
            except Exception as e:
                logger.warning(f"Fallback model {fallback_model} failed: {e}")
                continue

        raise RuntimeError(f"All models failed. Primary error: {original_error}")

    async def clone(self) -> AsyncFFLiteLLMClient:
        logger.debug(f"Cloning async client with model_string={self._model_string}")
        cloned = AsyncFFLiteLLMClient(
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
        cloned._reset_usage()
        return cloned
