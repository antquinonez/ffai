from __future__ import annotations

import logging
import os

from ..config import ClientTypeConfig, get_config
from ..core.client_base import FFAIClientBase
from .spec import ClientRef

logger = logging.getLogger(__name__)


class ClientFactory:
    def __init__(
        self,
        ffai_client: FFAIClientBase,
        workflow_clients: dict[str, ClientRef] | None = None,
        async_mode: bool = True,
    ) -> None:
        self._ffai_client = ffai_client
        self._workflow_clients = workflow_clients or {}
        self._async_mode = async_mode
        self._cache: dict[str, FFAIClientBase] = {}

    def resolve(self, ref: ClientRef | None) -> FFAIClientBase:
        if ref is None:
            return self._ffai_client

        if ref.is_named_ref:
            return self._resolve_named(ref.name)  # type: ignore[arg-type]

        return self._create_inline(ref)

    def _resolve_named(self, name: str) -> FFAIClientBase:
        if name in self._cache:
            return self._cache[name]

        if name in self._workflow_clients:
            client = self._create_inline(self._workflow_clients[name])
            self._cache[name] = client
            return client

        config = get_config()
        client_config = config.clients.get_client_type(name)
        if client_config is not None:
            client = self._create_from_config(name, client_config)
            self._cache[name] = client
            return client

        logger.warning(
            f"Client '{name}' not found in workflow or config, using default client"
        )
        return self._ffai_client

    def _create_from_config(
        self, name: str, config: ClientTypeConfig
    ) -> FFAIClientBase:
        if config.type == "native":
            return self._create_native_client(name, config)

        api_key = os.environ.get(config.api_key_env, "")
        model_string = (
            f"{config.provider_prefix}{config.default_model}"
            if config.provider_prefix
            else config.default_model
        )

        return self._instantiate_client(
            model_string=model_string,
            api_key=api_key,
            fallbacks=config.fallbacks,
        )

    def _create_native_client(
        self, name: str, config: ClientTypeConfig
    ) -> FFAIClientBase:
        client_class_name = config.client_class
        if not client_class_name:
            raise ValueError(f"Native client '{name}' has no client_class defined")

        import importlib

        module = importlib.import_module("..Clients", __package__)
        client_cls = getattr(module, client_class_name, None)
        if client_cls is None:
            raise ValueError(
                f"Native client class '{client_class_name}' not found in ffai.Clients"
            )

        api_key = os.environ.get(config.api_key_env, "")
        return client_cls(
            model_string=config.default_model,
            api_key=api_key or None,
        )

    def _create_inline(self, ref: ClientRef) -> FFAIClientBase:
        api_key = ""
        if ref.api_key_env:
            api_key = os.environ.get(ref.api_key_env, "")

        model_string = ref.model or ""
        if ref.provider_prefix and model_string:
            model_string = f"{ref.provider_prefix}{model_string}"

        return self._instantiate_client(
            model_string=model_string,
            api_key=api_key,
            fallbacks=ref.fallbacks,
        )

    def _instantiate_client(
        self,
        model_string: str,
        api_key: str = "",
        fallbacks: list[str] | None = None,
    ) -> FFAIClientBase:
        if self._async_mode:
            from ..Clients.AsyncFFLiteLLMClient import AsyncFFLiteLLMClient

            return AsyncFFLiteLLMClient(
                model_string=model_string,
                api_key=api_key or None,
                fallbacks=fallbacks,
            )
        else:
            from ..Clients.FFLiteLLMClient import FFLiteLLMClient

            return FFLiteLLMClient(
                model_string=model_string,
                api_key=api_key or None,
                fallbacks=fallbacks,
            )
