# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Configuration management using pydantic-settings.

Provides centralized configuration loaded from config/*.yaml files with
environment variable overrides and type-safe access.

Config files:
  config/main.yaml          - Retry and observability settings
  config/logging.yaml       - Logging configuration
  config/paths.yaml         - File system paths
  config/clients.yaml       - AI client configurations
  config/model_defaults.yaml - Per-model default parameters
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict


def _find_config_dir() -> Path:
    """Find config directory starting from current directory up to project root."""
    candidates = [
        Path.cwd() / "config",
        Path(__file__).parent.parent / "config",
        Path.cwd().parent / "config",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate
    return Path("config")


def _load_yaml_file(filename: str) -> dict[str, Any]:
    """Load a single YAML file from the config directory."""
    config_dir = _find_config_dir()
    filepath = config_dir / filename
    if not filepath.exists():
        return {}
    with open(filepath, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_all_configs() -> dict[str, Any]:
    """Load all configuration files and merge them."""
    main_yaml = _load_yaml_file("main.yaml")
    return {
        "logging": _load_yaml_file("logging.yaml").get("logging", {}),
        "paths": _load_yaml_file("paths.yaml").get("paths", {}),
        "retry": main_yaml.get("retry", {}),
        "clients": _load_yaml_file("clients.yaml"),
        "model_defaults": _load_yaml_file("model_defaults.yaml").get("model_defaults", {}),
        "observability": main_yaml.get("observability", {}),
        "rag": main_yaml.get("rag", {}),
    }


class YamlConfigSource(PydanticBaseSettingsSource):
    """Custom settings source that reads from merged YAML files."""

    def __init__(self, settings_cls: type[BaseSettings], yaml_data: dict[str, Any]):
        super().__init__(settings_cls)
        self._yaml_data = yaml_data

    def get_field_value(self, field: Any, field_name: str) -> tuple[Any, str, bool]:
        """Look up a field value from the merged YAML data.

        Args:
            field: The pydantic field metadata (unused).
            field_name: The field name to look up.

        Returns:
            A tuple of ``(value, field_name, value_is_complex)``.

        """
        field_value = self._yaml_data.get(field_name)
        return field_value, field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._yaml_data


class LoggingRotationConfig(BaseSettings):
    """Logging rotation configuration."""

    when: str = "midnight"
    interval: int = 1
    backup_count: int = 10


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    directory: str = "logs"
    filename: str = "orchestrator.log"
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    rotation: LoggingRotationConfig = Field(default_factory=LoggingRotationConfig)


class PathsConfig(BaseSettings):
    """Path configuration for FFAI data directory."""

    ffai_data: str = "./ffai_data"


class RetryConfig(BaseSettings):
    """Retry configuration for API calls."""

    max_attempts: int = 3
    min_wait_seconds: float = 1.0
    max_wait_seconds: float = 60.0
    exponential_base: float = 2.0
    exponential_jitter: bool = True
    retry_on_status_codes: list[int] = Field(default_factory=lambda: [429, 503, 502, 504])
    log_level: str = "INFO"


class ClientTypeConfig(BaseSettings):
    """Configuration for a single client type."""

    client_class: str = ""
    type: Literal["native", "litellm"] = "litellm"
    api_key_env: str = ""
    provider_prefix: str = ""
    default_model: str = ""
    fallbacks: list[str] = Field(default_factory=list)


class ClientsConfig(BaseSettings):
    """All clients configuration with client type definitions."""

    model_config = SettingsConfigDict(extra="allow")

    default_client: str = "litellm-mistral-small"
    client_types: dict[str, ClientTypeConfig] = Field(default_factory=dict)

    def get_client_type(self, name: str) -> ClientTypeConfig | None:
        """Get a client type configuration by name."""
        return self.client_types.get(name)

    def get_available_client_types(self) -> list[str]:
        """Get list of available client type names."""
        return list(self.client_types.keys())


class ModelDefaultsConfig(BaseSettings):
    """Model-specific defaults configuration."""

    generic: dict[str, Any] = Field(
        default_factory=lambda: {
            "max_tokens": 4096,
            "temperature": 0.7,
            "system_instructions": "You are a helpful assistant. Respond accurately to user queries.",
        }
    )
    models: dict[str, dict[str, Any]] = Field(default_factory=dict)


class OTelConfig(BaseSettings):
    """OpenTelemetry configuration."""

    service_name: str = "plico"
    endpoint: str = "http://localhost:4317"
    export_traces: bool = True
    insecure: bool = True


class ObservabilityConfig(BaseSettings):
    """Observability configuration for token tracking, cost, and telemetry."""

    enabled: bool = False
    otel: OTelConfig = Field(default_factory=OTelConfig)
    token_tracking: bool = True
    cost_tracking: bool = True


class RAGConfig(BaseSettings):
    """Retrieval-Augmented Generation configuration.

    Attributes:
        enabled: Whether RAG is active.
        persist_dir: Directory for the ChromaDB vector store.
        collection_name: ChromaDB collection name.
        embedding_model: LiteLLM-style embedding model identifier.
        chunker: Chunking strategy (``"recursive"`` or ``"fixed"``).
        chunk_size: Maximum characters per chunk.
        chunk_overlap: Overlap characters between adjacent chunks.
        bm25_alpha: Hybrid search alpha; ``None`` disables BM25.
        reranker: Reranker model identifier; ``None`` disables reranking.

    """

    enabled: bool = False
    persist_dir: str = "./chroma_db"
    collection_name: str = "ffai_kb"
    embedding_model: str = "mistral/mistral-embed"
    chunker: str = "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    bm25_alpha: float | None = None
    reranker: str | None = None


class Config(BaseSettings):
    """Main configuration class."""

    model_config = SettingsConfigDict(
        extra="ignore",
        validate_default=True,
        env_nested_delimiter="__",
    )

    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    clients: ClientsConfig = Field(default_factory=ClientsConfig)
    model_defaults: ModelDefaultsConfig = Field(default_factory=ModelDefaultsConfig)
    observability: ObservabilityConfig = Field(default_factory=ObservabilityConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,  # noqa: ARG003
        file_secret_settings: PydanticBaseSettingsSource,  # noqa: ARG003
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Define the priority order for configuration sources.

        Priority (highest first): init kwargs, environment variables,
        merged YAML files.

        Args:
            settings_cls: The settings class being configured.
            init_settings: Source for constructor kwargs.
            env_settings: Source for environment variables.
            dotenv_settings: Source for .env files (unused).
            file_secret_settings: Source for file secrets (unused).

        Returns:
            Tuple of settings sources in priority order.

        """
        yaml_data = _load_all_configs()
        yaml_source = YamlConfigSource(settings_cls, yaml_data)
        return (init_settings, env_settings, yaml_source)

    def get_client_type_config(self, name: str) -> ClientTypeConfig | None:
        """Get a client type configuration by name."""
        return self.clients.get_client_type(name)

    def get_default_client_type(self) -> str:
        """Get the default client type name."""
        return self.clients.default_client

    def get_available_client_types(self) -> list[str]:
        """Get list of available client type names."""
        return self.clients.get_available_client_types()

    def get_litellm_prefix(self, client_name: str) -> str:
        """Get LiteLLM provider prefix for a client."""
        client_type_config = self.get_client_type_config(client_name)
        if client_type_config is None:
            return ""
        return client_type_config.provider_prefix


_config: Config | None = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config()
    return _config


def reload_config() -> Config:
    """Reload configuration from files."""
    global _config
    _config = Config()
    return _config
