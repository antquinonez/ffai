# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Comprehensive tests for configuration system.

Tests cover:
- 12-factor precedence: init > env > yaml > pydantic defaults
- Singleton behavior (get_config, reload_config)
- Type safety and validation
- All config sections loading from YAML
- Client helper methods
"""

from __future__ import annotations

import pytest
from pydantic_core import ValidationError

from src.config import Config, RetryConfig, get_config, reload_config


class TestConfigPrecedence:
    """Test configuration precedence follows 12-factor methodology.

    Priority order (highest first):
    1. Init arguments
    2. Environment variables
    3. YAML files
    4. Pydantic defaults
    """

    def test_env_overrides_yaml(self, monkeypatch):
        """Env var should override YAML value."""
        monkeypatch.setenv("RETRY__MAX_ATTEMPTS", "7")
        config = reload_config()
        assert config.retry.max_attempts == 7

    def test_init_overrides_env(self, monkeypatch):
        """Init args should override env var."""
        monkeypatch.setenv("RETRY__MAX_ATTEMPTS", "7")
        config = Config(retry=RetryConfig(max_attempts=9))
        assert config.retry.max_attempts == 9

    def test_yaml_used_when_no_env_override(self):
        """YAML value used when no env override set."""
        config = reload_config()
        assert config.retry.max_attempts == 3

    def test_float_env_override(self, monkeypatch):
        """Float values can be overridden via env."""
        monkeypatch.setenv("RETRY__MIN_WAIT_SECONDS", "2.5")
        config = reload_config()
        assert config.retry.min_wait_seconds == 2.5

    def test_list_env_override(self, monkeypatch):
        """List values can be overridden via env (JSON-like)."""
        monkeypatch.setenv("RETRY__RETRY_ON_STATUS_CODES", "[429, 503]")
        config = reload_config()
        assert 429 in config.retry.retry_on_status_codes
        assert 503 in config.retry.retry_on_status_codes

    def test_pydantic_default_as_fallback(self, monkeypatch, tmp_path):
        """Pydantic defaults used when YAML missing and no env."""
        config = Config()
        assert config.logging.level == "INFO"

    def test_bool_env_override(self, monkeypatch):
        """Boolean values can be overridden via env."""
        monkeypatch.setenv("OBSERVABILITY__ENABLED", "true")
        config = reload_config()
        assert config.observability.enabled is True

    def test_clear_env_allows_yaml_to_take_effect(self, monkeypatch):
        """Clearing env var allows YAML value to be used."""
        monkeypatch.setenv("RETRY__MAX_ATTEMPTS", "7")
        config = reload_config()
        assert config.retry.max_attempts == 7

        monkeypatch.delenv("RETRY__MAX_ATTEMPTS")
        config = reload_config()
        assert config.retry.max_attempts == 3


class TestConfigSingleton:
    """Test singleton behavior of get_config and reload_config."""

    def test_get_config_returns_singleton(self):
        """get_config should return the same instance."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_reload_config_creates_new_instance(self):
        """reload_config should create a new instance."""
        config1 = get_config()
        config2 = reload_config()
        assert config1 is not config2

    def test_reload_config_updates_global(self):
        """reload_config should update the global singleton."""
        config1 = get_config()
        config2 = reload_config()
        config3 = get_config()
        assert config1 is not config3
        assert config2 is config3


class TestConfigTypeSafety:
    """Test type coercion and validation."""

    def test_int_coercion_from_env(self, monkeypatch):
        """String env vars are coerced to int."""
        monkeypatch.setenv("RETRY__MAX_WAIT_SECONDS", "120")
        config = reload_config()
        assert config.retry.max_wait_seconds == 120
        assert isinstance(config.retry.max_wait_seconds, (int, float))

    def test_float_coercion_from_env(self, monkeypatch):
        """String env vars are coerced to float."""
        monkeypatch.setenv("RETRY__MIN_WAIT_SECONDS", "0.95")
        config = reload_config()
        assert config.retry.min_wait_seconds == 0.95
        assert isinstance(config.retry.min_wait_seconds, float)

    def test_bool_coercion_from_env(self, monkeypatch):
        """String env vars are coerced to bool."""
        monkeypatch.setenv("OBSERVABILITY__COST_TRACKING", "true")
        config = reload_config()
        assert config.observability.cost_tracking is True

        monkeypatch.setenv("OBSERVABILITY__COST_TRACKING", "false")
        config = reload_config()
        assert config.observability.cost_tracking is False

    def test_invalid_type_raises_error(self):
        """Invalid types should raise validation error."""
        with pytest.raises(ValidationError):
            Config(retry={"max_attempts": "not_a_number"})  # type: ignore[arg-type]


class TestConfigSections:
    """Test all config sections load correctly from YAML."""

    def test_logging_section(self):
        """Logging section loads from logging.yaml."""
        config = get_config()
        assert hasattr(config, "logging")
        assert config.logging.directory == "logs"
        assert config.logging.filename == "orchestrator.log"
        assert config.logging.level == "INFO"
        assert hasattr(config.logging, "rotation")
        assert config.logging.rotation.when == "midnight"
        assert config.logging.rotation.backup_count == 10

    def test_paths_section(self):
        """Paths section loads from paths.yaml."""
        config = get_config()
        assert hasattr(config, "paths")
        assert config.paths.ffai_data == "./ffai_data"

    def test_clients_section(self):
        """Clients section loads from clients.yaml."""
        config = get_config()
        assert hasattr(config, "clients")
        assert config.clients.default_client == "litellm-mistral-small"
        assert hasattr(config.clients, "client_types")

    def test_model_defaults_section(self):
        """Model defaults section loads from model_defaults.yaml."""
        config = get_config()
        assert hasattr(config, "model_defaults")
        assert hasattr(config.model_defaults, "generic")
        assert config.model_defaults.generic.get("max_tokens") == 4096


class TestConfigClientMethods:
    """Test client-related helper methods."""

    def test_get_client_type_config(self):
        """get_client_type_config returns correct config."""
        config = get_config()
        client_config = config.get_client_type_config("litellm-mistral")
        assert client_config is not None
        assert client_config.type == "litellm"
        assert client_config.provider_prefix == "mistral/"

    def test_get_client_type_config_unknown_returns_none(self):
        """get_client_type_config returns None for unknown client."""
        config = get_config()
        client_config = config.get_client_type_config("nonexistent-client")
        assert client_config is None

    def test_get_default_client_type(self):
        """get_default_client_type returns default client name."""
        config = get_config()
        default = config.get_default_client_type()
        assert default == "litellm-mistral-small"

    def test_get_available_client_types(self):
        """get_available_client_types returns list of known client type names."""
        config = get_config()
        clients = config.get_available_client_types()
        assert len(clients) > 0
        assert "litellm-mistral" in clients
        assert "litellm-anthropic" in clients

    def test_get_litellm_prefix(self):
        """get_litellm_prefix returns correct prefix."""
        config = get_config()
        prefix = config.get_litellm_prefix("litellm-mistral")
        assert prefix == "mistral/"

    def test_get_litellm_prefix_unknown_returns_empty(self):
        """get_litellm_prefix returns empty string for unknown."""
        config = get_config()
        prefix = config.get_litellm_prefix("nonexistent-client")
        assert prefix == ""


class TestConfigEnvFormat:
    """Test environment variable format for nested config."""

    def test_double_underscore_separator(self, monkeypatch):
        """Double underscore accesses nested values."""
        monkeypatch.setenv("RETRY__MIN_WAIT_SECONDS", "5")
        config = reload_config()
        assert config.retry.min_wait_seconds == 5.0

    def test_triple_nested_path(self, monkeypatch):
        """Triple nested config works."""
        monkeypatch.setenv("OBSERVABILITY__OTEL__ENDPOINT", "http://custom:4317")
        config = reload_config()
        assert config.observability.otel.endpoint == "http://custom:4317"

    def test_case_insensitive_env_vars(self, monkeypatch):
        """Environment variable names are case-insensitive (handled by pydantic)."""
        monkeypatch.setenv("retry__max_attempts", "8")
        config = reload_config()
        assert config.retry.max_attempts == 8


class TestConfigExtraFields:
    """Test handling of extra/unknown fields."""

    def test_extra_fields_ignored_on_init(self):
        """Extra fields passed to Config are silently ignored, not rejected."""
        config = Config(retry=RetryConfig(max_attempts=3), unknown_field="should_be_ignored")  # type: ignore[callArg]
        assert config.retry.max_attempts == 3
        assert not hasattr(config, "unknown_field")


class TestRetryConfig:
    """Test retry configuration values loaded from main.yaml."""

    def test_retry_section_values(self):
        config = reload_config()
        assert config.retry.max_attempts == 3
        assert config.retry.min_wait_seconds == 1.0
        assert config.retry.max_wait_seconds == 60.0
        assert config.retry.exponential_base == 2.0
        assert config.retry.exponential_jitter is True
        assert config.retry.retry_on_status_codes == [429, 503, 502, 504]
        assert config.retry.log_level == "INFO"


class TestObservabilityConfig:
    """Test observability configuration values."""

    def test_observability_section_values(self):
        config = get_config()
        assert config.observability.enabled is False
        assert config.observability.otel.service_name == "plico"
        assert config.observability.otel.endpoint == "http://localhost:4317"
        assert config.observability.otel.export_traces is True
        assert config.observability.otel.insecure is True
        assert config.observability.token_tracking is True
        assert config.observability.cost_tracking is True


class TestClientsConfigMethods:
    """Test ClientsConfig active methods."""

    def test_get_client_type_returns_config(self):
        from src.config import ClientsConfig, ClientTypeConfig

        cc = ClientsConfig(
            client_types={
                "custom": ClientTypeConfig(
                    client_class="FFCustom",
                    type="native",
                    api_key_env="CUSTOM_KEY",
                    provider_prefix="custom/",
                    default_model="custom-v1",
                )
            }
        )
        result = cc.get_client_type("custom")
        assert result is not None
        assert result.default_model == "custom-v1"
        assert result.provider_prefix == "custom/"

    def test_get_client_type_returns_none_for_missing(self):
        from src.config import ClientsConfig

        cc = ClientsConfig()
        assert cc.get_client_type("missing") is None

    def test_get_available_client_types_returns_keys(self):
        from src.config import ClientsConfig, ClientTypeConfig

        cc = ClientsConfig(
            client_types={
                "alpha": ClientTypeConfig(default_model="a"),
                "beta": ClientTypeConfig(default_model="b"),
            }
        )
        names = cc.get_available_client_types()
        assert names == ["alpha", "beta"]


class TestConfigGetYamlHelpers:
    """Test _find_config_dir fallback and _load_yaml_file missing file."""

    def test_find_config_dir_finds_project_config(self):
        from src.config import _find_config_dir

        result = _find_config_dir()
        assert result.name == "config"
        assert result.exists()
        assert (result / "main.yaml").exists()

    def test_load_yaml_file_missing_returns_empty(self, tmp_path, monkeypatch):
        from src.config import _load_yaml_file

        monkeypatch.chdir(tmp_path)
        result = _load_yaml_file("does_not_exist.yaml")
        assert result == {}


class TestConfigModelDefaults:
    """Test model_defaults specific model overrides."""

    def test_model_defaults_contains_known_models(self):
        config = get_config()
        models = config.model_defaults.models
        assert "azure/mistral-small-2503" in models
        assert models["azure/mistral-small-2503"]["max_tokens"] == 40000
        assert models["azure/codestral"]["temperature"] == 0.3

    def test_model_defaults_generic_values(self):
        config = get_config()
        generic = config.model_defaults.generic
        assert generic["max_tokens"] == 4096
        assert generic["temperature"] == 0.7


class TestConfigLoggingRotation:
    """Test logging rotation sub-config."""

    def test_rotation_defaults(self):
        from src.config import LoggingRotationConfig

        r = LoggingRotationConfig()
        assert r.when == "midnight"
        assert r.interval == 1
        assert r.backup_count == 10

    def test_logging_rotation_via_config(self):
        config = get_config()
        assert config.logging.rotation.when == "midnight"
        assert config.logging.rotation.interval == 1
        assert config.logging.rotation.backup_count == 10
