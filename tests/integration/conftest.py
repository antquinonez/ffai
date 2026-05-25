# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

import os
from pathlib import Path

import pytest
import yaml

from src.Clients.FFLiteLLMClient import FFLiteLLMClient
from src.Clients.FFMistralSmall import FFMistralSmall

CLIENT_CLASSES = {
    "FFMistralSmall": FFMistralSmall,
    "FFLiteLLMClient": FFLiteLLMClient,
}

_config_path = Path(__file__).parent / "test_config.yaml"
with open(_config_path) as f:
    _config = yaml.safe_load(f)

_enabled = {
    name: cfg
    for name, cfg in _config.get("clients", {}).items()
    if cfg.get("enabled", False)
}


def _build_client(name: str, cfg: dict):
    api_key = os.getenv(cfg["api_key_env"])
    if not api_key:
        pytest.skip(f"{cfg['api_key_env']} not set (required for {name})")
    cls = CLIENT_CLASSES[cfg["client_class"]]
    params = dict(cfg.get("params", {}))
    params["api_key"] = api_key
    return cls(**params)


@pytest.fixture(
    params=list(_enabled.keys()),
    ids=list(_enabled.keys()),
)
def integration_client(request):
    return _build_client(request.param, _enabled[request.param])


def _first_of_class(class_name: str):
    for name, cfg in _enabled.items():
        if cfg["client_class"] == class_name:
            return _build_client(name, cfg)
    pytest.skip(f"No {class_name} client enabled in test_config.yaml")


@pytest.fixture
def ffmistralsmall_client():
    return _first_of_class("FFMistralSmall")
