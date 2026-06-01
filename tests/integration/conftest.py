# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

import os
import socket
import subprocess
from pathlib import Path

import pytest
import yaml

from ffai.Clients.FFLiteLLMClient import FFLiteLLMClient
from ffai.Clients.FFMistralSmall import FFMistralSmall

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

_vector_stores = {
    name: cfg
    for name, cfg in _config.get("vector_stores", {}).items()
    if cfg.get("enabled", False)
}

_COMPOSE_FILE = str(Path(__file__).parent.parent.parent / "docker-compose.dev.yaml")


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


@pytest.fixture
def fflitellm_client():
    return _first_of_class("FFLiteLLMClient")


# ──────────────────────────────────────────────
# pytest hooks for --qdrant-server flag
# ──────────────────────────────────────────────


def pytest_addoption(parser):
    parser.addoption(
        "--qdrant-server",
        action="store_true",
        default=False,
        help="Auto-start Qdrant Docker container for server-mode tests",
    )


def _qdrant_is_running(host="localhost", port=6333):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    try:
        sock.connect((host, port))
        sock.close()
    except (ConnectionRefusedError, OSError):
        sock.close()
        return False
    try:
        from qdrant_client import QdrantClient  # type: ignore[importMissing]

        c = QdrantClient(host=host, port=port, timeout=5)
        c.get_collections()
        c.close()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def qdrant_server_available(request):
    opt_in = request.config.getoption("--qdrant-server", default=False)

    if _qdrant_is_running():
        yield True
        return

    if not opt_in:
        yield False
        return

    compose_path = Path(_COMPOSE_FILE)
    if not compose_path.exists():
        yield False
        return

    subprocess.run(
        ["docker", "compose", "-f", _COMPOSE_FILE, "up", "-d", "qdrant"],
        check=True,
        capture_output=True,
    )

    import time

    for _ in range(60):
        if _qdrant_is_running():
            yield True
            break
        time.sleep(1)
    else:
        yield False
        return

    subprocess.run(
        ["docker", "compose", "-f", _COMPOSE_FILE, "down"],
        check=True,
        capture_output=True,
    )


@pytest.fixture(scope="session")
def qdrant_cloud_config():
    url = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    api_key = os.getenv("QDRANT_KEY")
    if not url or not api_key:
        pytest.skip(
            "QDRANT_CLUSTER_ENDPOINT and QDRANT_KEY not set. "
            "Set these to run Qdrant cloud mode tests."
        )
    return {"url": url, "api_key": api_key}
