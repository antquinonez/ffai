"""Registry and factory for vector store backends."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from .base import VectorStoreBase

logger = logging.getLogger(__name__)

_BACKENDS: dict[str, str] = {
    "chroma": "ffai.rag.stores.chroma",
    "pgvector": "ffai.rag.stores.pgvector",
    "qdrant": "ffai.rag.stores.qdrant",
    "sqlite_vss": "ffai.rag.stores.sqlite_vss",
}

STORE_REGISTRY: dict[str, type[VectorStoreBase]] = {}


def _register_backend(name: str, module_path: str) -> None:
    if name in STORE_REGISTRY:
        return
    try:
        mod = importlib.import_module(module_path)
        STORE_REGISTRY[name] = mod.get_store_class()
    except ImportError:
        logger.debug(f"Vector store backend '{name}' not available: {module_path} import failed")


def _ensure_registered(name: str) -> None:
    if name not in STORE_REGISTRY and name in _BACKENDS:
        _register_backend(name, _BACKENDS[name])


def get_store(backend: str = "chroma", **kwargs: Any) -> VectorStoreBase:
    """Get a vector store instance by backend name.

    Args:
        backend: Backend name (``"chroma"``, ``"pgvector"``,
            ``"qdrant"``, ``"sqlite_vss"``).
        **kwargs: Backend-specific constructor arguments.

    Returns:
        Configured vector store instance.

    Raises:
        ValueError: If backend name is not recognized.
        ImportError: If the backend's dependency is not installed.
    """
    name = backend.lower()
    _ensure_registered(name)

    if name not in STORE_REGISTRY:
        if name in _BACKENDS:
            raise ImportError(
                f"Vector store backend '{backend}' is known but its "
                f"dependency is not installed. Install the required "
                f"package to use this backend."
            )
        available = [n for n in _BACKENDS if is_store_available(n)]
        raise ValueError(
            f"Unknown vector store backend: '{backend}'. "
            f"Available: {', '.join(available) if available else 'none (install a backend)'}"
        )

    return STORE_REGISTRY[name](**kwargs)


def list_stores() -> list[str]:
    """List all known backend names (regardless of availability)."""
    return list(_BACKENDS.keys())


def list_available_stores() -> list[str]:
    """List backend names whose dependencies are installed."""
    return [name for name in _BACKENDS if is_store_available(name)]


def is_store_available(name: str) -> bool:
    """Check if a backend's dependencies are installed."""
    _ensure_registered(name)
    return name in STORE_REGISTRY


__all__ = [
    "STORE_REGISTRY",
    "VectorStoreBase",
    "get_store",
    "is_store_available",
    "list_available_stores",
    "list_stores",
]
