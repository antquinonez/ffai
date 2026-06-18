"""Backward-compatibility shim. Embeddings has moved to ffai.core.embeddings."""

from __future__ import annotations

from ..core.embeddings import Embeddings

__all__ = ["Embeddings"]
