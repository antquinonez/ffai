"""Backward-compatibility shim. run_sync has moved to ffai.core._async."""

from __future__ import annotations

from ..core._async import run_sync

__all__ = ["run_sync"]
