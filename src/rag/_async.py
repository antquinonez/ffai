"""Run async coroutines from synchronous contexts, handling both no-running-loop and already-running-loop cases."""

from __future__ import annotations

import asyncio
import concurrent.futures
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from synchronous code.

    If no event loop is running, delegates to :func:`asyncio.run`.
    If an event loop is already running (e.g. inside a Jupyter notebook
    or an existing async framework), runs the coroutine in a background
    thread to avoid ``RuntimeError``.

    Args:
        coro: The coroutine to execute.

    Returns:
        The coroutine's result.

    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()
