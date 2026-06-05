from __future__ import annotations

import asyncio

from ffai.rag._async import run_sync


async def _return_value(value: int) -> int:
    return value


async def _raise_value_error() -> None:
    raise ValueError("test error")


class TestRunSync:
    def test_returns_coroutine_result(self):
        result = run_sync(_return_value(42))
        assert result == 42

    def test_no_running_loop_uses_asyncio_run(self):
        result = run_sync(_return_value(7))
        assert result == 7

    def test_already_running_loop_uses_thread(self):
        async def async_caller():
            return run_sync(_return_value(99))

        result = asyncio.run(async_caller())
        assert result == 99

    def test_propagates_exception_from_coroutine(self):
        try:
            run_sync(_raise_value_error())
            raise AssertionError("Expected ValueError")
        except ValueError as e:
            assert "test error" in str(e)
