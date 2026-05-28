# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT

"""Tests for observability module: log_context and telemetry."""

from __future__ import annotations

import logging
import threading
from unittest.mock import MagicMock, patch

import pytest

from ffai.observability.log_context import (
    ContextFormatter,
    LogContextFilter,
    _batch_name,
    _prompt_name,
    clear_log_context,
    log_context,
    set_log_context,
)
from ffai.observability.telemetry import (
    NoOpSpan,
    TelemetryManager,
    get_telemetry_manager,
    reload_telemetry,
    reset_telemetry,
)


@pytest.fixture(autouse=True)
def _clean_context():
    clear_log_context()
    yield
    clear_log_context()


@pytest.fixture(autouse=True)
def _reset_telemetry_singleton():
    yield
    reset_telemetry()


class TestNoOpSpan:
    def test_is_recording_returns_false(self):
        span = NoOpSpan()
        assert span.is_recording() is False

    def test_set_attribute_returns_none(self):
        span = NoOpSpan()
        assert span.set_attribute("key", "value") is None

    def test_record_exception_returns_none(self):
        span = NoOpSpan()
        assert span.record_exception(ValueError("test")) is None


class TestSetLogContext:
    def test_set_batch_name(self):
        set_log_context(batch_name="my_batch")
        assert _batch_name.get() == "my_batch"
        assert _prompt_name.get() == "-"

    def test_set_prompt_name(self):
        set_log_context(prompt_name="my_prompt")
        assert _prompt_name.get() == "my_prompt"
        assert _batch_name.get() == "-"

    def test_set_both(self):
        set_log_context(batch_name="b", prompt_name="p")
        assert _batch_name.get() == "b"
        assert _prompt_name.get() == "p"

    def test_set_none_does_not_change(self):
        set_log_context(batch_name="original")
        set_log_context(batch_name=None, prompt_name="new_prompt")
        assert _batch_name.get() == "original"
        assert _prompt_name.get() == "new_prompt"


class TestClearLogContext:
    def test_resets_to_defaults(self):
        set_log_context(batch_name="b", prompt_name="p")
        clear_log_context()
        assert _batch_name.get() == "-"
        assert _prompt_name.get() == "-"


class TestLogContextManager:
    def test_restores_batch_name_on_exit(self):
        set_log_context(batch_name="outer")
        with log_context(batch_name="inner"):
            assert _batch_name.get() == "inner"
        assert _batch_name.get() == "outer"

    def test_restores_prompt_name_on_exit(self):
        set_log_context(prompt_name="outer")
        with log_context(prompt_name="inner"):
            assert _prompt_name.get() == "inner"
        assert _prompt_name.get() == "outer"

    def test_restores_on_exception(self):
        set_log_context(batch_name="before")
        with pytest.raises(RuntimeError):
            with log_context(batch_name="during"):
                raise RuntimeError("boom")
        assert _batch_name.get() == "before"

    def test_partial_override(self):
        set_log_context(batch_name="batch_a", prompt_name="prompt_a")
        with log_context(batch_name="batch_b"):
            assert _batch_name.get() == "batch_b"
            assert _prompt_name.get() == "prompt_a"
        assert _batch_name.get() == "batch_a"
        assert _prompt_name.get() == "prompt_a"

    def test_none_parameters_do_not_set(self):
        set_log_context(batch_name="original")
        with log_context(batch_name=None, prompt_name=None):
            assert _batch_name.get() == "original"
            assert _prompt_name.get() == "-"


class TestLogContextFilter:
    def test_adds_batch_and_prompt_to_record(self):
        set_log_context(batch_name="my_batch", prompt_name="my_prompt")
        f = LogContextFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f.filter(record)
        assert getattr(record, "batch_name", None) == "my_batch"
        assert getattr(record, "prompt_name", None) == "my_prompt"

    def test_returns_true(self):
        f = LogContextFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        assert f.filter(record) is True

    def test_default_values(self):
        f = LogContextFilter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        f.filter(record)
        assert getattr(record, "batch_name", None) == "-"
        assert getattr(record, "prompt_name", None) == "-"


class TestContextFormatter:
    def test_formats_with_context_values(self):
        set_log_context(batch_name="fmt_batch", prompt_name="fmt_prompt")
        fmt = ContextFormatter("[%(batch_name)s|%(prompt_name)s] %(message)s")
        record = logging.LogRecord("test", logging.INFO, "", 0, "hello", (), None)
        output = fmt.format(record)
        assert output == "[fmt_batch|fmt_prompt] hello"

    def test_formats_with_default_values(self):
        fmt = ContextFormatter("[%(batch_name)s|%(prompt_name)s] %(message)s")
        record = logging.LogRecord("test", logging.INFO, "", 0, "msg", (), None)
        output = fmt.format(record)
        assert output == "[-|-] msg"


class TestLogContextThreadSafety:
    def test_context_vars_isolated_across_threads(self):
        results = {}

        def thread_fn(name, barrier):
            barrier.wait()
            set_log_context(batch_name=name)
            barrier.wait()
            results[name] = _batch_name.get()

        barrier = threading.Barrier(2, timeout=5)
        t1 = threading.Thread(target=thread_fn, args=("thread_1", barrier))
        t2 = threading.Thread(target=thread_fn, args=("thread_2", barrier))
        t1.start()
        t2.start()
        t1.join(timeout=5)
        t2.join(timeout=5)

        assert results["thread_1"] == "thread_1"
        assert results["thread_2"] == "thread_2"


class TestTelemetryManagerInit:
    def test_defaults_when_config_disabled(self):
        mgr = TelemetryManager()
        assert mgr.enabled is False
        assert mgr.service_name == "plico"
        assert mgr.endpoint == "http://localhost:4317"

    def test_span_yields_noop_when_disabled(self):
        mgr = TelemetryManager()
        with mgr.span("test.span") as span:
            assert isinstance(span, NoOpSpan)
            assert span.is_recording() is False

    def test_shutdown_safe_when_disabled(self):
        mgr = TelemetryManager()
        mgr.shutdown()
        assert mgr._provider is None

    def test_shutdown_safe_when_no_provider(self):
        mgr = TelemetryManager()
        mgr._provider = None
        mgr.shutdown()
        assert mgr._provider is None


class TestTelemetrySingleton:
    def test_get_telemetry_manager_returns_same_instance(self):
        reset_telemetry()
        m1 = get_telemetry_manager()
        m2 = get_telemetry_manager()
        assert m1 is m2

    def test_reload_creates_new_instance(self):
        reset_telemetry()
        m1 = get_telemetry_manager()
        m2 = reload_telemetry()
        assert m2 is not m1

    def test_reset_clears_singleton(self):
        m1 = get_telemetry_manager()
        reset_telemetry()
        m2 = get_telemetry_manager()
        assert m2 is not m1

    def test_reload_shuts_down_previous(self):
        m1 = get_telemetry_manager()
        m1._provider = MagicMock()
        old_provider = m1._provider
        reload_telemetry()
        old_provider.shutdown.assert_called_once()

    def test_reset_shuts_down_provider(self):
        m1 = get_telemetry_manager()
        m1._provider = MagicMock()
        old_provider = m1._provider
        reset_telemetry()
        old_provider.shutdown.assert_called_once()

    def test_reset_safe_when_none(self):
        reset_telemetry()
        reset_telemetry()
        assert get_telemetry_manager() is not None


class TestTelemetryManagerSpan:
    def test_span_yields_noop_when_tracer_none(self):
        mgr = TelemetryManager()
        mgr._enabled = True
        mgr._tracer = None
        with mgr.span("test") as span:
            assert isinstance(span, NoOpSpan)
            assert span.is_recording() is False

    def test_span_attribute_and_exception_on_noop(self):
        mgr = TelemetryManager()
        with mgr.span("test") as span:
            assert span.is_recording() is False
            assert span.set_attribute("key", "value") is None
            assert span.record_exception(ValueError("err")) is None


class TestTelemetryManagerSetupTracerGenericException:
    def test_setup_tracer_generic_exception_disables(self):
        mock_obs = MagicMock()
        mock_obs.enabled = True
        mock_obs.otel.service_name = "svc"
        mock_obs.otel.endpoint = "http://localhost:4317"
        mock_obs.otel.insecure = True
        mock_config = MagicMock()
        mock_config.observability = mock_obs

        with (
            patch("ffai.config.get_config", return_value=mock_config),
            patch("opentelemetry.trace.set_tracer_provider", side_effect=RuntimeError("broken")),
        ):
            m = TelemetryManager()
        assert m.enabled is False


class TestTelemetryShutdownExceptionHandled:
    def test_shutdown_exception_is_swallowed(self):
        m = TelemetryManager()
        m._provider = MagicMock()
        m._provider.shutdown.side_effect = RuntimeError("failed")
        m.shutdown()
        m._provider.shutdown.assert_called_once()
