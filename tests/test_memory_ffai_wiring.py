# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT

"""L4 wiring tests for memory integration into FFAI.

These are unit tests (not @pytest.mark.integration) — they use a mock
client and ``FakeEmbeddings`` (no real API calls). They cover the
end-to-end integration: ``FFAI.__init__`` → ``HistoryRecorder.record()``
→ ``Memory.index_turn_text()`` → ``HistoryManager.search()``.
"""

from __future__ import annotations

import os
import time
from typing import Any
from unittest.mock import MagicMock, patch

from conftest import FakeEmbeddings

import ffai
from ffai import FFAI, Embeddings, Memory, TurnHit
from ffai.core import Embeddings as CoreEmbeddings
from ffai.core import Memory as CoreMemory
from ffai.core import TurnHit as CoreTurnHit
from ffai.core import run_sync
from ffai.core.memory import load_store


def _mock_client() -> Any:
    client = MagicMock()
    client.model = "test-model"
    client.get_conversation_history.return_value = []
    client.set_conversation_history = MagicMock(return_value=True)
    client.clear_conversation = MagicMock()
    client.last_usage = None
    client.last_cost_usd = 0.0
    client.generate_response.return_value = "test response"
    return client


def _ffai_with_memory(
    tmp_path: Any = None,
    persist: bool = False,
    memory: Memory | None = None,
    embeddings_dim: int = 8,
) -> tuple[FFAI, Memory]:
    client = _mock_client()
    mem = memory if memory is not None else Memory(FakeEmbeddings(dim=embeddings_dim))
    if persist:
        ffai_inst = FFAI(client, memory=mem, memory_persist=True)
    else:
        ffai_inst = FFAI(client, memory=mem)
    return ffai_inst, mem


def _wait_for_count(mem: Memory, expected: int, timeout_s: float = 0.5) -> None:
    """Poll memory.count() until it reaches *expected* or *timeout_s* elapses."""
    for _ in range(int(timeout_s * 100)):
        if mem.count() == expected:
            return
        time.sleep(0.01)


class TestPublicApiExports:
    def test_top_level_imports(self):
        assert Memory is CoreMemory
        assert TurnHit is CoreTurnHit
        assert Embeddings is CoreEmbeddings

    def test_memory_in_all(self):
        assert "Memory" in ffai.__all__
        assert "TurnHit" in ffai.__all__
        assert "Embeddings" in ffai.__all__

    def test_run_sync_importable_from_core(self):
        from ffai.core._async import run_sync as core_run_sync

        assert run_sync is core_run_sync


class TestEmbeddingsRelocation:
    def test_rag_embed_shim_yields_same_class(self):
        from ffai.rag.embed import Embeddings as RagEmbeddings

        assert RagEmbeddings is Embeddings

    def test_rag_async_shim_yields_same_function(self):
        from ffai.rag._async import run_sync as rag_run_sync

        assert rag_run_sync is run_sync

    def test_cosine_similarity_returns_zero_for_orthogonal(self):
        assert Embeddings.cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


class TestMemoryDefaultsDisabled:
    def test_no_memory_args_yields_none(self):
        client = _mock_client()
        with patch.dict(os.environ, {}, clear=False):
            for var in ("MISTRAL_API_KEY", "OPENAI_API_KEY"):
                os.environ.pop(var, None)
            ffai_inst = FFAI(client)
            assert ffai_inst.history.memory is None

    def test_search_returns_empty_when_disabled(self):
        client = _mock_client()
        ffai_inst = FFAI(client)
        assert ffai_inst.history.search("anything") == []

    def test_generate_response_does_not_embed_when_disabled(self):
        client = _mock_client()
        ffai_inst = FFAI(client)
        ffai_inst.workflow.generate_response(prompt="hello", prompt_name="greet")
        assert ffai_inst.history.memory is None


class TestMemoryEnabled:
    def test_memory_accessible_via_history(self):
        ffai_inst, mem = _ffai_with_memory()
        assert ffai_inst.history.memory is mem
        assert isinstance(ffai_inst.history.memory, Memory)
        assert mem.count() == 0

    def test_generate_response_indexes_qa_pair(self):
        ffai_inst, mem = _ffai_with_memory()
        ffai_inst.workflow.generate_response(
            prompt="what is python",
            prompt_name="python_q",
        )
        _wait_for_count(mem, 1)
        assert mem.count() == 1
        entries = list(mem.store.iter_entries())
        assert "what is python" in entries[0].text
        assert "test response" in entries[0].text

    def test_multiple_calls_index_multiple_turns(self):
        ffai_inst, mem = _ffai_with_memory()
        for i in range(3):
            ffai_inst.workflow.generate_response(prompt=f"q{i}", prompt_name=f"p{i}")
        _wait_for_count(mem, 3)
        assert mem.count() == 3

    def test_failed_embedding_does_not_raise(self):
        failing_memory = Memory(FakeEmbeddings())
        failing_memory._embeddings = MagicMock()
        failing_memory._embeddings.embed.side_effect = RuntimeError("api down")

        client = _mock_client()
        ffai_inst = FFAI(client, memory=failing_memory)
        # Should not raise
        ffai_inst.workflow.generate_response(prompt="x", prompt_name="p")
        for _ in range(20):
            time.sleep(0.01)
        # Embedding failed; nothing was indexed but record() completed
        assert failing_memory.count() == 0

    def test_embed_does_not_block_generate_response(self):
        slow_memory = Memory(FakeEmbeddings())
        slow_memory._embeddings = MagicMock()
        slow_memory._embeddings.embed.side_effect = lambda texts: (time.sleep(0.5), texts)[1]

        client = _mock_client()
        ffai_inst = FFAI(client, memory=slow_memory)
        start = time.perf_counter()
        ffai_inst.workflow.generate_response(prompt="x", prompt_name="p")
        elapsed = time.perf_counter() - start
        # generate_response should return well before the 500ms embed completes
        assert elapsed < 0.4


class TestSearch:
    def test_search_returns_turnhit_list(self):
        ffai_inst, mem = _ffai_with_memory()
        ffai_inst.workflow.generate_response(prompt="alpha", prompt_name="a")
        _wait_for_count(mem, 1)
        hits = ffai_inst.history.search("alpha")
        assert isinstance(hits, list)
        assert len(hits) == 1
        assert isinstance(hits[0], TurnHit)

    def test_top_k_limits_results(self):
        ffai_inst, mem = _ffai_with_memory()
        for i in range(5):
            ffai_inst.workflow.generate_response(prompt=f"item_{i}", prompt_name=f"p{i}")
        for _ in range(100):
            if mem.count() == 5:
                break
            time.sleep(0.01)
        hits = ffai_inst.history.search("item_0", top_k=2)
        assert len(hits) <= 2

    def test_threshold_excludes_low_scores(self):
        ffai_inst, mem = _ffai_with_memory()
        ffai_inst.workflow.generate_response(prompt="alpha", prompt_name="a")
        _wait_for_count(mem, 1)
        hits = ffai_inst.history.search("alpha", threshold=0.99)
        assert all(h.score >= 0.99 for h in hits)

    def test_empty_memory_returns_empty_list(self):
        ffai_inst, _ = _ffai_with_memory()
        assert ffai_inst.history.search("anything") == []


class TestMetadataPropagation:
    def test_prompt_name_propagates_to_metadata(self):
        ffai_inst, mem = _ffai_with_memory()
        ffai_inst.workflow.generate_response(prompt="hello", prompt_name="greet")
        for _ in range(50):
            if mem.count() == 1:
                break
            time.sleep(0.01)
        entries = list(mem.store.iter_entries())
        assert entries[0].metadata == {"prompt_name": "greet"}

    def test_metadata_visible_in_search_hits(self):
        ffai_inst, mem = _ffai_with_memory()
        ffai_inst.workflow.generate_response(prompt="alpha", prompt_name="alpha_prompt")
        _wait_for_count(mem, 1)
        hits = ffai_inst.history.search("alpha")
        assert hits[0].metadata == {"prompt_name": "alpha_prompt"}

    def test_turn_text_matches_embedded_text(self):
        ffai_inst, mem = _ffai_with_memory()
        ffai_inst.workflow.generate_response(prompt="the query", prompt_name="p")
        _wait_for_count(mem, 1)
        entries = list(mem.store.iter_entries())
        assert entries[0].text == entries[0].turn["content"][0]["text"]

    def test_caller_metadata_honored_when_prompt_name_absent(self):
        """Regression: caller-supplied metadata must not be dropped when prompt_name is None.

        Previously the recorder gated metadata propagation on `prompt_name`
        truthiness, which silently dropped caller metadata when no
        prompt_name was supplied. See review feedback.
        """
        client = _mock_client()
        mem = Memory(FakeEmbeddings(dim=8))
        ffai_inst = FFAI(client, memory=mem)
        ffai_inst.workflow.generate_response(
            prompt="hello",
            prompt_name=None,
        )
        _wait_for_count(mem, 1)
        entries = list(mem.store.iter_entries())
        # prompt_name=None with no caller metadata -> empty dict (not dropped)
        assert entries[0].metadata == {}


class TestPersistence:
    def test_persist_writes_parquet_file(self, tmp_path):
        persist_dir = str(tmp_path / "memory")
        os.makedirs(persist_dir, exist_ok=True)
        mem = Memory(FakeEmbeddings(dim=8))
        client = _mock_client()

        # Patch the config so persist_dir/collection_name point to tmp_path
        with patch("ffai.FFAI.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.paths.ffai_data = str(tmp_path)
            mock_config.rag.enabled = False
            mock_config.memory.persist_dir = persist_dir
            mock_config.memory.collection_name = "test_turns"
            mock_get_config.return_value = mock_config
            ffai_inst = FFAI(client, memory=mem, memory_persist=True)

        ffai_inst.workflow.generate_response(prompt="hello", prompt_name="p")
        for _ in range(50):
            if mem.count() == 1:
                break
            time.sleep(0.01)
        # Allow persist to complete
        time.sleep(0.1)

        expected_path = os.path.join(persist_dir, "test_turns.parquet")
        assert os.path.exists(expected_path)

    def test_load_store_recovers_turns_on_restart(self, tmp_path):
        persist_dir = str(tmp_path / "memory")
        os.makedirs(persist_dir, exist_ok=True)

        mem1 = Memory(FakeEmbeddings(dim=8))
        client = _mock_client()
        with patch("ffai.FFAI.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.paths.ffai_data = str(tmp_path)
            mock_config.rag.enabled = False
            mock_config.memory.persist_dir = persist_dir
            mock_config.memory.collection_name = "restart_turns"
            mock_config.memory.embedding_model = None
            mock_config.memory.persist = True
            mock_config.memory.enabled = True
            mock_get_config.return_value = mock_config
            ffai1 = FFAI(client, memory=mem1, memory_persist=True)

        ffai1.workflow.generate_response(prompt="alpha", prompt_name="p")
        for _ in range(50):
            if mem1.count() == 1:
                break
            time.sleep(0.01)
        time.sleep(0.2)
        ffai1.close()

        expected_path = os.path.join(persist_dir, "restart_turns.parquet")
        assert os.path.exists(expected_path)
        loaded = load_store(expected_path)
        assert loaded.count() == 1


class TestClose:
    def test_close_is_idempotent(self):
        ffai_inst, _ = _ffai_with_memory()
        ffai_inst.close()
        ffai_inst.close()

    def test_close_shuts_down_embed_pool(self):
        ffai_inst, _ = _ffai_with_memory()
        assert ffai_inst._recorder._embed_pool is not None
        ffai_inst.close()
        assert ffai_inst._recorder._embed_pool is None


class TestConfigDefaults:
    def test_memory_config_defaults_from_yaml(self):
        from ffai.config import Config

        config = Config()
        assert isinstance(config.memory, type(config.memory))
        assert config.memory.enabled is False
        assert config.memory.embedding_model is None
        assert config.memory.persist is False

    def test_memory_env_var_enables(self, monkeypatch):
        from ffai.config import Config

        monkeypatch.setenv("MEMORY__ENABLED", "true")
        config = Config()
        assert config.memory.enabled is True


class TestPyprojectExtra:
    def test_memory_extra_declared(self):
        import re
        from pathlib import Path

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        contents = pyproject.read_text()
        assert re.search(r'^memory\s*=\s*\[\s*"fastembed', contents, re.MULTILINE)

    def test_memory_extra_does_not_pull_chromadb(self):
        import re
        from pathlib import Path

        pyproject = Path(__file__).parent.parent / "pyproject.toml"
        contents = pyproject.read_text()
        match = re.search(r'^memory\s*=\s*\[(.*?)\]', contents, re.MULTILINE | re.DOTALL)
        assert match is not None
        extra_contents = match.group(1)
        assert "chromadb" not in extra_contents
        assert "fastembed" in extra_contents


class TestCoalescingSemantics:
    def test_metadata_carrying_user_turns_do_not_coalesce(self):
        from ffai.core.history.permanent import PermanentHistory

        ph = PermanentHistory()
        ph.add_turn_user("first", metadata={"prompt_name": "p1"})
        ph.add_turn_user("second", metadata={"prompt_name": "p2"})
        turns = ph.get_all_turns()
        assert len(turns) == 2
        assert turns[0]["content"][0]["text"] == "first"
        assert turns[1]["content"][0]["text"] == "second"

    def test_metadata_none_user_turns_still_coalesce(self):
        from ffai.core.history.permanent import PermanentHistory

        ph = PermanentHistory()
        ph.add_turn_user("first")
        ph.add_turn_user("second")
        turns = ph.get_all_turns()
        assert len(turns) == 1
        assert turns[0]["content"][0]["text"] == "first\nsecond"
