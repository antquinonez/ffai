# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Declarative context handling API wrapper for AI clients.

This module provides the FFAI class which wraps AI client implementations
and adds declarative context management, history tracking, and DataFrame
export capabilities.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
from typing import Any

from .config import get_config
from .core.client_base import FFAIClientBase
from .core.conversation_manager import ConversationManager
from .core.history.ordered import OrderedPromptHistory
from .core.history.permanent import PermanentHistory
from .core.history.recorder import HistoryRecorder
from .core.history_exporter import HistoryExporter
from .core.history_manager import HistoryManager
from .core.memory import Memory
from .core.prompt_utils import extract_json_field, interpolate_prompt
from .core.response_context import ResponseContext
from .core.response_utils import clean_response as _clean_response_impl
from .core.response_utils import extract_json
from .core.workflow_engine import WorkflowEngine
from .rag.rag import RAG

__all__ = ["FFAI", "extract_json_field", "interpolate_prompt"]

logger = logging.getLogger(__name__)


class FFAI:
    """Declarative context handling wrapper for AI clients.

    This class wraps an AI client implementation and exposes three
    namespaced managers:

    - ``workflow`` — execution orchestration (generate, DAG, workflows)
    - ``history`` — interaction queries, DataFrame export, persistence
    - ``rag`` — retrieval-augmented generation (query, index, search)

    Attributes:
        client: The underlying AI client instance.
        workflow: ``WorkflowEngine`` for execution orchestration.
        history: ``HistoryManager`` for history queries and export.
            ``history.memory`` exposes the ``Memory`` instance when
            memory is enabled (else ``None``).
            ``history.search()`` performs semantic recall.
        rag: ``RAG`` instance, or ``None`` if not configured.
        persist_dir: Directory for history persistence files.
        persist_name: Filename stem for persisted files.
        auto_persist: Whether histories auto-persist after each call.

    Args:
        client: AI client to wrap.
        persist_dir: Directory for history persistence files.
        persist_name: Filename stem for persisted files.
        auto_persist: Whether to auto-persist histories after each call.
        shared_prompt_attr_history: Optional shared prompt-attr list.
        history_lock: Optional thread lock for history access.
        rag: Optional pre-constructed ``RAG`` instance.
        memory_enabled: Override ``config.memory.enabled``. ``None``
            falls through to config (default ``False``).
        memory_embeddings: Override ``config.memory.embedding_model``.
            ``None`` falls through to config; triggers the resolution
            ladder if config is also ``None``.
        memory_persist: Override ``config.memory.persist``. ``None``
            falls through to config (default ``False``).

    """

    def __init__(
        self,
        client: FFAIClientBase,
        persist_dir: str | None = None,
        persist_name: str | None = None,
        auto_persist: bool = False,
        shared_prompt_attr_history: list[dict[str, Any]] | None = None,
        history_lock: threading.Lock | None = None,
        rag: RAG | None = None,
        memory_enabled: bool | None = None,
        memory_embeddings: str | None = None,
        memory_persist: bool | None = None,
        memory: Memory | None = None,
    ) -> None:
        logger.info("Initializing FFAI wrapper")

        config = get_config()
        self.persist_dir = persist_dir if persist_dir is not None else config.paths.ffai_data
        self.persist_name = persist_name
        self.auto_persist = auto_persist
        os.makedirs(self.persist_dir, exist_ok=True)

        self.client = client
        self._rag: RAG | None = None
        self.rag = rag if rag is not None else (RAG.from_config() if config.rag.enabled else None)
        self._conversation = ConversationManager(client=client)

        self._context = ResponseContext(
            shared_prompt_attr_history=shared_prompt_attr_history,
            history_lock=history_lock,
        )

        self._permanent = PermanentHistory()
        self._ordered = OrderedPromptHistory()

        if memory is not None:
            self._memory: Memory | None = memory
            memory_cfg_persist = bool(memory_persist) if memory_persist is not None else False
            memory_persist_path: str | None = None
            if memory_cfg_persist:
                memory_persist_path = os.path.join(
                    config.memory.persist_dir, f"{config.memory.collection_name}.parquet"
                )
        else:
            memory_cfg_enabled = (
                config.memory.enabled if memory_enabled is None else memory_enabled
            )
            memory_cfg_persist = (
                config.memory.persist if memory_persist is None else memory_persist
            )
            memory_cfg_embeddings = (
                config.memory.embedding_model
                if memory_embeddings is None
                else memory_embeddings
            )
            self._memory = None
            memory_persist_path = None
            if memory_cfg_enabled:
                self._memory = self._construct_memory(
                    embedding_model=memory_cfg_embeddings,
                    persist=memory_cfg_persist,
                    persist_dir=config.memory.persist_dir,
                    collection_name=config.memory.collection_name,
                )
                if memory_cfg_persist and self._memory is not None:
                    memory_persist_path = os.path.join(
                        config.memory.persist_dir, f"{config.memory.collection_name}.parquet"
                    )

        self._recorder = HistoryRecorder(
            context=self._context,
            permanent_history=self._permanent,
            ordered_history=self._ordered,
            memory=self._memory,
            memory_persist=memory_cfg_persist,
            memory_persist_path=memory_persist_path,
        )

        self._exporter = HistoryExporter(
            history=self._recorder.history,
            clean_history=self._recorder.clean_history,
            prompt_attr_history=self._context.prompt_attr_history,
            ordered_history=self._ordered,
            persist_dir=self.persist_dir,
            persist_name=self.persist_name,
            auto_persist=self.auto_persist,
        )

        self.history = HistoryManager(
            recorder=self._recorder,
            context=self._context,
            permanent=self._permanent,
            ordered=self._ordered,
            exporter=self._exporter,
            memory=self._memory,
        )
        self.workflow = WorkflowEngine(
            client=self.client,
            conversation=self._conversation,
            recorder=self._recorder,
            prompt_attr_history=self._context.prompt_attr_history,
            clean_response_fn=_clean_response_impl,
        )

    @staticmethod
    def _construct_memory(
        embedding_model: str | None,
        persist: bool,
        persist_dir: str,
        collection_name: str,
    ) -> Memory | None:
        """Resolve an embedding backend and construct a ``Memory`` instance.

        Implements the embedding backend resolution ladder:

        1. If ``embedding_model`` is explicit, try to construct with it.
        2. Else try ``local/all-MiniLM-L6-v2`` (requires fastembed or
           sentence-transformers).
        3. Else fall back to API embeddings keyed off
           ``MISTRAL_API_KEY`` / ``OPENAI_API_KEY``.
        4. Else log a warning and return ``None`` (memory disabled).

        When ``persist`` is ``True``, loads the store from
        ``<persist_dir>/<collection_name>.parquet`` if it exists.

        Returns ``None`` when no backend is available.

        """
        from .core.embeddings import Embeddings
        from .core.memory import Memory, TurnVectorStore, load_store

        embeddings: Embeddings | None = None
        if embedding_model:
            try:
                embeddings = Embeddings(model=embedding_model)
            except ImportError as exc:
                logger.warning(
                    "Configured memory.embedding_model %r unavailable: %s",
                    embedding_model,
                    exc,
                )
                return None
        else:
            with contextlib.suppress(ImportError):
                embeddings = Embeddings(model="local/all-MiniLM-L6-v2")
            if embeddings is None and os.getenv("MISTRAL_API_KEY"):
                embeddings = Embeddings(model="mistral/mistral-embed")
            if embeddings is None and os.getenv("OPENAI_API_KEY"):
                embeddings = Embeddings(model="openai/text-embedding-3-small")

        if embeddings is None:
            logger.warning(
                "Memory enabled but no embedding backend available. "
                "Install ffai[memory] (or ffai[rag]) for local embeddings, "
                "or set memory.embedding_model / MISTRAL_API_KEY / OPENAI_API_KEY."
            )
            return None

        store: TurnVectorStore | None = None
        if persist:
            persist_path = os.path.join(persist_dir, f"{collection_name}.parquet")
            try:
                store = load_store(persist_path)
                logger.info("Loaded %d memory turns from %s", store.count(), persist_path)
            except FileNotFoundError:
                store = None

        return Memory(embeddings, store=store)

    @property
    def rag(self) -> RAG | None:
        """RAG instance for retrieval-augmented generation.

        Setting this property automatically wires the current client
        as the RAG's default ``generate_fn`` via ``ClientAdapter``.
        """
        return self._rag

    @rag.setter
    def rag(self, value: RAG | None) -> None:
        self._rag = value
        if value is not None:
            from .rag import ClientAdapter

            value.set_generate_fn(ClientAdapter(self.client))

    def set_client(self, client: FFAIClientBase) -> None:
        """Switch to a different AI client."""
        logger.info(f"Switching client to {client.__class__.__name__}")
        self.client = client
        self._conversation.client = client
        self.workflow.client = client
        if self._rag is not None:
            from .rag import ClientAdapter

            self._rag.set_generate_fn(ClientAdapter(client))

    def _extract_json(self, text: str) -> Any | None:
        return extract_json(text)

    def get_system_instructions(self) -> str | None:
        """Return the system instructions configured on the client, or ``None``."""
        if hasattr(self.client, "system_instructions"):
            return self.client.system_instructions
        return None

    def clear_conversation(self) -> None:
        """Clear conversation in client but retain history."""
        self._conversation.clear()

    def get_client_conversation_history(self) -> list[dict[str, str]]:
        """Get the raw conversation history from the underlying client."""
        return self._conversation.get_history()

    def set_client_conversation_history(self, history: list[dict[str, str]]) -> bool:
        """Set the raw conversation history in the underlying client."""
        return self._conversation.set_history(history)

    def add_client_message(self, role: str, content: str, **kwargs: Any) -> bool:
        """Add a single message to the client's conversation history."""
        try:
            history = self.get_client_conversation_history()
            message = {"role": role, "content": content, **kwargs}
            history.append(message)
            return self._conversation.set_history(history)
        except Exception as e:
            logger.error(f"Error adding message to conversation history: {e!s}")
            return False

    def close(self) -> None:
        """Release background resources.

        Shuts down the memory embed thread pool if one is running. Safe
        to call multiple times. Call this at process teardown to avoid
        leaking daemon threads (e.g., in long-running services).
        """
        if hasattr(self, "_recorder") and self._recorder is not None:
            self._recorder.shutdown()
