# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Single point for writing interactions to all history stores."""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import TYPE_CHECKING, Any

from ..response_context import ResponseContext
from .ordered import OrderedPromptHistory
from .permanent import PermanentHistory

if TYPE_CHECKING:
    from ..memory import Memory

logger = logging.getLogger(__name__)


class HistoryRecorder:
    """Records interactions to all 5 history stores in a single operation.

    Owns the raw ``history`` and ``clean_history`` lists and coordinates
    writes to ``PermanentHistory``, ``OrderedPromptHistory``, and
    ``ResponseContext``. Optionally embeds each recorded Q+A pair into
    a ``Memory`` instance on a fire-and-forget background thread.

    Callers invoke ``record()`` instead of manually writing to 5
    separate stores.

    Args:
        context: The ResponseContext for prompt_attr_history recording.
        permanent_history: The PermanentHistory for chronological turns.
        ordered_history: The OrderedPromptHistory for named interactions.
        memory: Optional Memory instance. When provided, each
            successful ``record()`` call submits an embedding task to
            a dedicated single-worker thread pool. Failed embeds are
            logged at ``WARNING`` and dropped; they never propagate to
            the caller.
        memory_persist: If ``True``, persist the memory store to Parquet
            after each successful embed. Requires ``memory_persist_path``
            to be set.
        memory_persist_path: Fully-qualified file path for the Parquet
            persistence file. Ignored when ``memory_persist`` is
            ``False`` or ``memory`` is ``None``. Typically computed by
            ``FFAI.__init__()`` as
            ``f"{config.memory.persist_dir}/{config.memory.collection_name}.parquet"``.

    """

    def __init__(
        self,
        context: ResponseContext,
        permanent_history: PermanentHistory,
        ordered_history: OrderedPromptHistory,
        memory: Memory | None = None,
        memory_persist: bool = False,
        memory_persist_path: str | None = None,
    ) -> None:
        self.history: list[dict[str, Any]] = []
        self.clean_history: list[dict[str, Any]] = []
        self._context = context
        self._permanent = permanent_history
        self._ordered = ordered_history
        self._memory = memory
        self._memory_persist = memory_persist
        self._memory_persist_path = memory_persist_path
        self._embed_pool: concurrent.futures.ThreadPoolExecutor | None = (
            concurrent.futures.ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="ffai-memory"
            )
            if memory is not None
            else None
        )

    def record(
        self,
        prompt: str,
        response: Any,
        model: str,
        prompt_name: str | None = None,
        history: list[str] | None = None,
        status: str = "success",
        resolved_prompt: str | None = None,
        usage: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Record an interaction to all 5 history stores.

        When a ``Memory`` instance is configured and ``status == "success"``,
        the Q+A pair is embedded on a fire-and-forget background thread.
        Metadata is derived from ``prompt_name`` when not explicitly
        passed.

        Args:
            prompt: The resolved prompt text.
            response: The cleaned response.
            model: Model identifier used.
            prompt_name: Logical name for the prompt.
            history: List of prompt names this call depends on.
            status: Execution status ("success", "skipped", "failed").
                Only ``"success"`` triggers embedding.
            resolved_prompt: The fully interpolated prompt sent to the model.
            usage: Token usage from the API call.
            metadata: Optional caller metadata. When ``None``, derived
                from ``prompt_name`` as ``{"prompt_name": prompt_name}``
                (or ``{}`` if ``prompt_name`` is also ``None``).

        """
        md = (
            metadata
            if metadata is not None
            else ({"prompt_name": prompt_name} if prompt_name else {})
        )

        # Coalescing control: pass metadata to PermanentHistory only when
        # we want to prevent coalescing (i.e., when prompt_name is present
        # or the caller supplied explicit metadata). Without that, the
        # turn still gets stored, just with empty metadata so consecutive
        # user turns coalesce as before.
        permanent_metadata = md if (prompt_name or metadata is not None) else None
        self._permanent.add_turn_user(prompt, metadata=permanent_metadata)
        self._permanent.add_turn_assistant(
            str(response) if response is not None else "",
            metadata=permanent_metadata,
        )

        interaction: dict[str, Any] = {
            "prompt": prompt,
            "response": response,
            "prompt_name": prompt_name,
            "timestamp": time.time(),
            "model": model,
            "history": history,
            "status": status,
            "resolved_prompt": resolved_prompt,
            "usage": usage,
        }

        self.history.append(interaction)
        self.clean_history.append(interaction)

        self._context.record(prompt, response, model, prompt_name, history)

        self._ordered.add_interaction(
            model=model,
            prompt=prompt,
            response=str(response) if response is not None else "",
            prompt_name=prompt_name,
            history=history,
        )

        if (
            self._memory is not None
            and self._embed_pool is not None
            and status == "success"
        ):
            response_str = str(response) if response is not None else ""
            embedded_text = f"{prompt}\n{response_str}" if response_str else prompt
            turn = {
                "role": "assistant",
                "content": [{"type": "text", "text": embedded_text}],
                "timestamp": time.time(),
                "metadata": md,
            }
            self._embed_pool.submit(self._safe_index_turn, embedded_text, turn, md)

    def _safe_index_turn(
        self,
        text: str,
        turn: dict[str, Any],
        metadata: dict[str, Any],
    ) -> None:
        """Embed and optionally persist a turn on the background thread.

        All exceptions are caught and logged at ``WARNING`` so that
        embedding failures never propagate to ``record()`` callers
        (the whole point of fire-and-forget).
        """
        if self._memory is None:
            return
        try:
            self._memory.index_turn_text(text=text, turn=turn, metadata=metadata)
            if self._memory_persist and self._memory_persist_path:
                from ..memory import persist_store

                persist_store(self._memory.store, path=self._memory_persist_path)
        except Exception as exc:
            logger.warning("Memory embedding failed: %s", exc)

    def shutdown(self) -> None:
        """Shut down the background embed thread pool.

        Safe to call multiple times. Called by ``FFAI.close()`` on
        teardown. Pending submitted tasks are not awaited
        (``wait=False``); they may complete or be abandoned at
        interpreter exit.
        """
        if self._embed_pool is not None:
            self._embed_pool.shutdown(wait=False)
            self._embed_pool = None
