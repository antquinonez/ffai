Memory Vector Recall
====================

FFAI's memory feature embeds completed Q+A turns and retrieves them by
semantic similarity. It is **opt-in** and disabled by default, matching
the ``rag.enabled`` convention.

This guide covers configuration, the embedding backend resolution ladder,
persistence semantics, the ``Memory`` class API, and troubleshooting. For
a hands-on walk-through, see :doc:`../tutorials/memory_recall`.

How it works
------------

When ``memory.enabled`` is true:

1. Each call to :meth:`~ffai.FFAI.FFAI.workflow.generate_response` with
   ``status="success"`` submits an embed task to a single-worker
   background thread.
2. The thread concatenates the prompt and response into a Q+A pair, embeds
   it, and appends it to an in-memory :class:`~ffai.core.memory.TurnVectorStore`.
3. :meth:`~ffai.core.history_manager.HistoryManager.search` embeds the
   query and returns ranked :class:`~ffai.core.memory.TurnHit` objects.

The embed runs fire-and-forget — it does not block the main
``generate_response()`` call. Failures are logged at ``WARNING`` and
dropped; they never propagate to the caller.

Configuration
-------------

All settings live under the ``memory:`` key in ``config/main.yaml``:

.. code-block:: yaml

   memory:
     enabled: false                    # opt-in; matches rag.enabled convention
     embedding_model: null             # null -> resolution ladder
     store_backend: memory             # reserved; Tier 1 uses in-memory store
     store_config: {}
     persist_dir: "./ffai_data/memory"
     collection_name: "ffai_turns"
     persist: false                    # ephemeral by default

Each field also has a constructor override on :class:`~ffai.FFAI.FFAI`:

.. list-table:: Constructor overrides
   :header-rows: 1
   :widths: 30 50 20

   * - Keyword
     - Effect
     - Default
   * - ``memory_enabled``
     - Override ``config.memory.enabled``. ``True`` enables; ``False``
       disables regardless of config.
     - ``None`` (use config)
   * - ``memory_embeddings``
     - Override ``config.memory.embedding_model``. ``None`` triggers the
       resolution ladder.
     - ``None`` (use config)
   * - ``memory_persist``
     - Override ``config.memory.persist``.
     - ``None`` (use config)
   * - ``memory``
     - Supply a pre-built :class:`~ffai.core.memory.Memory` instance.
       Bypasses the resolution ladder entirely. Useful for tests and for
       injecting a Memory with a custom embedding backend.
     - ``None``

Priority order: **constructor kwargs > environment variables > YAML**.

Environment variables
~~~~~~~~~~~~~~~~~~~~~

Pydantic settings reads nested fields via ``__`` delimiter. The env var
for ``memory.enabled`` is:

.. code-block:: bash

   MEMORY__ENABLED=true

There is no ``FFAI_`` prefix (the ``Config`` class does not set
``env_prefix``).

Embedding backend resolution
----------------------------

When ``memory.enabled`` is true and ``embedding_model`` is null,
FFAI resolves a backend at construction time:

1. If ``embedding_model`` is set, try to construct
   :class:`~ffai.core.embeddings.Embeddings` with it. ``ImportError`` →
   log a warning and disable memory.
2. Otherwise, try ``local/all-MiniLM-L6-v2``. Works if ``fastembed`` or
   ``sentence-transformers`` is importable.
3. Otherwise, if ``MISTRAL_API_KEY`` is set, use ``mistral/mistral-embed``.
4. Otherwise, if ``OPENAI_API_KEY`` is set, use
   ``openai/text-embedding-3-small``.
5. Otherwise: log a warning and disable memory (``ffai.history.memory``
   becomes ``None``, search returns ``[]``).

Local embeddings
~~~~~~~~~~~~~~~~

Install one of:

.. code-block:: bash

   pip install "ffai[memory]"    # fastembed only (~80 MB, no torch)
   pip install "ffai[rag]"       # fastembed + chromadb (heavier)
   pip install sentence-transformers   # alternative local backend

Local models are identified by a ``local/`` prefix:

.. code-block:: python

   from ffai.core.embeddings import Embeddings

   embeddings = Embeddings(model="local/all-MiniLM-L6-v2")

The first call downloads the model; subsequent calls use the cache.

API embeddings
~~~~~~~~~~~~~~

Any LiteLLM-supported embedding provider works. Configure via
``embedding_model``:

.. code-block:: yaml

   memory:
     enabled: true
     embedding_model: "mistral/mistral-embed"

Common options:

============================ ==================================================
Model string                 Notes
============================ ==================================================
``mistral/mistral-embed``    1024-dim, requires ``MISTRAL_API_KEY``
``openai/text-embedding-3-small`` 1536-dim, requires ``OPENAI_API_KEY``
``voyage/voyage-2``          Requires ``VOYAGE_API_KEY``
``local/all-MiniLM-L6-v2``   384-dim, no API key, ~80 MB download
============================ ==================================================

Persistence
-----------

When ``persist: true``:

- **Startup**: FFAI loads
  ``<persist_dir>/<collection_name>.parquet`` if it exists. The loaded
  :class:`~ffai.core.memory.TurnVectorStore` is wired into the active
  :class:`~ffai.core.memory.Memory` instance.
- **Per embed**: after each successful ``index_turn_text``, the store is
  rewritten to the same path.

The Parquet schema:

================== ===================== ==========================================
Column            polars dtype          Source
================== ===================== ==========================================
``text``           ``String``            The plain text that was embedded (Q+A pair)
``embedding``      ``List(Float64)``    The float vector
``turn``           ``String`` (JSON)     The raw turn dict
``metadata``       ``String`` (JSON)     The caller metadata dict
``_schema_version`` ``Int8``             Always ``1`` for this version
================== ===================== ==========================================

Writes are **overwrite-on-persist**, not append. Each write rewrites the
full file — acceptable for Tier 1 scale (<100K turns). For larger
workloads, future versions will support Chroma or Qdrant backends via
``store_backend`` (currently reserved).

.. warning::

   Writes are **not atomic** against process death mid-write. If the
   process crashes during a Parquet write, the file may be corrupted on
   restart. Acceptable for short-running workflows; long-running services
   should snapshot-then-rename or implement periodic backups.

Schema versioning
~~~~~~~~~~~~~~~~~

The ``_schema_version`` column lets future releases migrate the format.
Loaders reject unknown versions with a clear error rather than silently
corrupting data:

.. code-block:: python

   from ffai.core.memory import load_store

   store = load_store("turns.parquet")
   # ValueError: Unsupported memory persistence schema version(s): [2].
   # This build supports version 1 only.

TurnHit reference
-----------------

.. autoclass:: ffai.core.memory.TurnHit
   :members:

Fields:

``score``
    Cosine similarity in ``[-1.0, 1.0]``. Sorted descending by ``search()``.

``turn``
    The raw turn dict stored at index time. Mirrors the
    :class:`~ffai.core.history.permanent.PermanentHistory` turn shape:
    ``{"role": str, "content": [{"type": "text", "text": str}],
    "timestamp": float, "metadata": dict}``.

``turn_index``
    Position of the entry in the store at search time. Stable within a
    single search call; may shift after ``clear()``.

``text``
    The plain text that was embedded. Equal to the Q+A pair
    (``f"{prompt}\\n{response}"``).

``metadata``
    Caller-provided metadata. Always present (possibly empty dict).
    Currently carries ``prompt_name``; Tier 2 will add ``user_id``,
    ``session_id``, ``agent_id``.

Memory class API
----------------

The :class:`~ffai.core.memory.Memory` facade is exposed via
``ffai.history.memory`` (or ``None`` when disabled). You can also use it
directly without an :class:`~ffai.FFAI.FFAI` wrapper:

.. code-block:: python

   from ffai.core.embeddings import Embeddings
   from ffai.core.memory import Memory

   embeddings = Embeddings(model="local/all-MiniLM-L6-v2")
   memory = Memory(embeddings)

   memory.index_turn_text(
       text="user question\nassistant answer",
       turn={"role": "assistant", "content": [...], "timestamp": ...},
       metadata={"prompt_name": "demo"},
   )

   hits = memory.search("related question", top_k=3, threshold=0.3)

.. autoclass:: ffai.core.memory.Memory
   :members: index_turn, index_turn_text, aindex_turn, aindex_turn_text,
             search, asearch, reindex, count, clear

.. autoclass:: ffai.core.memory.TurnVectorStore
   :members: add, search, iter_entries, count, clear

Persistence API
---------------

.. autofunction:: ffai.core.memory.persist_store

.. autofunction:: ffai.core.memory.load_store

Lifecycle and resource cleanup
------------------------------

The background embed thread pool is owned by
:class:`~ffai.core.history.recorder.HistoryRecorder`. It is created when
a ``Memory`` is wired in and shut down by :meth:`~ffai.FFAI.FFAI.close`:

.. code-block:: python

   ffai = FFAI(client, memory_enabled=True)
   # ... use it ...
   ffai.close()   # shuts down the embed thread pool

``close()`` is idempotent. Calling it at process teardown avoids leaking
daemon threads, which matters in long-running services.

If you forget to call ``close()``, the daemon thread will not block
interpreter shutdown (it is a daemon thread by default).

Troubleshooting
---------------

Memory is ``None`` even though I set ``memory_enabled=True``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The resolution ladder fell through to step 5 (no backend available).
Check:

1. Is ``fastembed`` or ``sentence-transformers`` importable?
   (``pip install "ffai[memory]"``)
2. Is ``MISTRAL_API_KEY`` or ``OPENAI_API_KEY`` set?
3. Is ``config.memory.embedding_model`` pointing to a valid model?

A warning is logged at construction time when this happens:

.. code-block:: text

   Memory enabled but no embedding backend available.
   Install ffai[memory] (or ffai[rag]) for local embeddings, or set
   memory.embedding_model / MISTRAL_API_KEY / OPENAI_API_KEY.

Embeddings are slow
~~~~~~~~~~~~~~~~~~~

API embeddings add one network call per ``record()``. Local embeddings
(``local/all-MiniLM-L6-v2``) avoid the network but use CPU. The embed
runs on a fire-and-forget background thread, so it does not block the
main call — but it can fall behind if ``record()`` is called faster than
embeds can complete.

Symptoms: ``ffai.history.memory.count()`` lags behind the number of
``generate_response()`` calls.

Search returns nothing relevant
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Lower ``threshold`` (default is no floor)
- Increase ``top_k``
- Verify the embedding model matches the one used at index time. If you
  change ``embedding_model``, existing vectors become invalid; call
  :meth:`~ffai.core.memory.Memory.reindex` with the new backend.

Parquet file is corrupted after a crash
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Overwrite-on-persist is not atomic. If your service crashes mid-write,
recover from a backup or rebuild the store by re-running the original
workflow. Future versions may add atomic writes (write-temp-then-rename).

See also
--------

- :doc:`../tutorials/memory_recall` — hands-on tutorial
- :doc:`history` — the four other history stores (raw, clean, ordered,
  permanent)
- :doc:`configuration` — full configuration reference
