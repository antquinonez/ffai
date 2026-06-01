Vector Store Backends
=====================

FFAI's RAG system uses a pluggable vector store architecture. All backends
implement the same :class:`~ffai.rag.stores.base.VectorStoreBase` interface,
so you can swap backends without changing your indexing, search, or query code.

Available backends
------------------

+------------------+-------------------------------------+-------------------------------------------+
| Backend          | Install                             | Mode                                      |
+==================+=====================================+===========================================+
| **ChromaDB**     | Included with ``ffai[rag]``         | Local files (persistent client)           |
+------------------+-------------------------------------+-------------------------------------------+
| **Qdrant**       | ``pip install qdrant-client``       | Local, server, or cloud                   |
+------------------+-------------------------------------+-------------------------------------------+
| **pgvector**     | ``pip install psycopg asyncpg``     | Server (Docker or managed PostgreSQL)     |
+------------------+-------------------------------------+-------------------------------------------+
| **SQLite-vss**   | ``pip install sqlite-vss``          | Local files (SQLite extension)            |
+------------------+-------------------------------------+-------------------------------------------+

Check which backends are available in your environment:

.. code-block:: python

   from ffai.rag.stores import list_stores, list_available_stores

   print(list_stores())
   # ['chroma', 'pgvector', 'qdrant', 'sqlite_vss']

   print(list_available_stores())
   # ['chroma', 'qdrant']

``list_stores()`` returns all known backends. ``list_available_stores()``
returns only those whose dependencies are installed.

Choosing a backend
------------------

- **ChromaDB** — best for getting started. Zero configuration, persistent
  local storage, included with ``ffai[rag]``.
- **Qdrant** — best for production. Supports local mode (no server), dedicated
  server, and Qdrant Cloud. High performance with filtering.
- **pgvector** — best when you already have PostgreSQL. Uses the pgvector
  extension for vector similarity search. Requires a running Postgres instance.
- **SQLite-vss** — best for embedded and edge deployments. Zero infrastructure,
  single-file database.

Using get_store()
-----------------

The :func:`~ffai.rag.stores.get_store` factory creates a store instance by
backend name. Each backend accepts different constructor arguments:

.. code-block:: python

   from ffai.rag.stores import get_store

   # ChromaDB (default)
   store = get_store("chroma", collection_name="my_kb", dir="./chroma_db")

   # Qdrant local mode (no server needed)
   store = get_store("qdrant", path="./qdrant_db", embedding_dim=1024)

   # Qdrant server
   store = get_store("qdrant", host="localhost", port=6333, embedding_dim=1024)

   # Qdrant Cloud
   store = get_store("qdrant", url="https://cluster.qdrant.io", api_key="...", embedding_dim=1024)

   # pgvector (requires running PostgreSQL)
   store = get_store("pgvector",
       connection_string="postgresql://ffai:ffai@localhost:5432/ffai",
       embedding_dim=1024)

   # SQLite-vss
   store = get_store("sqlite_vss", db_path="./vss.db", embedding_dim=1024)

All stores implement the same interface. After construction, the rest of your
code stays the same:

.. code-block:: python

   from ffai.rag.embed import Embeddings
   from ffai.rag import RAG

   embed = Embeddings("mistral/mistral-embed", api_key="your-key")
   rag = RAG(embed=embed, store=store, chunk_size=500, chunk_overlap=100)

   rag.index("Python is a programming language...", source="python_intro")
   hits = rag.search("programming")

Configuration via YAML
----------------------

Set the backend in ``config/main.yaml`` under the ``rag:`` key:

.. code-block:: yaml

   rag:
     enabled: true
     store_backend: qdrant
     store_config:
       path: "./qdrant_db"
       embedding_dim: 1024
     embedding_model: "mistral/mistral-embed"
     chunker: recursive
     chunk_size: 1000
     chunk_overlap: 200

``store_config`` passes backend-specific constructor arguments directly to the
backend. Omit it (or pass ``{}``) to use defaults.

Then use :meth:`~ffai.rag.rag.RAG.from_config`:

.. code-block:: python

   from ffai.rag import RAG

   rag = RAG.from_config()  # reads store_backend + store_config from YAML

Backend-specific notes
----------------------

ChromaDB
^^^^^^^^

The default backend. No additional installation beyond ``ffai[rag]``.

.. code-block:: python

   store = get_store("chroma", collection_name="my_kb", dir="./chroma_db")

Constructor arguments:

- ``collection_name`` (str) — Collection name. Default: ``"ffai_kb"``
- ``dir`` (str) — Filesystem path for persistent storage. Default: ``"./chroma_db"``

Qdrant
^^^^^^

Supports four modes: **local** (file-based, no server), **server** (connect to
a running Qdrant instance), **cloud** (Qdrant Cloud), and **in-memory**
(ephemeral, data lost on exit).

Local mode uses ``asyncio.to_thread`` internally to avoid file-lock conflicts
between sync and async clients. Non-UUID string IDs are automatically converted
to deterministic UUID5 values. Payload indexes for ``source`` and
``chunking_strategy`` are created automatically when a new collection is
initialized (required for filtering on Qdrant Cloud).

.. code-block:: python

   # Local mode (no server)
   store = get_store("qdrant", path="./qdrant_db", embedding_dim=1024)

   # In-memory mode (ephemeral)
   store = get_store("qdrant", location=":memory:", embedding_dim=1024)

   # Server mode
   store = get_store("qdrant", host="localhost", port=6333, embedding_dim=1024)

   # Cloud mode
   store = get_store("qdrant", url="https://your-cluster.qdrant.io",
                     api_key="your-key", embedding_dim=1024)

Constructor arguments:

- ``collection_name`` (str) — Default: ``"ffai_kb"``
- ``embedding_dim`` (int) — Vector dimensionality. Default: ``1024``
- ``path`` (str or None) — Local storage path. Set this for local mode.
- ``location`` (str or None) — ``":memory:"`` for in-memory mode.
- ``host`` (str) — Server host. Default: ``"localhost"``
- ``port`` (int) — Server port. Default: ``6333``
- ``url`` (str or None) — Cloud URL.
- ``api_key`` (str or None) — Cloud API key.

pgvector
^^^^^^^^

Requires a running PostgreSQL instance with the pgvector extension. Use
``docker-compose.dev.yaml`` to start a local instance:

.. code-block:: bash

   docker compose -f docker-compose.dev.yaml up -d

.. code-block:: python

   store = get_store("pgvector",
       connection_string="postgresql://ffai:ffai@localhost:5432/ffai",
       embedding_dim=1024)

Constructor arguments:

- ``connection_string`` (str) — PostgreSQL connection string.
- ``collection_name`` (str) — Table name prefix. Default: ``"ffai_kb"``
- ``embedding_dim`` (int) — Default: ``1024``

SQLite-vss
^^^^^^^^^^

Zero-infrastructure option. Uses the SQLite ``vss0`` extension for vector
similarity search. Good for embedded deployments and prototyping.

.. code-block:: python

   store = get_store("sqlite_vss", db_path="./vss.db", embedding_dim=1024)

Constructor arguments:

- ``db_path`` (str) — Path to the SQLite database file. Default: ``"./vss.db"``
- ``collection_name`` (str) — Table name. Default: ``"ffai_kb"``
- ``embedding_dim`` (int) — Default: ``1024``

VectorStoreBase interface
-------------------------

All backends implement these methods:

========================= =====================================================
Method                    Description
========================= =====================================================
``name``                  Backend identifier (``"chroma"``, ``"qdrant"``, etc.)
``aadd(ids, texts, ...``)  Add documents with embeddings (async)
``asearch(embedding,``)   Search by vector similarity (async)
``delete_by_source()``    Delete all chunks for a source
``delete_by_source_``     Delete chunks matching source + chunking strategy
``and_strategy()``
``count()``               Total number of stored chunks
``clear()``               Delete all data and recreate
``list_sources()``        Sorted list of indexed source names
``get_all()``             All stored documents as dicts
``needs_reindex()``       Check if source needs re-indexing (checksum)
========================= =====================================================

Docker setup for pgvector and Qdrant
-------------------------------------

A ``docker-compose.dev.yaml`` is provided for local development:

.. code-block:: bash

   # Start both pgvector and Qdrant
   docker compose -f docker-compose.dev.yaml up -d

   # Stop
   docker compose -f docker-compose.dev.yaml down

Example notebooks
-----------------

The ``examples/vector_stores/`` directory contains runnable Jupyter notebooks
that demonstrate each backend and mode. All notebooks clean up after themselves
— no data left on disk or in the cloud after execution.

+----------------------------------------+------------------------------------------------+--------------------------------------------------+
| Notebook                               | What it demonstrates                           | Cleanup                                          |
+========================================+================================================+==================================================+
| ``qdrant_memory.ipynb``                | In-memory mode: add, search, filter, delete    | Data vanishes with kernel                        |
+----------------------------------------+------------------------------------------------+--------------------------------------------------+
| ``qdrant_local.ipynb``                 | Local file mode: persistence, re-open verify   | ``shutil.rmtree()`` on temp dir                  |
+----------------------------------------+------------------------------------------------+--------------------------------------------------+
| ``qdrant_server.ipynb``                | Server mode: auto-starts Docker, search        | Clears collection, stops container               |
+----------------------------------------+------------------------------------------------+--------------------------------------------------+
| ``qdrant_cloud.ipynb``                 | Cloud mode: indexes to remote cluster, queries | ``delete_collection()`` on cluster               |
+----------------------------------------+------------------------------------------------+--------------------------------------------------+
| ``backend_comparison.ipynb``           | ChromaDB vs Qdrant side-by-side via factory    | Clears both stores + removes temp dir            |
+----------------------------------------+------------------------------------------------+--------------------------------------------------+

All notebooks use synthetic embeddings (random unit vectors via NumPy) so no
embedding API key is required. The ``qdrant_cloud`` notebook requires
``QDRANT_CLUSTER_ENDPOINT`` and ``QDRANT_KEY`` in your ``.env`` file. The
``qdrant_server`` notebook requires Docker.

See also
--------

- :doc:`rag_search` — search strategies (vector, BM25, hybrid, rerankers)
- :doc:`../tutorials/rag_pipeline` — end-to-end RAG tutorial
- :doc:`installation` — installation and setup
