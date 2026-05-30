Installation
============

Basic install:

.. code-block:: bash

   pip install ffai

Or with uv_:

.. code-block:: bash

   uv add ffai

Install directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/antquinonez/ffai.git

.. _uv: https://docs.astral.sh/uv/

Optional extras
---------------

+-----------------+-------------------------------+-----------------------------------+
| Extra           | pip                           | uv                                |
+=================+===============================+===================================+
| RAG             | ``pip install "ffai[rag]"``   | ``uv add "ffai[rag]"``           |
+-----------------+-------------------------------+-----------------------------------+
| OpenTelemetry   | ``pip install "ffai[otel]"``  | ``uv add "ffai[otel]"``          |
+-----------------+-------------------------------+-----------------------------------+
| Both            | ``pip install "ffai[rag,otel]"`` | ``uv add "ffai[rag,otel]"``   |
+-----------------+-------------------------------+-----------------------------------+

RAG installs ChromaDB for persistent vector storage. OpenTelemetry installs OTLP
span export for tracing.

.. note::

   Quotes are required around ``ffai[rag]`` in zsh and some other shells, since
   brackets are special characters. Bash does not require quotes.

Vector store backends
---------------------

FFAI's RAG system supports multiple vector store backends. ChromaDB is included
with ``ffai[rag]``; others install separately:

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

All backends implement the same :class:`~ffai.rag.stores.base.VectorStoreBase`
interface. You can swap backends without changing any other code.

See :doc:`vector_stores` for configuration and usage details.

Running tests
-------------

Clone the repository and install dev dependencies:

.. code-block:: bash

   git clone https://github.com/antquinonez/ffai.git
   cd ffai
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev,rag]"

Run the full test suite:

.. code-block:: bash

   pytest tests/ -x -q

Run tests for a specific backend:

.. code-block:: bash

   pytest tests/ -m chroma            # ChromaDB unit + integration
   pytest tests/ -m qdrant            # Qdrant unit + integration
   pytest tests/ -m pgvector          # pgvector unit tests (mocked)
   pytest tests/ -m sqlite_vss        # SQLite-vss unit tests (mocked)

Run integration tests (require real packages installed):

.. code-block:: bash

   pytest tests/integration/ -m integration -v

Integration tests for vector stores use synthetic embeddings (no LLM calls
needed). They are excluded from the default ``pytest`` run via the
``-m 'not integration'`` filter in ``pyproject.toml``.

Qdrant server-mode tests require a running Qdrant instance. Pass
``--qdrant-server`` to auto-start a Docker container:

.. code-block:: bash

   pytest tests/integration/ -m integration --qdrant-server -v

Qdrant cloud-mode tests require environment variables:

.. code-block:: bash

   export QDRANT_CLUSTER_ENDPOINT="https://your-cluster.qdrant.io"
   export QDRANT_KEY="your-api-key"
   pytest tests/integration/ -m integration -v -k Cloud

Requirements
------------

- Python >= 3.10
