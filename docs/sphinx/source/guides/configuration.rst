Configuration
=============

FFAI reads YAML config from a ``config/`` directory at the project root.

Config files
------------

``config/main.yaml``
   Retry settings, observability toggles, and general options.

``config/clients.yaml``
   Client type definitions.

``config/paths.yaml``
   File system paths for output and persistence.

``config/model_defaults.yaml``
   Per-model default parameters (temperature, max_tokens, etc.).

``config/logging.yaml``
   Logging format with context variables (``batch_name``, ``prompt_name``).

Programmatic model defaults
---------------------------

.. code-block:: python

   from ffai.Clients.model_defaults import register_model_defaults

   register_model_defaults("my-custom-model", {
       "temperature": 0.3,
       "max_tokens": 4096,
   })

RAG configuration
-----------------

``RAG.from_config()`` reads settings from ``config/main.yaml`` under the
``rag:`` key:

.. code-block:: yaml

   rag:
     enabled: true
     persist_dir: "./chroma_db"
     collection_name: "default"
     embedding_model: "mistral/mistral-embed"
     chunker: "recursive"
     chunk_size: 500
     chunk_overlap: 100
     bm25_alpha: 0.5
     reranker: "diversity"

.. code-block:: python

   from ffai.rag import RAG
   from ffai.rag.embed import Embeddings

   embed = Embeddings("mistral/mistral-embed", api_key="your-key")
   rag = RAG.from_config(embed=embed)
