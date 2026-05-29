Configuration
=============

FFAI reads YAML config from a ``config/`` directory at the project root.
Configuration is loaded once and cached for the process lifetime.

Resolution order
----------------

Values are resolved in priority order (highest first):

1. **Constructor kwargs** — explicit parameters passed to Python constructors
2. **Environment variables** — e.g. ``MISTRAL_API_KEY``, ``OPENAI_API_KEY``
3. **YAML files** — defaults from ``config/*.yaml``

This means environment variables always override YAML, and explicit Python
arguments always override both.

Config file discovery
---------------------

FFAI searches for the ``config/`` directory in this order:

1. ``./config`` relative to the current working directory
2. ``config/`` relative to the package installation directory
3. ``../config/`` one level up from the current working directory

The first directory that exists is used. If none is found, ``./config`` is
assumed (and built-in defaults apply).

Config files
------------

``config/main.yaml``
   Retry settings, RAG settings, and observability toggles.

``config/clients.yaml``
   Client type definitions with API key env vars, model strings, and fallbacks.

``config/paths.yaml``
   File system paths for data persistence.

``config/model_defaults.yaml``
   Per-model default parameters (temperature, max_tokens, etc.).

``config/logging.yaml``
   Logging format, level, and rotation settings.

Retry settings
--------------

Defined in ``config/main.yaml`` under the ``retry:`` key:

.. code-block:: yaml

   retry:
     max_attempts: 3
     min_wait_seconds: 1
     max_wait_seconds: 60
     exponential_base: 2
     exponential_jitter: true
     retry_on_status_codes:
       - 429
       - 503
       - 502
       - 504
     log_level: "INFO"

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``max_attempts``
     - Maximum retry attempts per API call (default: 3)
   * - ``min_wait_seconds``
     - Minimum wait between retries in seconds (default: 1)
   * - ``max_wait_seconds``
     - Maximum wait between retries in seconds (default: 60)
   * - ``exponential_base``
     - Base for exponential backoff (default: 2)
   * - ``exponential_jitter``
     - Add random jitter to backoff intervals (default: true)
   * - ``retry_on_status_codes``
     - HTTP status codes that trigger a retry (default: [429, 503, 502, 504])
   * - ``log_level``
     - Log level for retry messages (default: "INFO")

Client configuration
--------------------

Defined in ``config/clients.yaml``. Each client type specifies the client
class, API key environment variable, default model, and optional fallbacks.

FFAI uses `LiteLLM <https://github.com/BerriAI/litellm>`_ as its primary
routing layer, supporting 100+ providers through a unified interface. For the
full list of supported providers and their model string formats, see the
`LiteLLM Providers docs <https://docs.litellm.ai/docs/providers>`_.

.. code-block:: yaml

   default_client: "litellm-mistral-small"

   client_types:
     litellm-mistral-small:
       client_class: "FFLiteLLMClient"
       type: "litellm"
       provider_prefix: "mistral/"
       api_key_env: "MISTRAL_API_KEY"
       default_model: "mistral-small-latest"
       fallbacks:
         - "openai/gpt-4o-mini"

     litellm-openai:
       client_class: "FFLiteLLMClient"
       type: "litellm"
       provider_prefix: "openai/"
       api_key_env: "OPENAI_API_KEY"
       default_model: "gpt-4o-mini"

     mistral-small:
       client_class: "FFMistralSmall"
       type: "native"
       api_key_env: "MISTRALSMALL_KEY"
       default_model: "mistral-small-2503"

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``default_client``
     - Client type name used when no client is specified (default: "litellm-mistral-small")
   * - ``client_class``
     - Python class name: ``"FFLiteLLMClient"`` or ``"FFMistralSmall"``
   * - ``type``
     - ``"litellm"`` for LiteLLM routing, ``"native"`` for direct SDK
   * - ``provider_prefix``
     - LiteLLM provider prefix (e.g. ``"mistral/"``, ``"openai/"``, ``"anthropic/"``)
   * - ``api_key_env``
     - Environment variable name holding the API key
   * - ``default_model``
     - Default model string for this client type
   * - ``fallbacks``
     - Optional list of fallback model strings tried on failure

Adding a new LiteLLM provider
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use a provider not already in ``clients.yaml``, add a new entry under
``client_types:``. The ``provider_prefix`` must match LiteLLM's convention.

For provider-specific model strings and prefixes, consult the
`LiteLLM Providers docs <https://docs.litellm.ai/docs/providers>`_.

Example — adding Groq:

.. code-block:: yaml

   client_types:
     litellm-groq:
       client_class: "FFLiteLLMClient"
       type: "litellm"
       provider_prefix: "groq/"
       api_key_env: "GROQ_API_KEY"
       default_model: "llama-3.1-70b-versatile"

Then set the environment variable:

.. code-block:: bash

   export GROQ_API_KEY="your-groq-key"

Model string format
^^^^^^^^^^^^^^^^^^^

LiteLLM model strings follow the pattern ``provider/model-name``. Some common
prefixes:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Prefix
     - Example model string
     - API key env var
   * - ``openai/``
     - ``openai/gpt-4o``
     - ``OPENAI_API_KEY``
   * - ``anthropic/``
     - ``anthropic/claude-3-5-sonnet-20241022``
     - ``ANTHROPIC_API_KEY``
   * - ``mistral/``
     - ``mistral/mistral-small-latest``
     - ``MISTRAL_API_KEY``
   * - ``azure/``
     - ``azure/my-deployment-name``
     - ``AZURE_OPENAI_API_KEY``
   * - ``gemini/``
     - ``gemini/gemini-1.5-pro``
     - ``GEMINI_API_KEY``
   * - ``groq/``
     - ``groq/llama-3.1-70b-versatile``
     - ``GROQ_API_KEY``
   * - ``deepseek/``
     - ``deepseek/deepseek-chat``
     - ``DEEPSEEK_API_KEY``
   * - ``perplexity/``
     - ``perplexity/sonar``
     - ``PERPLEXITY_API_KEY``
   * - ``ollama/``
     - ``ollama/llama3``
     - (local, no key needed)

For the complete and current list, see
`LiteLLM Providers <https://docs.litellm.ai/docs/providers>`_.

Paths configuration
-------------------

Defined in ``config/paths.yaml``:

.. code-block:: yaml

   paths:
     ffai_data: "./ffai_data"

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``ffai_data``
     - Directory for history persistence and exported data (default: ``"./ffai_data"``)

Model defaults
--------------

Defined in ``config/model_defaults.yaml``. The ``generic`` block applies to
all models. The ``models`` block overrides per model string:

.. code-block:: yaml

   model_defaults:
     generic:
       max_tokens: 4096
       temperature: 0.7
       system_instructions: "You are a helpful assistant."
     models:
       mistral/mistral-small-latest:
         max_tokens: 32000
         temperature: 0.7
       anthropic/claude-3-5-sonnet-20241022:
         max_tokens: 8192
         temperature: 0.7

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``generic``
     - Defaults applied to all models
   * - ``models``
     - Per-model overrides keyed by LiteLLM model string
   * - ``max_tokens``
     - Maximum tokens in the response (default: 4096)
   * - ``temperature``
     - Sampling temperature 0–2 (default: 0.7)
   * - ``system_instructions``
     - Default system prompt

Programmatic overrides:

.. code-block:: python

   from ffai.Clients.model_defaults import register_model_defaults

   register_model_defaults("my-custom-model", {
       "temperature": 0.3,
       "max_tokens": 4096,
   })

Logging configuration
---------------------

Defined in ``config/logging.yaml``:

.. code-block:: yaml

   logging:
     directory: "logs"
     filename: "orchestrator.log"
     level: "INFO"
     format: "%(asctime)s - %(name)s - %(levelname)s - batch=%(batch_name)s prompt=%(prompt_name)s - %(message)s"
     rotation:
       when: "midnight"
       interval: 1
       backup_count: 10

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``directory``
     - Log file directory (default: ``"logs"``)
   * - ``filename``
     - Log file name (default: ``"orchestrator.log"``)
   * - ``level``
     - Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL (default: ``"INFO"``)
   * - ``format``
     - Python logging format string. ``%(batch_name)s`` and ``%(prompt_name)s`` are context variables
   * - ``rotation.when``
     - TimedRotatingFileHandler unit: "midnight", "D", "H", etc. (default: ``"midnight"``)
   * - ``rotation.interval``
     - Rotation interval (default: 1)
   * - ``rotation.backup_count``
     - Number of rotated log files to retain (default: 10)

Observability configuration
---------------------------

Defined in ``config/main.yaml`` under the ``observability:`` key:

.. code-block:: yaml

   observability:
     enabled: false
     otel:
       service_name: "ffai"
       endpoint: "http://localhost:4317"
       export_traces: true
       insecure: true
     token_tracking: true
     cost_tracking: true

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``enabled``
     - Enable OpenTelemetry span emission (default: false)
   * - ``otel.service_name``
     - Service name for OTLP spans (default: ``"plico"``)
   * - ``otel.endpoint``
     - OTLP gRPC endpoint (default: ``"http://localhost:4317"``)
   * - ``otel.export_traces``
     - Whether to export traces (default: true)
   * - ``otel.insecure``
     - Use insecure gRPC connection (default: true)
   * - ``token_tracking``
     - Track token usage per call (default: true)
   * - ``cost_tracking``
     - Estimate cost per call (default: true)

RAG configuration
-----------------

Defined in ``config/main.yaml`` under the ``rag:`` key.
``RAG.from_config()`` reads these settings and creates the ``Embeddings``
and ``VectorStore`` automatically:

.. code-block:: yaml

   rag:
     enabled: true
     persist_dir: "./chroma_db"
     collection_name: "ffai_kb"
     embedding_model: "mistral/mistral-embed"
     chunker: "recursive"
     chunk_size: 1000
     chunk_overlap: 200
     bm25_alpha: 0.5
     reranker: "diversity"

.. code-block:: python

   from ffai.rag import RAG

   # Zero-config: reads embedding_model from config, API key from env
   rag = RAG.from_config()

   # Or with explicit API key
   rag = RAG.from_config(api_key="your-key")

The API key is resolved in order: the ``api_key`` parameter, the
provider-specific environment variable (e.g. ``MISTRAL_API_KEY``), and
finally ``None`` (which will raise at embed time if no key is found).

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Key
     - Description
   * - ``enabled``
     - Whether FFAI auto-creates a RAG instance on init (default: false)
   * - ``persist_dir``
     - ChromaDB storage directory (default: ``"./chroma_db"``)
   * - ``collection_name``
     - ChromaDB collection name (default: ``"ffai_kb"``)
   * - ``embedding_model``
     - LiteLLM model string for embeddings (default: ``"mistral/mistral-embed"``)
   * - ``chunker``
     - Chunking strategy: ``"recursive"``, ``"character"``, ``"markdown"``, ``"code"``, ``"hierarchical"`` (default: ``"recursive"``)
   * - ``chunk_size``
     - Maximum characters per chunk (default: 1000)
   * - ``chunk_overlap``
     - Overlap characters between chunks (default: 200)
   * - ``bm25_alpha``
     - Hybrid search alpha (0 = pure BM25, 1 = pure vector). ``null`` disables BM25 (default: null)
   * - ``reranker``
     - Reranker strategy: ``"diversity"``, ``"cross_encoder"``, ``"noop"``. ``null`` disables reranking (default: null)

API key environment variables
-----------------------------

FFAI reads API keys from environment variables based on the provider prefix:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Provider
     - Model prefix
     - Environment variable
   * - Mistral
     - ``mistral/``
     - ``MISTRAL_API_KEY``
   * - OpenAI
     - ``openai/``
     - ``OPENAI_API_KEY``
   * - Anthropic
     - ``anthropic/``
     - ``ANTHROPIC_API_KEY``
   * - Azure
     - ``azure/``
     - ``AZURE_OPENAI_API_KEY``
   * - Other
     - ``<provider>/``
     - ``<PROVIDER>_API_KEY``

For RAG embeddings, the same mapping applies based on the
``embedding_model`` string. For example, ``embedding_model: "openai/text-embedding-3-small"``
reads ``OPENAI_API_KEY``.

Set keys in your environment:

.. code-block:: bash

   export MISTRAL_API_KEY="your-key"
   export OPENAI_API_KEY="your-key"

Or use a ``.env`` file in the project root (loaded automatically by
``python-dotenv``).

See also
--------

- :doc:`quickstart` — minimal working examples
- :doc:`../tutorials/rag_pipeline` — full RAG tutorial
- :doc:`rag_search` — search strategies (BM25, hybrid, reranking)
- :doc:`chunking` — chunking strategy guide
- `LiteLLM Providers <https://docs.litellm.ai/docs/providers>`_ — supported providers and model strings
