Cost & Usage Tracking
=====================

FFAI tracks token usage, estimated cost, and wall-clock duration on every call.
This guide shows how to access and export that data.

Per-call tracking
-----------------

Every ``generate_response()`` call returns a ``ResponseResult`` with usage data:

.. code-block:: python

   result = ffai.generate_response(
       prompt="What is 2+2?",
       prompt_name="math",
   )

   print(result.usage)          # TokenUsage(input_tokens=30, output_tokens=9, total_tokens=39)
   print(result.cost_usd)       # 3e-06
   print(result.duration_ms)    # 842.3
   print(result.model)          # "mistral/mistral-small-latest"

``TokenUsage`` fields:

- ``input_tokens`` (``int``) — tokens in the prompt
- ``output_tokens`` (``int``) — tokens in the response
- ``total_tokens`` (``int``) — sum of input + output

``cost_usd`` is estimated based on per-model pricing. ``duration_ms`` is the
wall-clock time for the API call in milliseconds.

Aggregated stats
----------------

Get usage counts grouped by model or prompt name:

.. code-block:: python

   model_stats = ffai.get_model_usage_stats()
   print(model_stats)
   # {"mistral/mistral-small-latest": 5, "openai/gpt-4o": 1}

   prompt_stats = ffai.get_prompt_name_usage_stats()
   print(prompt_stats)
   # {"math": 1, "geography": 1, "translate": 3}

Both return ``dict[str, int]`` mapping names to call counts.

DataFrame export
----------------

Export usage data as Polars DataFrames for analysis:

.. code-block:: python

   df = ffai.history_to_dataframe()
   print(df.head())

   df = ffai.get_model_stats_df()
   df = ffai.get_prompt_name_stats_df()
   df = ffai.get_response_length_stats()
   df = ffai.interaction_counts_by_date()

Search history by text or model:

.. code-block:: python

   df = ffai.search_history(text="error", model="gpt-4o")

Persistence
-----------

Write all history stores to Parquet files:

.. code-block:: python

   ffai.persist_all_histories()

Read them back later with Polars:

.. code-block:: python

   import polars as pl

   df = pl.read_parquet("output_dir/history.parquet")

RAG query tracking
------------------

``FFAI.query()`` and ``RAG.query()`` also track cost and duration:

.. code-block:: python

   result = ffai.query("What is Python?")
   print(result.cost_usd)       # cost of the generation step
   print(result.duration_ms)    # total time including search + generation

``QueryResult`` includes the same ``cost_usd`` and ``duration_ms`` fields as
``ResponseResult``.

See also
--------

- :doc:`../tutorials/rag_pipeline` — RAG tutorial with cost tracking
- :doc:`history` — full history management guide
- :ref:`modindex` — API reference for ``TokenUsage`` and ``ResponseResult``
