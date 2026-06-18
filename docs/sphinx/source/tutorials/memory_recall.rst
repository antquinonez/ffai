Memory Vector Recall: Remember What Was Discussed
=================================================

In this tutorial you will add semantic recall to an FFAI-powered assistant.
By the end, your assistant will retrieve past Q+A turns by *meaning*, not
by exact keyword match — so a later question like "the database issue from
earlier" finds the turn about a Postgres migration even though neither
"database" nor "issue" appears in that turn.

You'll learn:

- How to enable memory in ``FFAI``
- How ``generate_response()`` eagerly embeds each Q+A pair
- How ``ffai.history.search()`` retrieves semantically related turns
- How to filter by similarity threshold
- How to persist memory across process restarts

Prerequisites
-------------

- Python >= 3.10
- ``pip install "ffai[memory]"`` (pulls ``fastembed`` for local embeddings)
- A Mistral API key (or any LiteLLM-supported provider) set as
  ``MISTRAL_API_KEY``

.. note::

   The companion notebook ``examples/memory_basics/memory_basics.ipynb``
   runs the same examples with a local embedding backend and no API key.
   Use it if you want to try the code without setting up a provider.

Step 1: Enable memory
^^^^^^^^^^^^^^^^^^^^^

Memory is **opt-in**. Pass ``memory_enabled=True`` to ``FFAI``:

.. code-block:: python

   import os

   from ffai.Clients import FFLiteLLMClient
   from ffai import FFAI

   client = FFLiteLLMClient(
       model_string="mistral/mistral-small-latest",
       api_key=os.environ["MISTRAL_API_KEY"],
   )

   ffai = FFAI(client, memory_enabled=True)

At construction time FFAI resolves an embedding backend via this ladder:

1. ``config.memory.embedding_model`` (if set in ``config/main.yaml``)
2. ``local/all-MiniLM-L6-v2`` if ``fastembed`` or ``sentence-transformers``
   is importable
3. ``mistral/mistral-embed`` if ``MISTRAL_API_KEY`` is set
4. ``openai/text-embedding-3-small`` if ``OPENAI_API_KEY`` is set
5. Otherwise: memory is disabled with a warning

You can also enable memory via configuration instead of the constructor
keyword — see :doc:`../guides/memory`.

Step 2: Generate turns — embedding happens automatically
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every successful ``generate_response()`` call embeds the **Q+A pair**
(``f"{prompt}\\n{response}"``) on a fire-and-forget background thread.
The call returns immediately; embedding does not block the response.

.. code-block:: python

   prompts = [
       ("Let's debug the Postgres migration", "postgres_debug"),
       ("Now update the user table schema", "schema_update"),
       ("What is Python's GIL?", "gil_concept"),
       ("How do I deploy to AWS?", "aws_deploy"),
       ("Write a haiku about autumn", "haiku"),
   ]

   for prompt, prompt_name in prompts:
       result = ffai.workflow.generate_response(prompt=prompt, prompt_name=prompt_name)
       print(f"{prompt_name}: {str(result.response)[:60]}...")

Output (response text is illustrative):

.. code-block:: text

   postgres_debug: The migration issue is the missing index on users.email...
   schema_update: ALTER TABLE users ADD COLUMN verified_at TIMESTAMP...
   gil_concept: The Global Interpreter Lock prevents multiple native...
   aws_deploy: Use the AWS CDK to define infrastructure as TypeScript...
   haiku: Red leaves whisper down / Branches bare against the sky...

The ``prompt_name`` you pass becomes ``metadata["prompt_name"]`` on the
indexed turn, which is how you'll trace hits back to specific prompts.

Step 3: Semantic search — find related turns by meaning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Call ``ffai.history.search(query, top_k)`` to retrieve turns ranked by
cosine similarity to the query:

.. code-block:: python

   hits = ffai.history.search("the database issue from earlier", top_k=3)

   for i, hit in enumerate(hits, 1):
       print(f"{i}. [{hit.score:.3f}] {hit.metadata['prompt_name']}")
       print(f"   {hit.text[:80]}...")

Output (scores are deterministic for a given embedding model):

.. code-block:: text

   1. [0.407] schema_update
      Now update the user table schema
      ALTER TABLE users ADD COLUMN verified_at TIMESTAMP DEFAULT NO...
   2. [0.291] postgres_debug
      Let's debug the Postgres migration
      The migration issue is the missing index on users.email...
   3. [0.056] haiku
      Write a haiku about autumn
      Red leaves whisper down / Branches bare against the sky...

Notice the query said "database issue" — words that appear in *none* of
the indexed turns. Linear substring search would return nothing. Vector
recall catches the schema-update and Postgres turns because they are
*semantically* related to databases.

Step 4: Inspect TurnHit fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each hit is a frozen dataclass with five fields:

.. code-block:: python

   hit = ffai.history.search("Python threading", top_k=1)[0]

   print(f"score:       {hit.score:.3f}  (cosine similarity, [-1.0, 1.0])")
   print(f"turn_index:  {hit.turn_index}  (position in the store)")
   print(f"text:        {hit.text[:60]}...")
   print(f"metadata:    {hit.metadata}")
   print(f"turn.role:   {hit.turn['role']}")

Output:

.. code-block:: text

   score:       0.566  (cosine similarity, [-1.0, 1.0])
   turn_index:  2  (position in the store)
   text:        What is Python's GIL?...
   metadata:    {'prompt_name': 'gil_concept'}
   turn.role:   assistant

``metadata`` is where Tier 2 (auto context injection) will later carry
``user_id`` / ``session_id`` for cross-user scoping. For now it carries
whatever you passed via ``prompt_name``.

Step 5: Filter by similarity threshold
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass ``threshold=N`` to drop hits below a cosine similarity floor. Useful
when you'd rather get ``[]`` than a noisy long tail:

.. code-block:: python

   query = "ship code to production"

   for threshold in [None, 0.3, 0.5]:
       hits = ffai.history.search(query, top_k=5, threshold=threshold)
       label = "no floor" if threshold is None else f">= {threshold}"
       print(f"Threshold {label:>10}: {len(hits)} hits")

Output:

.. code-block:: text

   Threshold   no floor: 5 hits
   Threshold     >= 0.3: 1 hits
   Threshold     >= 0.5: 0 hits

A typical starting threshold is ``0.3`` — high enough to drop unrelated
turns, low enough to catch synonym-heavy matches.

Step 6: Persist memory across restarts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, memory is **ephemeral** — the store dies with the process.
Set ``memory_persist=True`` to load and write a Parquet file:

.. code-block:: python

   ffai = FFAI(
       client,
       memory_enabled=True,
       memory_persist=True,
   )

At startup, FFAI loads ``<persist_dir>/<collection_name>.parquet`` if it
exists. After each successful embed, it writes the file. On restart,
prior turns are searchable immediately — no re-embedding required.

The defaults (configurable in ``config/main.yaml``):

.. code-block:: yaml

   memory:
     persist_dir: "./ffai_data/memory"
     collection_name: "ffai_turns"

Step 7: Disable search returns empty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When memory is disabled, ``ffai.history.search()`` returns ``[]`` — it
never raises. This means downstream code can call ``search()``
unconditionally:

.. code-block:: python

   ffai_off = FFAI(client)  # memory_enabled defaults to False

   print(ffai_off.history.memory)          # None
   print(ffai_off.history.search("x"))     # []

Complete listing
^^^^^^^^^^^^^^^^

.. code-block:: python

   import os

   from ffai.Clients import FFLiteLLMClient
   from ffai import FFAI

   client = FFLiteLLMClient(
       model_string="mistral/mistral-small-latest",
       api_key=os.environ["MISTRAL_API_KEY"],
   )

   ffai = FFAI(client, memory_enabled=True)

   for prompt, name in [
       ("Let's debug the Postgres migration", "postgres_debug"),
       ("Now update the user table schema", "schema_update"),
       ("What is Python's GIL?", "gil_concept"),
   ]:
       ffai.workflow.generate_response(prompt=prompt, prompt_name=name)

   hits = ffai.history.search("the database issue", top_k=2, threshold=0.3)
   for hit in hits:
       print(f"[{hit.score:.3f}] {hit.metadata['prompt_name']}")

Next steps
^^^^^^^^^^

- :doc:`../guides/memory` — configuration reference, embedding backend
  resolution, persistence semantics, troubleshooting
- Runnable notebooks in ``examples/``:

  - ``memory_basics/memory_basics.ipynb`` — Memory class directly with
    local fastembed (no API key)
  - ``memory_persistence/memory_persistence.ipynb`` — Parquet round-trip
  - ``memory_ffai_integration/memory_ffai_integration.ipynb`` — full FFAI
    wiring with a mock client (no API key)

- :doc:`../guides/history` — the four other history stores that complement
  memory vector recall
