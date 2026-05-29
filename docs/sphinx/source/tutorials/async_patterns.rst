Async Patterns with FFAI
=========================

In this tutorial you will use FFAI's async API to run LLM calls and RAG
operations concurrently. By the end you will be able to use async clients,
async RAG methods, and async DAG execution.

Prerequisites
-------------

- Python >= 3.10
- ``pip install "ffai[rag]"``
- A Mistral API key (or any LiteLLM-supported provider) set as
  ``MISTRAL_API_KEY``
- Familiarity with the :doc:`quickstart` and :doc:`rag_pipeline`

Step 1: Create an async client
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``AsyncFFLiteLLMClient`` instead of ``FFLiteLLMClient``:

.. code-block:: python

   import os

   from ffai.Clients import AsyncFFLiteLLMClient
   from ffai.FFAI import FFAI

   client = AsyncFFLiteLLMClient(
       model_string="mistral/mistral-small-latest",
       api_key=os.environ["MISTRAL_API_KEY"],
   )

   ffai = FFAI(client)

Async clients support the same interface as sync clients. ``generate_response``
still works synchronously, but ``execute_graph`` and async RAG methods require
an async client.

Step 2: Async RAG operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FFAI provides async counterparts for all RAG methods. Use ``await`` with
``aquery``, ``asearch``, and ``aindex``:

.. code-block:: python

   from ffai.rag import RAG
   from ffai.rag.embed import Embeddings
   from ffai.rag.store import VectorStore

   embed = Embeddings("mistral/mistral-embed", api_key=os.environ["MISTRAL_API_KEY"])
   store = VectorStore(collection_name="async_kb", dir="./async_db")
   rag = RAG(embed=embed, store=store, chunk_size=500, chunk_overlap=100)

   ffai = FFAI(client, rag=rag)

   # Async index
   count = await ffai.aindex("Python is a high-level language.", source="doc1")
   print(f"Indexed {count} chunks")
   # Indexed 1 chunks

   # Async search
   hits = await ffai.asearch("What is Python?", top_k=5)
   print(f"Found {len(hits)} hits")
   # Found 1 hits

   # Async query
   result = await ffai.aquery("What is Python?")
   print(result.answer)
   print(result.sources)

Async methods return the same types as their sync counterparts:
``aindex`` → ``int``, ``asearch`` → ``list[SearchHit]``,
``aquery`` → ``QueryResult``.

Step 3: Async DAG execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``execute_graph`` runs prompts in topological-parallel order using
``asyncio.gather``. It requires an ``AsyncFFLiteLLMClient``:

.. code-block:: python

   prompts = [
       {"prompt_name": "topic", "prompt": "Suggest a topic"},
       {"prompt_name": "outline", "prompt": "Create an outline about {{topic.response}}",
        "history": ["topic"]},
       {"prompt_name": "article", "prompt": "Write an article based on:\n{{outline.response}}",
        "history": ["outline"]},
   ]

   graph_result = await ffai.execute_graph(prompts, max_concurrency=10)

   for name, r in graph_result.results.items():
       print(f"{name}: {r.status} ({r.duration_ms:.0f}ms)")

Output (illustrative):

.. code-block:: text

   topic: success (842ms)
   outline: success (1203ms)
   article: success (2156ms)

``max_concurrency`` controls how many prompts run in parallel at each level.

Step 4: Parallel execution with fan-out
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When multiple prompts at the same level have no dependencies on each other,
they execute concurrently:

.. code-block:: python

   prompts = [
       {"prompt_name": "topic", "prompt": "Suggest a technology topic"},
       {"prompt_name": "angles", "prompt": "List 3 angles on {{topic.response}}",
        "history": ["topic"]},
       {"prompt_name": "audience", "prompt": "Describe the audience for {{topic.response}}",
        "history": ["topic"]},
       {"prompt_name": "article", "prompt":
           "Write about {{topic.response}}\nAngles: {{angles.response}}\nAudience: {{audience.response}}",
        "history": ["angles", "audience"]},
   ]

   graph_result = await ffai.execute_graph(prompts, max_concurrency=10)

   print(f"Success: {graph_result.success_count}")
   print(f"Failed: {graph_result.failed_count}")

Here ``angles`` and ``audience`` run in parallel at level 1, since neither
depends on the other.

Complete listing
----------------

.. code-block:: python

   import os

   from ffai.Clients import AsyncFFLiteLLMClient
   from ffai.FFAI import FFAI
   from ffai.rag import RAG
   from ffai.rag.embed import Embeddings
   from ffai.rag.store import VectorStore

   client = AsyncFFLiteLLMClient(
       model_string="mistral/mistral-small-latest",
       api_key=os.environ["MISTRAL_API_KEY"],
   )

   # Async RAG
   embed = Embeddings("mistral/mistral-embed", api_key=os.environ["MISTRAL_API_KEY"])
   store = VectorStore(collection_name="async_kb", dir="./async_db")
   rag = RAG(embed=embed, store=store, chunk_size=500, chunk_overlap=100)
   ffai = FFAI(client, rag=rag)

   count = await ffai.aindex("Python is a versatile language.", source="doc1")
   hits = await ffai.asearch("What is Python?", top_k=5)
   result = await ffai.aquery("What is Python?")
   print(result.answer)

   # Async DAG
   prompts = [
       {"prompt_name": "topic", "prompt": "Suggest a topic"},
       {"prompt_name": "outline", "prompt": "Outline about {{topic.response}}",
        "history": ["topic"]},
       {"prompt_name": "article", "prompt": "Article from:\n{{outline.response}}",
        "history": ["outline"]},
   ]
   graph_result = await ffai.execute_graph(prompts, max_concurrency=10)
   for name, r in graph_result.results.items():
       print(f"{name}: {r.status}")

Next steps
----------

- :doc:`dag_execution` — synchronous DAG concepts and graph validation
- :doc:`../guides/configuration` — configure concurrency and timeouts
- :ref:`modindex` — API reference for ``AsyncFFLiteLLMClient`` and ``GraphResult``
