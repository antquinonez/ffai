Building a RAG Pipeline with FFAI
=================================

In this tutorial you will build a document search and question-answering tool
using FFAI's RAG pipeline. By the end you will be able to index documents,
search them with vector similarity, and generate retrieval-augmented answers.

Prerequisites
-------------

- Python >= 3.10
- ``pip install "ffai[rag]"``
- A Mistral API key (or any LiteLLM-supported provider) set as
  ``MISTRAL_API_KEY``

Step 1: Set up the client and RAG pipeline
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create an LLM client, an embedding model, a vector store, and wire them
together through ``RAG`` and ``FFAI``:

.. code-block:: python

   import os

   from ffai.Clients import FFLiteLLMClient
   from ffai.FFAI import FFAI
   from ffai.rag import RAG
   from ffai.rag.embed import Embeddings
   from ffai.rag.store import VectorStore

   client = FFLiteLLMClient(
       model_string="mistral/mistral-small-latest",
       api_key=os.environ["MISTRAL_API_KEY"],
   )

   embed = Embeddings(
       model="mistral/mistral-embed",
       api_key=os.environ["MISTRAL_API_KEY"],
   )

   store = VectorStore(
       collection_name="tutorial_kb",
       dir="./tutorial_db",
   )

   rag = RAG(embed=embed, store=store, chunk_size=500, chunk_overlap=100)
   ffai = FFAI(client, rag=rag)

What each component does:

- **Embeddings** generates vector representations of your text.
- **VectorStore** stores and retrieves those vectors via ChromaDB.
- **RAG** orchestrates chunking, embedding, search, and generation.
- **FFAI** ties the LLM client to the RAG pipeline so you can call
  ``ffai.query()`` directly.

Step 2: Index documents
^^^^^^^^^^^^^^^^^^^^^^^^

Feed documents into the pipeline. Each document gets a ``source`` label so you
can manage it later:

.. code-block:: python

   doc1 = (
       "Python is a high-level programming language known for its readability "
       "and versatility. It supports multiple programming paradigms including "
       "procedural, object-oriented, and functional programming."
   )

   doc2 = (
       "FastAPI is a modern, fast web framework for building APIs with Python. "
       "It is based on standard Python type hints and provides automatic "
       "OpenAPI documentation generation."
   )

   count1 = ffai.index(doc1, source="python_intro")
   count2 = ffai.index(doc2, source="fastapi_intro")

   print(f"Indexed {count1} chunks from python_intro")
   print(f"Indexed {count2} chunks from fastapi_intro")

Output (deterministic):

.. code-block:: text

   Indexed 1 chunks from python_intro
   Indexed 1 chunks from fastapi_intro

The ``chunk_size`` from the ``RAG`` constructor controls how text is split.
``index()`` returns the number of chunks created.

Step 3: Search without generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before calling the LLM, you can search directly. This is a pure vector
similarity search with no generation cost:

.. code-block:: python

   hits = ffai.search("What is Python?", top_k=3)

   for hit in hits:
       print(f"[{hit.score:.2f}] {hit.content[:80]}...")
       print(f"  Source: {hit.source}")

Output (score is illustrative):

.. code-block:: text

   [0.78] Python is a high-level programming language known for its readability ...
     Source: python_intro

``search()`` returns a ``list[SearchHit]``. Each hit has ``score``,
``content``, ``source``, and ``metadata`` fields.

Step 4: Ask questions with retrieval-augmented generation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now combine search with LLM generation. ``query()`` searches for relevant
chunks, formats them as context, and sends the question to the LLM:

.. code-block:: python

   result = ffai.query(
       "What programming paradigms does Python support?"
   )

   print(result.answer)
   print(result.sources)
   print(f"Cost: ${result.cost_usd:.6f}")

Output (``answer`` is illustrative; ``sources`` and ``cost_usd`` type are
deterministic):

.. code-block:: text

   Python supports multiple programming paradigms including procedural,
   object-oriented, and functional programming.
   ['python_intro']
   Cost: $0.000003

``query()`` returns a ``QueryResult`` with:

- ``answer`` (``str``) — the LLM's response
- ``hits`` (``list[SearchHit]``) — the retrieved chunks
- ``sources`` (``list[str]``) — which documents contributed
- ``cost_usd`` (``float``) — estimated cost
- ``duration_ms`` (``float`` or ``None``) — wall-clock time

Step 5: Manage the index
^^^^^^^^^^^^^^^^^^^^^^^^

Check how many chunks are stored, delete documents you no longer need, and
skip re-indexing unchanged documents with a checksum:

.. code-block:: python

   total = ffai.count()
   print(f"Total chunks: {total}")

   ffai.delete("python_intro")
   print(f"After deletion: {ffai.count()} chunks")

   # Skip re-indexing if the document hasn't changed
   ffai.index(doc1, source="python_intro", checksum="abc123")

Output (deterministic):

.. code-block:: text

   Total chunks: 2
   After deletion: 1 chunks

Step 6: Use a custom prompt template
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Override the default prompt to control the LLM's behavior. Your template must
include ``{context}`` and ``{question}`` placeholders:

.. code-block:: python

   result = ffai.query(
       "Compare Python and FastAPI",
       top_k=5,
       prompt_template=(
           "You are a technical advisor. Answer based on this context.\n\n"
           "Context:\n{context}\n\n"
           "Question: {question}\n\n"
           "Provide a structured comparison."
       ),
   )

   print(result.answer)

Output (illustrative):

.. code-block:: text

   Based on the available context:

   **Python** is a high-level programming language supporting multiple paradigms.
   **FastAPI** is a web framework built on Python that provides automatic API
   documentation.

Complete listing
----------------

Here is the full script assembled from all steps:

.. code-block:: python

   import os

   from ffai.Clients import FFLiteLLMClient
   from ffai.FFAI import FFAI
   from ffai.rag import RAG
   from ffai.rag.embed import Embeddings
   from ffai.rag.store import VectorStore

   # Step 1: Set up
   client = FFLiteLLMClient(
       model_string="mistral/mistral-small-latest",
       api_key=os.environ["MISTRAL_API_KEY"],
   )
   embed = Embeddings("mistral/mistral-embed", api_key=os.environ["MISTRAL_API_KEY"])
   store = VectorStore(collection_name="tutorial_kb", dir="./tutorial_db")
   rag = RAG(embed=embed, store=store, chunk_size=500, chunk_overlap=100)
   ffai = FFAI(client, rag=rag)

   # Step 2: Index
   doc1 = (
       "Python is a high-level programming language known for its readability "
       "and versatility. It supports multiple programming paradigms including "
       "procedural, object-oriented, and functional programming."
   )
   doc2 = (
       "FastAPI is a modern, fast web framework for building APIs with Python. "
       "It is based on standard Python type hints and provides automatic "
       "OpenAPI documentation generation."
   )
   ffai.index(doc1, source="python_intro")
   ffai.index(doc2, source="fastapi_intro")

   # Step 3: Search
   hits = ffai.search("What is Python?", top_k=3)
   for hit in hits:
       print(f"[{hit.score:.2f}] {hit.content[:80]}...")

   # Step 4: Query
   result = ffai.query("What programming paradigms does Python support?")
   print(result.answer)
   print(result.sources)

   # Step 5: Manage
   print(f"Total chunks: {ffai.count()}")
   ffai.delete("python_intro")
   print(f"After deletion: {ffai.count()} chunks")

   # Step 6: Custom prompt
   result = ffai.query(
       "Compare Python and FastAPI",
       top_k=5,
       prompt_template=(
           "You are a technical advisor. Answer based on this context.\n\n"
           "Context:\n{context}\n\n"
           "Question: {question}\n\n"
           "Provide a structured comparison."
       ),
   )
   print(result.answer)

Next steps
----------

- :doc:`../guides/rag_search` — BM25 hybrid search, rerankers, query expansion
- :doc:`../guides/chunking` — chunking strategies and when to use each
- :doc:`dag_execution` — multi-step RAG pipelines with DAG execution
