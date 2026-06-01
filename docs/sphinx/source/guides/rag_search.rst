RAG Search Strategies
=====================

FFAI provides multiple search strategies for retrieving relevant chunks from
your RAG index. This guide covers vector search, BM25 keyword search, hybrid
search with reciprocal rank fusion, rerankers, and query expansion.

Vector search (default)
-----------------------

By default, ``RAG.search()`` uses vector similarity via the configured store
backend (ChromaDB by default):

.. code-block:: python

   hits = rag.search("What is Python?", top_k=5)

   for hit in hits:
       print(f"[{hit.score:.2f}] {hit.content[:80]}")

Vector search finds semantically similar chunks, even when the query uses
different wording than the source text. See :doc:`vector_stores` for available
backends.

BM25 keyword search
-------------------

BM25 (Okapi BM25) finds chunks based on keyword matching. It excels at
exact-term queries, part numbers, names, and code identifiers:

.. code-block:: python

   from ffai.rag.indexing.bm25 import BM25Index

   bm25 = BM25Index()

   bm25.add_document("doc1", "Python is a high-level programming language")
   bm25.add_document("doc2", "FastAPI is a modern web framework for Python")
   bm25.add_document("doc3", "JavaScript is used for web development")

   results = bm25.search("Python programming", n_results=2)

   for r in results:
       print(f"  [{r['score']:.2f}] {r['id']}: {r['content'][:60]}")

Output (deterministic):

.. code-block:: text

   [1.49] doc1: Python is a high-level programming language
   [0.45] doc2: FastAPI is a modern web framework for Python

BM25 returns ``list[dict]`` with ``id``, ``score``, and ``content`` keys.

Hybrid search with bm25_alpha
------------------------------

Combine vector and BM25 search by setting ``bm25_alpha`` on the ``RAG``
constructor. The alpha value controls the blend (0 = pure BM25, 1 = pure
vector):

.. code-block:: python

   rag = RAG(
       embed=embed,
       store=store,
       bm25_alpha=0.5,
   )

``bm25_alpha=0.5`` gives equal weight to both methods. Lower values favor
keyword matches; higher values favor semantic similarity.

Reciprocal rank fusion
----------------------

When combining multiple result lists (e.g., from different search strategies or
query variations), use reciprocal rank fusion (RRF) to merge them:

.. code-block:: python

   from ffai.rag.search.hybrid import reciprocal_rank_fusion

   vector_results = [
       {"id": "doc1", "score": 0.9},
       {"id": "doc2", "score": 0.7},
   ]
   bm25_results = [
       {"id": "doc2", "score": 0.8},
       {"id": "doc1", "score": 0.6},
   ]

   fused = reciprocal_rank_fusion([vector_results, bm25_results], k=60)

   for r in fused:
       print(f"  {r['id']}: rrf_score={r['rrf_score']:.4f}")

Output (deterministic):

.. code-block:: text

   doc1: rrf_score=0.0163
   doc2: rrf_score=0.0163

RRF assigns scores based on rank position rather than raw scores, making it
robust across result lists with different score distributions.

Rerankers
---------

After initial retrieval, rerankers reorder results to improve relevance. Set
the ``reranker`` parameter on ``RAG``:

.. code-block:: python

   rag = RAG(
       embed=embed,
       store=store,
       reranker="diversity",
   )

Available rerankers:

+---------------------+--------------------------------------------------+
| Reranker            | Description                                      |
+=====================+==================================================+
| ``None`` (default)  | No reranking — return results as-is              |
+---------------------+--------------------------------------------------+
| ``"diversity"``     | Promotes diverse results, penalizes redundancy   |
+---------------------+--------------------------------------------------+
| ``DiversityReranker`` | Same as ``"diversity"``, instantiated directly  |
+---------------------+--------------------------------------------------+

Query expansion
---------------

Expand a single query into multiple variations to improve recall:

.. code-block:: python

   from ffai.rag.search.query_expansion import QueryExpander

   expander = QueryExpander(client)

   variations = expander.expand("How does Python handle memory?")
   # ['How does Python handle memory?',
   #  'Python memory management garbage collection',
   #  'memory allocation in CPython interpreter']

``QueryExpander`` uses the LLM to generate alternative phrasings. Pass it to
``RAG`` via the ``query_expander`` parameter:

.. code-block:: python

   rag = RAG(
       embed=embed,
       store=store,
       query_expander=expander.expand,
   )

When set, ``rag.search()`` and ``rag.query()`` automatically expand the query
and fuse the results.

Fuse results from multiple searches
-----------------------------------

Combine results from different queries or strategies with deduplication:

.. code-block:: python

   from ffai.rag.search.query_expansion import fuse_search_results

   results_a = [{"id": "a", "content": "doc a", "score": 0.9}]
   results_b = [{"id": "b", "content": "doc b", "score": 0.8}]

   fused = fuse_search_results(
       [results_a, results_b],
       n_results=5,
       dedupe_by="id",
   )

``fuse_search_results`` removes duplicates based on the ``dedupe_by`` field
while preserving relevance ordering.

See also
--------

- :doc:`../tutorials/rag_pipeline` — end-to-end RAG tutorial
- :doc:`chunking` — choosing a chunking strategy
- :ref:`modindex` — full API reference
