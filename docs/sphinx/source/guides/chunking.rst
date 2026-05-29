Chunking Strategies
===================

Before documents can be embedded and indexed for RAG, they must be split into
chunks. FFAI provides five chunking strategies, each suited to different
content types.

Choosing a strategy
-------------------

+-------------------+-------------------------------+----------------------------------+
| Strategy          | Best for                      | Use when                         |
+===================+===============================+==================================+
| ``recursive``     | General prose, mixed content  | Default choice for most documents |
+-------------------+-------------------------------+----------------------------------+
| ``character``     | Simple fixed-size splitting   | You need uniform chunk sizes     |
+-------------------+-------------------------------+----------------------------------+
| ``markdown``      | Markdown documents            | Content has headings/sections    |
+-------------------+-------------------------------+----------------------------------+
| ``code``          | Source code                   | Indexing codebases               |
+-------------------+-------------------------------+----------------------------------+
| ``hierarchical``  | Long documents with structure | You need parent-child context    |
+-------------------+-------------------------------+----------------------------------+

List all available strategies:

.. code-block:: python

   from ffai.rag.splitters import list_chunkers

   print(list_chunkers())
   # ['character', 'recursive', 'markdown', 'code', 'hierarchical']

Getting a chunker
-----------------

Use ``get_chunker()`` to instantiate a chunker by name:

.. code-block:: python

   from ffai.rag.splitters import get_chunker

   chunker = get_chunker("recursive", chunk_size=500, chunk_overlap=100)

Or instantiate directly:

.. code-block:: python

   from ffai.rag.splitters.recursive import RecursiveChunker

   chunker = RecursiveChunker(chunk_size=500, chunk_overlap=100)

All chunkers return ``list[TextChunk]``, where each ``TextChunk`` has:

- ``content`` (``str``) — the chunk text
- ``chunk_index`` (``int``) — position in the document
- ``start_char`` (``int``) — start position in the original text
- ``end_char`` (``int``) — end position in the original text
- ``metadata`` (``dict`` or ``None``) — optional metadata

Recursive chunker (default)
---------------------------

Splits text at paragraph boundaries first, then sentences, then characters.
Preserves natural text boundaries:

.. code-block:: python

   from ffai.rag.splitters.recursive import RecursiveChunker

   chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)

   text = "Python is a programming language. " * 20
   chunks = chunker.chunk(text)

   for c in chunks[:3]:
       print(f"chunk {c.chunk_index}: chars {c.start_char}-{c.end_char}, len={len(c.content)}")

Output (deterministic):

.. code-block:: text

   chunk 0: chars 0-100, len=100
   chunk 1: chars 92-170, len=77
   chunk 2: chars 160-238, len=77

Character chunker
-----------------

Splits at fixed character counts. Simple and predictable, but may break
mid-word or mid-sentence:

.. code-block:: python

   from ffai.rag.splitters.character import CharacterChunker

   chunker = CharacterChunker(chunk_size=200, chunk_overlap=50)
   chunks = chunker.chunk(text)

Markdown chunker
----------------

Respects Markdown heading structure. Each chunk starts with its heading
context, preserving the document hierarchy:

.. code-block:: python

   from ffai.rag.splitters.markdown import MarkdownChunker

   md = """# Introduction

Python is versatile.

## Setup

Install with pip.

## Usage

Import and initialize."""

   chunker = MarkdownChunker(chunk_size=500, chunk_overlap=50)
   chunks = chunker.chunk(md)

   for c in chunks:
       print(f"chunk {c.chunk_index}: {c.content[:50]}...")

Code chunker
------------

Splits source code at function and class boundaries:

.. code-block:: python

   from ffai.rag.splitters.code import CodeChunker

   code = """
   def hello():
       return "hello"

   def goodbye():
       return "goodbye"

   class Greeter:
       def greet(self):
           return "hi"
   """

   chunker = CodeChunker(chunk_size=500, chunk_overlap=50)
   chunks = chunker.chunk(code)

Hierarchical chunker
--------------------

Creates parent-child relationships between chunks. Parent chunks cover a
larger span of the document; child chunks are smaller and more specific.
This is useful for long documents where a search hit on a specific paragraph
should also return the surrounding section for context.

**How it works during indexing:**

1. ``HierarchicalChunker`` splits text into parent chunks (``parent_chunk_size``)
   and then subdivides each parent into child chunks (``chunk_size``).
2. ``RAG.aindex()`` detects the hierarchical chunks and stores **only child
   chunks** in the vector store. Each child's metadata includes
   ``parent_content`` — the full text of its parent section.
3. Parent chunks are never embedded or stored directly, avoiding redundancy.

**How it works during search:**

1. The query matches child chunks by vector similarity.
2. Each ``SearchHit`` has ``parent_content`` populated from metadata.
3. ``format_hits()`` includes a ``[Parent context: ...]`` snippet when parent
   content is available.
4. ``FFAI.query()`` sends both the child content and parent context to the LLM.

.. code-block:: python

   from ffai.rag import RAG

   rag = RAG(
       embed=embed,
       store=store,
       chunker="hierarchical",
       chunk_size=200,
       chunk_overlap=20,
       parent_chunk_size=1500,
   )

   rag.index(contract_text, source="contract")
   hits = rag.search("termination conditions")

   for hit in hits:
       print(f"Child: {hit.content[:80]}...")
       print(f"Parent: {hit.parent_content[:80]}...")

Output (illustrative):

.. code-block:: text

   Child: The agreement may be terminated by either party with 30 days notice...
   Parent: Section 5. Termination. This section covers termination conditions, notice...

Constructor parameters:

- ``chunk_size`` — maximum characters per child chunk (default: 400)
- ``chunk_overlap`` — overlap between child chunks (default: 100)
- ``parent_chunk_size`` — maximum characters per parent chunk (default: 1500)
- ``max_levels`` — hierarchy depth, currently only 2 levels (default: 2)

Using with RAG
--------------

Set the chunker when creating a ``RAG`` instance:

.. code-block:: python

   from ffai.rag import RAG

   rag = RAG(
       embed=embed,
       store=store,
       chunker="recursive",
       chunk_size=500,
       chunk_overlap=100,
   )

The ``chunker`` parameter accepts a strategy name (``"recursive"``,
``"markdown"``, etc.) or a ``ChunkerBase`` instance for custom configuration.

Tuning chunk_size and chunk_overlap
------------------------------------

- **chunk_size**: Larger chunks capture more context but may dilute relevance.
  Start with 500 for prose, 200-300 for code.
- **chunk_overlap**: Overlap prevents losing information at chunk boundaries.
  10-20% of ``chunk_size`` is a good starting point.

See also
--------

- :doc:`../tutorials/rag_pipeline` — end-to-end RAG tutorial
- :doc:`rag_search` — search strategies for retrieving chunks
- :ref:`modindex` — full API reference
