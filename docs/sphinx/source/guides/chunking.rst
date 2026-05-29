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

Creates parent-child relationships between chunks. Child chunks provide
granularity; parent chunks provide broader context. When a child chunk is
retrieved, its parent content is available via ``parent_content`` on
``SearchHit``:

.. code-block:: python

   from ffai.rag.splitters.hierarchical import HierarchicalChunker

   chunker = HierarchicalChunker(chunk_size=200, chunk_overlap=20)
   chunks = chunker.chunk(long_document)

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
