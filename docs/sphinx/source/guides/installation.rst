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

Requirements
------------

- Python >= 3.10
