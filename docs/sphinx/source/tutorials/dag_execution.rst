Building a DAG Execution Pipeline with FFAI
============================================

In this tutorial you will build a multi-step research pipeline using FFAI's
DAG (directed acyclic graph) execution. By the end you will be able to define
prompt dependency graphs, validate them, and execute prompts in parallel where
possible.

Prerequisites
-------------

- Python >= 3.10
- ``pip install ffai``
- A Mistral API key (or any LiteLLM-supported provider) set as
  ``MISTRAL_API_KEY``
- Familiarity with the :doc:`quickstart` and basic ``FFAI`` usage

Step 1: Set up the async client
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

DAG execution requires an async client. The ``execute_graph`` method runs
prompts in topological-parallel order using ``asyncio.gather``:

.. code-block:: python

   import os

   from ffai.Clients import AsyncFFLiteLLMClient
   from ffai.FFAI import FFAI

   client = AsyncFFLiteLLMClient(
       model_string="mistral/mistral-small-latest",
       api_key=os.environ["MISTRAL_API_KEY"],
   )

   ffai = FFAI(client)

Step 2: Define a prompt graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Define prompts as a list of dicts. Each prompt has a ``prompt_name`` and a
``prompt``. Use the ``history`` key to declare dependencies — FFAI will ensure
that dependent prompts run after their prerequisites:

.. code-block:: python

   prompts = [
       {
           "prompt_name": "topic",
           "prompt": "Suggest a topic for a blog post about technology",
       },
       {
           "prompt_name": "outline",
           "prompt": "Create a detailed outline for a blog post about {{topic.response}}",
           "history": ["topic"],
       },
       {
           "prompt_name": "article",
           "prompt": "Write a 3-paragraph blog post based on this outline:\n{{outline.response}}",
           "history": ["outline"],
       },
   ]

The ``{{topic.response}}`` and ``{{outline.response}}`` placeholders are
resolved at execution time — each prompt receives the output of the prompts it
depends on.

Step 3: Validate the graph
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before executing, validate the graph to check for cycles, missing dependencies,
and other issues:

.. code-block:: python

   graph, warnings = ffai.validate_graph(prompts)

   print(f"Graph has {len(graph.nodes)} nodes, {len(graph.edges)} edges")
   print(f"Max level: {graph.max_level}")

   if warnings:
       for w in warnings:
           print(f"Warning: {w}")

Output (deterministic):

.. code-block:: text

   Graph has 3 nodes, 2 edges
   Max level: 2

The ``max_level`` indicates the depth of the dependency chain (0 = root, 1 =
depends on root, etc.). Prompts at the same level run in parallel.

Step 4: Execute the graph
^^^^^^^^^^^^^^^^^^^^^^^^^^

Execute the full pipeline. ``execute_graph`` requires an ``AsyncFFLiteLLMClient``
and runs prompts level by level:

.. code-block:: python

   graph_result = await ffai.execute_graph(prompts, max_concurrency=10)

   for name, r in graph_result.results.items():
       print(f"{name}: {r.status} ({r.duration_ms:.0f}ms)")

Output (illustrative):

.. code-block:: text

   topic: success (842ms)
   outline: success (1203ms)
   article: success (2156ms)

``execute_graph`` returns a ``GraphResult`` with:

- ``results`` (``dict[str, ResponseResult]``) — one result per prompt
- ``success_count`` (``int``) — how many succeeded
- ``failed_count`` (``int``) — how many failed
- ``skipped_count`` (``int``) — how many were skipped by conditions
- ``aborted`` (``bool``) — whether an abort condition was triggered

Step 5: Add parallel branches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When prompts at the same level have no dependencies on each other, they run in
parallel. Here, ``angles`` and ``audience`` both depend on ``topic`` but not on
each other:

.. code-block:: python

   prompts = [
       {
           "prompt_name": "topic",
           "prompt": "Suggest a topic for a blog post about technology",
       },
       {
           "prompt_name": "angles",
           "prompt": "List 3 unique angles on {{topic.response}}",
           "history": ["topic"],
       },
       {
           "prompt_name": "audience",
           "prompt": "Describe the target audience for {{topic.response}}",
           "history": ["topic"],
       },
       {
           "prompt_name": "article",
           "prompt": (
               "Write a blog post about {{topic.response}}\n"
               "Angles: {{angles.response}}\n"
               "Audience: {{audience.response}}"
           ),
           "history": ["angles", "audience"],
       },
   ]

   graph, _ = ffai.validate_graph(prompts)
   print(f"Nodes: {len(graph.nodes)}, Edges: {len(graph.edges)}, Levels: {graph.max_level}")

Output (deterministic):

.. code-block:: text

   Nodes: 4, Edges: 4, Levels: 2

``angles`` and ``audience`` run in parallel at level 1. ``article`` waits for
both at level 2.

Complete listing
----------------

.. code-block:: python

   import os

   from ffai.Clients import AsyncFFLiteLLMClient
   from ffai.FFAI import FFAI

   client = AsyncFFLiteLLMClient(
       model_string="mistral/mistral-small-latest",
       api_key=os.environ["MISTRAL_API_KEY"],
   )
   ffai = FFAI(client)

   # Define the prompt graph
   prompts = [
       {
           "prompt_name": "topic",
           "prompt": "Suggest a topic for a blog post about technology",
       },
       {
           "prompt_name": "outline",
           "prompt": "Create a detailed outline for a blog post about {{topic.response}}",
           "history": ["topic"],
       },
       {
           "prompt_name": "article",
           "prompt": "Write a 3-paragraph blog post based on this outline:\n{{outline.response}}",
           "history": ["outline"],
       },
   ]

   # Validate
   graph, warnings = ffai.validate_graph(prompts)
   print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")
   if warnings:
       for w in warnings:
           print(f"Warning: {w}")

   # Execute
   graph_result = await ffai.execute_graph(prompts, max_concurrency=10)
   for name, r in graph_result.results.items():
       print(f"{name}: {r.status} ({r.duration_ms:.0f}ms)")

   print(f"\nSuccess: {graph_result.success_count}")
   print(f"Failed: {graph_result.failed_count}")

Next steps
----------

- :doc:`agent_tools` — add tool-calling to your pipelines
- :doc:`../guides/configuration` — configure retries and observability
- :ref:`modindex` — full API reference for ``ExecutionGraph`` and ``GraphResult``
