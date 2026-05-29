Conditional Execution with the Condition DSL
==============================================

In this tutorial you will use FFAI's condition DSL to skip or abort prompts
based on the results of earlier steps. By the end you will understand
``condition``, ``abort_condition``, and the expression language.

Prerequisites
-------------

- Python >= 3.10
- ``pip install ffai``
- A Mistral API key (or any LiteLLM-supported provider) set as
  ``MISTRAL_API_KEY``
- Familiarity with the :doc:`quickstart`

Step 1: The condition parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``condition`` in ``ResponseOptions`` to skip a prompt when the expression
evaluates to ``False``:

.. code-block:: python

   import os

   from ffai.Clients import FFLiteLLMClient
   from ffai.FFAI import FFAI
   from ffai import ResponseOptions

   client = FFLiteLLMClient(
       model_string="mistral/mistral-small-latest",
       api_key=os.environ["MISTRAL_API_KEY"],
   )
   ffai = FFAI(client)

   ffai.generate_response("List three languages", prompt_name="languages")

   result = ffai.generate_response(
       "Which is easiest?",
       prompt_name="recommendation",
       options=ResponseOptions(
           condition="len({{languages.response}}) > 0",
           dependencies=["languages"],
       ),
   )

   print(result.status)           # "success"
   print(result.condition_trace)  # None (only set when condition is False)

When the condition is ``True``, the prompt runs normally and ``status`` is
``"success"``. ``condition_trace`` is ``None``.

Step 2: Skipped prompts
^^^^^^^^^^^^^^^^^^^^^^^^

When a condition evaluates to ``False``, the prompt is skipped:

.. code-block:: python

   result = ffai.generate_response(
       "Which is easiest?",
       prompt_name="skipped_rec",
       options=ResponseOptions(
           condition="len({{languages.response}}) > 99999",
           dependencies=["languages"],
       ),
   )

   print(result.status)           # "skipped"
   print(result.condition_trace)  # 'len("Python, JavaScript, Rust") > 99999'

``status`` is ``"skipped"`` and ``condition_trace`` shows the resolved
expression with ``{{languages.response}}`` replaced by the actual value.

Step 3: The abort_condition parameter
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``abort_condition`` to halt the entire pipeline when a condition is met.
This is useful for stopping DAG execution when an earlier step produces
undesirable output:

.. code-block:: python

   result = ffai.generate_response(
       "Analyze the text",
       prompt_name="analysis",
       options=ResponseOptions(
           abort_condition='"error" in lower({{languages.response}})',
           dependencies=["languages"],
       ),
   )

If the abort condition is ``True``, ``status`` is ``"failed"`` and subsequent
dependent prompts in a DAG are not executed.

Step 4: Expression language reference
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The condition DSL uses AST-based safe evaluation (no ``eval()``). It supports:

**Comparisons:**

.. code-block:: python

   len({{languages.response}}) > 0
   {{analysis.status}} == "success"
   int({{count.response}}) >= 5

**Boolean logic:**

.. code-block:: python

   {{languages.status}} == "success" and not is_empty({{languages.response}})
   len({{items.response}}) > 0 or {{fallback.status}} == "success"

**String operations:**

.. code-block:: python

   "error" in lower({{result.response}})
   trim({{raw.response}}) != ""
   "Python" in {{languages.response}}

**JSON navigation:**

.. code-block:: python

   json_get({{analysis.response}}, "sentiment") == "positive"
   json_has({{data.response}}, "items")

**Built-in functions:** ``len``, ``int``, ``float``, ``str``, ``bool``,
``abs``, ``min``, ``max``, ``round``, ``lower``, ``upper``, ``trim``,
``strip``, ``split``, ``replace``, ``count``, ``is_null``, ``is_empty``,
``json_get``, ``json_has``, ``json_keys``, ``json_values``, ``json_type``.

**Regex matching** (via ``%`` operator):

.. code-block:: python

   {{result.response}} % r"\d{4}"

Step 5: Conditions in DAG execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conditions are most powerful in DAG pipelines. Prompts that depend on a skipped
prompt are also skipped:

.. code-block:: python

   prompts = [
       {"prompt_name": "check", "prompt": "Check if data is valid"},
       {
           "prompt_name": "process",
           "prompt": "Process: {{check.response}}",
           "condition": 'not is_empty({{check.response}})',
           "history": ["check"],
       },
       {
           "prompt_name": "report",
           "prompt": "Report: {{process.response}}",
           "history": ["process"],
       },
   ]

   graph, _ = ffai.validate_graph(prompts)
   graph_result = await ffai.execute_graph(prompts)

If ``check`` returns an empty response, ``process`` is skipped, and ``report``
is also skipped because its dependency was not fulfilled.

Complete listing
----------------

.. code-block:: python

   import os

   from ffai.Clients import FFLiteLLMClient
   from ffai.FFAI import FFAI
   from ffai import ResponseOptions

   client = FFLiteLLMClient(
       model_string="mistral/mistral-small-latest",
       api_key=os.environ["MISTRAL_API_KEY"],
   )
   ffai = FFAI(client)

   # Generate initial data
   ffai.generate_response("List three programming languages", prompt_name="languages")

   # Condition passes
   result = ffai.generate_response(
       "Which is easiest?",
       prompt_name="recommendation",
       options=ResponseOptions(
           condition="len({{languages.response}}) > 0",
           dependencies=["languages"],
       ),
   )
   print(f"Status: {result.status}")
   print(f"Trace: {result.condition_trace}")

   # Condition fails (skipped)
   result = ffai.generate_response(
       "Which is hardest?",
       prompt_name="skipped",
       options=ResponseOptions(
           condition="len({{languages.response}}) > 99999",
           dependencies=["languages"],
       ),
   )
   print(f"Status: {result.status}")
   print(f"Trace: {result.condition_trace}")

Next steps
----------

- :doc:`dag_execution` — full DAG execution with conditions
- :ref:`modindex` — API reference for ``ConditionEvaluator`` and ``ResponseOptions``
