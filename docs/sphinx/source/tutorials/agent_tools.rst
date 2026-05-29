Building an Agent with Tool Calling
====================================

In this tutorial you will build a tool-calling agent that can search for
information, use a calculator, and validate its own responses. By the end you
will understand tool registries, the agent execution loop, and LLM-as-judge
validation.

Prerequisites
-------------

- Python >= 3.10
- ``pip install ffai``
- A Mistral API key (or any LiteLLM-supported provider) set as
  ``MISTRAL_API_KEY``
- Familiarity with the :doc:`quickstart`

Step 1: Register tools
^^^^^^^^^^^^^^^^^^^^^^^

Define the tools your agent can use. Each tool has a name, description, and
JSON Schema for its parameters:

.. code-block:: python

   from ffai.tools.tool_registry import ToolRegistry, ToolDefinition

   registry = ToolRegistry()

   registry.register(ToolDefinition(
       name="search",
       description="Search the web for information",
       parameters={
           "type": "object",
           "properties": {
               "query": {"type": "string", "description": "Search query"},
           },
           "required": ["query"],
       },
   ))

   registry.register(ToolDefinition(
       name="calculator",
       description="Evaluate a mathematical expression",
       parameters={
           "type": "object",
           "properties": {
               "expression": {"type": "string", "description": "Math expression to evaluate"},
           },
           "required": ["expression"],
       },
   ))

Step 2: Provide tool implementations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each tool needs an executor — a callable that receives the tool's arguments as
a dict and returns a string:

.. code-block:: python

   def search_executor(args: dict) -> str:
       query = args["query"]
       return f"Search results for '{query}': Python was created by Guido van Rossum in 1991."

   def calculator_executor(args: dict) -> str:
       try:
           result = eval(args["expression"])  # safe for this demo
           return str(result)
       except Exception as e:
           return f"Error: {e}"

   registry.register_executor("search", search_executor)
   registry.register_executor("calculator", calculator_executor)

Step 3: Create and run the agent loop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``AgentLoop`` manages multi-round tool-call execution. It sends the prompt
to the LLM, dispatches any tool calls the LLM requests, feeds the results back,
and repeats until the LLM provides a final answer or the round limit is hit:

.. code-block:: python

   import os

   from ffai.Clients import FFLiteLLMClient
   from ffai.agent.agent_loop import AgentLoop

   client = FFLiteLLMClient(
       model_string="mistral/mistral-small-latest",
       api_key=os.environ["MISTRAL_API_KEY"],
   )

   loop = AgentLoop(
       client,
       registry,
       max_rounds=5,
       tool_timeout=30.0,
       continue_on_tool_error=True,
   )

   result = loop.execute(
       prompt="Who created Python, and how many years ago was it created?",
       tools=["search", "calculator"],
   )

``execute()`` returns an ``AgentResult``:

.. code-block:: python

   print(result.response)
   print(f"Tool calls: {len(result.tool_calls)}")
   print(f"Rounds: {result.total_rounds}")
   print(f"Status: {result.status}")

Output (``response`` is illustrative; counts are deterministic):

.. code-block:: text

   Python was created by Guido van Rossum in 1991. It was created approximately
   34 years ago (2025 - 1991 = 34).
   Tool calls: 2
   Rounds: 3
   Status: success

Step 4: Inspect tool call records
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Each tool call is recorded with timing and results:

.. code-block:: python

   for tc in result.tool_calls:
       print(f"Round {tc.round}: {tc.tool_name}({tc.arguments})")
       print(f"  Result: {tc.result}")
       print(f"  Duration: {tc.duration_ms:.0f}ms")
       if tc.error:
           print(f"  Error: {tc.error}")

Output (illustrative):

.. code-block:: text

   Round 1: search({'query': 'Who created Python'})
     Result: Search results for 'Who created Python': Python was created by Guido van Rossum in 1991.
     Duration: 150ms
   Round 2: calculator({'expression': '2025 - 1991'})
     Result: 34
     Duration: 12ms

Step 5: Validate responses with LLM-as-judge
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Use ``ResponseValidator`` to check whether a response meets quality criteria.
The validator uses the LLM itself as a judge:

.. code-block:: python

   from ffai.agent.response_validator import ResponseValidator

   validator = ResponseValidator(client)

   validation = validator.validate(
       response=result.response,
       criteria="Response must mention the creator's name and include a calculation",
   )

   print(f"Passed: {validation.passed}")
   print(f"Attempts: {validation.attempts}")

   if not validation.passed:
       print(f"Critique: {validation.critique}")

Output (illustrative):

.. code-block:: text

   Passed: True
   Attempts: 1

``validate()`` returns a ``ValidationResult`` with:

- ``passed`` (``bool`` or ``None``) — whether the response meets the criteria
- ``attempts`` (``int``) — number of validation attempts
- ``critique`` (``str`` or ``None``) — reason for failure (``None`` if passed)

Step 6: Serialize and restore results
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``AgentResult`` supports serialization for logging or storage:

.. code-block:: python

   data = result.to_dict()
   restored = AgentResult.from_dict(data)

   print(restored.response == result.response)
   print(len(restored.tool_calls) == len(result.tool_calls))

Output (deterministic):

.. code-block:: text

   True
   True

Complete listing
----------------

.. code-block:: python

   import os

   from ffai.Clients import FFLiteLLMClient
   from ffai.tools.tool_registry import ToolRegistry, ToolDefinition
   from ffai.agent.agent_loop import AgentLoop
   from ffai.agent.agent_result import AgentResult
   from ffai.agent.response_validator import ResponseValidator

   # Step 1: Register tools
   registry = ToolRegistry()

   registry.register(ToolDefinition(
       name="search",
       description="Search the web for information",
       parameters={
           "type": "object",
           "properties": {
               "query": {"type": "string", "description": "Search query"},
           },
           "required": ["query"],
       },
   ))

   registry.register(ToolDefinition(
       name="calculator",
       description="Evaluate a mathematical expression",
       parameters={
           "type": "object",
           "properties": {
               "expression": {"type": "string", "description": "Math expression"},
           },
           "required": ["expression"],
       },
   ))

   # Step 2: Provide implementations
   registry.register_executor(
       "search",
       lambda args: f"Search results for '{args['query']}': Python was created in 1991.",
   )
   registry.register_executor(
       "calculator",
       lambda args: str(eval(args["expression"])),
   )

   # Step 3: Run agent
   client = FFLiteLLMClient(
       model_string="mistral/mistral-small-latest",
       api_key=os.environ["MISTRAL_API_KEY"],
   )

   loop = AgentLoop(client, registry, max_rounds=5, tool_timeout=30.0)
   result = loop.execute(
       prompt="Who created Python, and how many years ago?",
       tools=["search", "calculator"],
   )

   print(result.response)
   print(f"Tool calls: {len(result.tool_calls)}, Rounds: {result.total_rounds}")

   # Step 5: Validate
   validator = ResponseValidator(client)
   validation = validator.validate(
       response=result.response,
       criteria="Response must mention the creator and include a calculation",
   )
   print(f"Validation: {'passed' if validation.passed else 'failed'}")

   # Step 6: Serialize
   data = result.to_dict()
   restored = AgentResult.from_dict(data)
   assert restored.response == result.response

Next steps
----------

- :doc:`dag_execution` — combine agents with DAG execution for complex workflows
- :doc:`../guides/configuration` — configure tool timeouts and error handling
- :ref:`modindex` — full API reference for ``AgentLoop`` and ``ToolRegistry``
