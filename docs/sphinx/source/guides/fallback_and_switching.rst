Fallback Models & Client Switching
===================================

FFAI supports fallback model chains and runtime client switching so your
application can continue working even when a provider has an outage.

Fallback models
---------------

Configure fallback models on the LiteLLM client. If the primary model fails,
alternatives are tried in order:

.. code-block:: python

   from ffai.Clients import FFLiteLLMClient

   client = FFLiteLLMClient(
       model_string="mistral/mistral-small-latest",
       api_key="your-key",
       fallbacks=["mistral/mistral-medium-latest", "openai/gpt-4o-mini"],
   )

When ``mistral-small-latest`` fails (rate limit, server error, timeout), the
client automatically tries ``mistral-medium-latest``, then
``openai/gpt-4o-mini``. The response includes which model actually served the
request:

.. code-block:: python

   result = ffai.generate_response("Hello", prompt_name="greeting")
   print(result.model)
   # "mistral/mistral-small-latest" (or whichever succeeded)

Per-call model override
-----------------------

Override the model for a single call using ``ResponseOptions``:

.. code-block:: python

   from ffai import ResponseOptions

   result = ffai.generate_response(
       "Translate to French: Hello",
       prompt_name="translate",
       options=ResponseOptions(model="mistral/mistral-large-latest"),
   )

   print(result.model)
   # "mistral/mistral-large-latest"

This does not change the client's default model — only this call.

Runtime client switching
------------------------

Switch the underlying client at runtime without creating a new ``FFAI``
instance. History is preserved:

.. code-block:: python

   from ffai.Clients import FFLiteLLMClient

   new_client = FFLiteLLMClient(
       model_string="openai/gpt-4o",
       api_key="your-key",
   )

   ffai.set_client(new_client)

Subsequent calls use the new client. The existing history and RAG configuration
remain intact.

Use cases:

- **Provider failover** — switch from Mistral to OpenAI during an outage
- **Model tiering** — start with a fast/cheap model, switch to a powerful one
  for complex tasks
- **Testing** — swap between real and mock clients without rebuilding state

See also
--------

- :doc:`../tutorials/dag_execution` — DAG pipelines with model overrides
- :doc:`cost_tracking` — tracking costs across providers and models
