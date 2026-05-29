Quick Start
===========

Install ffai:

.. code-block:: bash

   pip install ffai

Basic usage:

.. code-block:: python

   from ffai.Clients import FFLiteLLMClient
   from ffai.FFAI import FFAI

   client = FFLiteLLMClient(
       model_string="mistral/mistral-small-latest",
       api_key="your-key",
   )

   ffai = FFAI(client)

   result = ffai.generate_response(
       prompt="What is 2+2?",
       prompt_name="math_question"
   )

   print(result.response)       # "2 + 2 equals 4."
   print(result.usage)          # TokenUsage(input_tokens=30, output_tokens=9, ...)
   print(result.cost_usd)       # 3e-06
   print(result.duration_ms)    # 842.3

Named prompt references
-----------------------

Reference earlier responses by name using ``{{prompt_name.response}}`` interpolation:

.. code-block:: python

   ffai.generate_response(
       prompt="What is the capital of France?",
       prompt_name="geography"
   )

   result = ffai.generate_response(
       prompt="Write a poem about {{geography.response}}",
       prompt_name="poem"
   )

Multi-step with dependencies
-----------------------------

.. code-block:: python

   from ffai import ResponseOptions

   ffai.generate_response(
       prompt="List three programming languages",
       prompt_name="languages"
   )

   result = ffai.generate_response(
       "Which of {{languages.response}} is best for beginners?",
       prompt_name="recommendation",
       options=ResponseOptions(dependencies=["languages"]),
   )

Structured output
-----------------

.. code-block:: python

   from pydantic import BaseModel, Field
   from ffai import ResponseOptions

   class Sentiment(BaseModel):
       label: str = Field(description="positive, negative, or neutral")
       confidence: float = Field(ge=0.0, le=1.0)

   result = ffai.generate_response(
       "The food was amazing but service was slow.",
       options=ResponseOptions(response_model=Sentiment),
   )

   print(result.parsed.label)       # "neutral"
   print(result.parsed.confidence)  # 0.7

Fallback models
---------------

.. code-block:: python

   from ffai.Clients import FFLiteLLMClient

   client = FFLiteLLMClient(
       model_string="mistral/mistral-small-latest",
       api_key="your-key",
       fallbacks=["mistral/mistral-medium-latest", "openai/gpt-4o-mini"],
   )

Next steps
----------

- :doc:`guides/installation` — install options and extras
- :doc:`guides/configuration` — YAML config and runtime settings
- :doc:`guides/history` — history views, DataFrame export, and persistence
- :doc:`api` — full API reference
