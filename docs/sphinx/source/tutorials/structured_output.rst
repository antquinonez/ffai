Getting Structured Output from LLMs
====================================

In this tutorial you will use Pydantic models to get typed, validated responses
from LLMs. By the end you will be able to define output schemas, handle
validation errors, and use nested models and enums.

Prerequisites
-------------

- Python >= 3.10
- ``pip install ffai``
- A Mistral API key (or any LiteLLM-supported provider) set as
  ``MISTRAL_API_KEY``
- Familiarity with the :doc:`quickstart`

Step 1: Define a Pydantic output model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a Pydantic ``BaseModel`` that describes the structure you want the LLM
to return:

.. code-block:: python

   from pydantic import BaseModel, Field

   class Sentiment(BaseModel):
       label: str = Field(description="positive, negative, or neutral")
       confidence: float = Field(ge=0.0, le=1.0)

The ``Field`` descriptions help the LLM understand what each field should
contain. Validation constraints like ``ge`` and ``le`` are enforced after the
LLM responds.

Step 2: Generate a structured response
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass your model to ``ResponseOptions`` via the ``response_model`` parameter:

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

   result = ffai.generate_response(
       "The food was amazing but service was slow.",
       options=ResponseOptions(response_model=Sentiment),
   )

   print(result.parsed.label)       # "neutral"
   print(result.parsed.confidence)  # 0.7

Output (``label`` and ``confidence`` are illustrative):

.. code-block:: text

   neutral
   0.7

``result.parsed`` is a validated instance of your Pydantic model. If the LLM
returns invalid data, FFAI automatically retries.

Step 3: Access validation details
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If structured output fails after retries, ``parsing_errors`` contains the
error messages:

.. code-block:: python

   if result.parsing_errors:
       for err in result.parsing_errors:
           print(f"Validation error: {err}")
   else:
       print(f"Success: {result.parsed}")

Output when successful (deterministic):

.. code-block:: text

   Success: label='neutral' confidence=0.7

``parsing_errors`` is ``None`` when parsing succeeds, or a ``list[str]`` of
error messages when it fails.

Step 4: Nested models
^^^^^^^^^^^^^^^^^^^^^^

Use nested Pydantic models for complex output structures:

.. code-block:: python

   class Entity(BaseModel):
       name: str
       entity_type: str = Field(description="person, organization, or location")

   class Extraction(BaseModel):
       entities: list[Entity]
       summary: str

   result = ffai.generate_response(
       "Apple CEO Tim Cook announced new products in Cupertino.",
       options=ResponseOptions(response_model=Extraction),
   )

   print(result.parsed.summary)
   for entity in result.parsed.entities:
       print(f"  {entity.name} ({entity.entity_type})")

Output (illustrative):

.. code-block:: text

   Apple CEO Tim Cook announced new products at their Cupertino headquarters.
     Tim Cook (person)
     Apple (organization)
     Cupertino (location)

Step 5: Enum fields
^^^^^^^^^^^^^^^^^^^^

Use Python ``Enum`` types to constrain the LLM to a fixed set of values:

.. code-block:: python

   from enum import Enum

   class Category(str, Enum):
       sports = "sports"
       politics = "politics"
       tech = "tech"
       science = "science"

   class Classified(BaseModel):
       category: Category
       headline: str
       confidence: float = Field(ge=0.0, le=1.0)

   result = ffai.generate_response(
       "OpenAI releases GPT-5 with breakthrough reasoning capabilities",
       options=ResponseOptions(response_model=Classified),
   )

   print(result.parsed.category)     # "tech"
   print(result.parsed.headline)
   print(result.parsed.confidence)

Output (illustrative):

.. code-block:: text

   tech
   OpenAI releases GPT-5 with breakthrough reasoning capabilities
   0.95

Complete listing
----------------

.. code-block:: python

   import os
   from enum import Enum

   from pydantic import BaseModel, Field

   from ffai.Clients import FFLiteLLMClient
   from ffai.FFAI import FFAI
   from ffai import ResponseOptions

   client = FFLiteLLMClient(
       model_string="mistral/mistral-small-latest",
       api_key=os.environ["MISTRAL_API_KEY"],
   )
   ffai = FFAI(client)

   # Simple model
   class Sentiment(BaseModel):
       label: str = Field(description="positive, negative, or neutral")
       confidence: float = Field(ge=0.0, le=1.0)

   result = ffai.generate_response(
       "The food was amazing but service was slow.",
       options=ResponseOptions(response_model=Sentiment),
   )
   print(f"Sentiment: {result.parsed.label} ({result.parsed.confidence})")

   # Nested model
   class Entity(BaseModel):
       name: str
       entity_type: str = Field(description="person, organization, or location")

   class Extraction(BaseModel):
       entities: list[Entity]
       summary: str

   result = ffai.generate_response(
       "Apple CEO Tim Cook announced new products in Cupertino.",
       options=ResponseOptions(response_model=Extraction),
   )
   print(f"Summary: {result.parsed.summary}")
   for e in result.parsed.entities:
       print(f"  {e.name} ({e.entity_type})")

   # Enum field
   class Category(str, Enum):
       sports = "sports"
       politics = "politics"
       tech = "tech"

   class Classified(BaseModel):
       category: Category
       confidence: float = Field(ge=0.0, le=1.0)

   result = ffai.generate_response(
       "New AI model breaks benchmark records",
       options=ResponseOptions(response_model=Classified),
   )
   print(f"Category: {result.parsed.category} ({result.parsed.confidence})")

Next steps
----------

- :doc:`dag_execution` — combine structured output with DAG pipelines
- :doc:`agent_tools` — use structured output inside agent loops
- :ref:`modindex` — API reference for ``ResponseOptions`` and ``ResponseResult``
