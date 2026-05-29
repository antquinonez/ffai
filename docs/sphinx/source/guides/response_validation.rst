Response Validation
====================

FFAI can validate LLM responses using a second LLM call as a judge. This guide
covers the ``ResponseValidator`` and how to re-execute when validation fails.

Basic validation
----------------

Pass a response and quality criteria to ``validate()``:

.. code-block:: python

   from ffai.agent.response_validator import ResponseValidator

   validator = ResponseValidator(client)

   result = validator.validate(
       response=some_response,
       criteria="Response must mention at least 3 key points and include a summary",
   )

   print(result.passed)    # True or False
   print(result.attempts)  # 1 (number of validation attempts)
   print(result.critique)  # None if passed, reason if failed

``ValidationResult`` fields:

- ``passed`` (``bool`` or ``None``) — whether the response meets the criteria
- ``attempts`` (``int``) — number of validation attempts
- ``critique`` (``str`` or ``None``) — reason for failure (``None`` if passed)

Re-execution on failure
------------------------

When validation fails, re-execute with the critique as feedback:

.. code-block:: python

   if not result.passed:
       new_result = ffai.generate_response(
           f"Previous attempt was rejected: {result.critique}\n\n"
           f"Original prompt: Write a summary of the document",
           prompt_name="summary_retry",
       )

This pattern gives the LLM a second chance with specific feedback on what was
wrong.

Validation with retries
-----------------------

``validate()`` accepts a ``max_retries`` parameter. When combined with a
re-execution function, it automatically retries until validation passes:

.. code-block:: python

   def regenerate(prompt: str) -> str:
       result = ffai.generate_response(prompt, prompt_name="regen")
       return result.response

   result = validator.validate(
       response=initial_response,
       criteria="Response must be under 100 words",
       max_retries=2,
       re_execute_fn=regenerate,
   )

Writing good criteria
---------------------

Be specific and measurable:

- **Good**: "Response must list at least 3 items and include a conclusion"
- **Good**: "Response must not contain any code examples"
- **Bad**: "Response should be good"
- **Bad**: "Make it better"

Criteria are passed directly to the judge LLM. Vague criteria produce
unreliable validation results.

See also
--------

- :doc:`../tutorials/agent_tools` — agent tutorial with validation
- :ref:`modindex` — API reference for ``ResponseValidator`` and ``ValidationResult``
