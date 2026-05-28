History & Query API
====================

FFAI maintains four parallel history stores:

+---------------------+-----------------------------------------+----------------------------------------------+
| Store               | Access                                  | Description                                  |
+=====================+=========================================+==============================================+
| ``history``         | ``ffai.history``                        | Raw interaction list                          |
+---------------------+-----------------------------------------+----------------------------------------------+
| ``clean_history``   | ``ffai.clean_history``                  | Cleaned interaction list                      |
+---------------------+-----------------------------------------+----------------------------------------------+
| ``ordered_history`` | ``ffai.ordered_history``                | OrderedPromptHistory with sequence numbers    |
+---------------------+-----------------------------------------+----------------------------------------------+
| ``permanent_history`` | ``ffai.permanent_history``            | PermanentHistory with timestamps, incremental |
+---------------------+-----------------------------------------+----------------------------------------------+

Query methods
-------------

.. code-block:: python

   ffai.get_all_interactions()
   ffai.get_latest_interaction()
   ffai.get_latest_interaction_by_prompt_name("analysis")
   ffai.get_last_n_interactions(5)
   ffai.get_interaction(sequence_number=3)
   ffai.get_model_interactions("mistral-small-latest")
   ffai.get_interactions_by_prompt_name("summary")
   ffai.get_prompt_history()
   ffai.get_response_history()
   ffai.get_model_usage_stats()
   ffai.get_prompt_name_usage_stats()
   ffai.get_prompt_dict()
   ffai.get_latest_responses_by_prompt_names(["a", "b"])
   ffai.get_formatted_responses(["analysis", "summary"])

DataFrame export
-----------------

.. code-block:: python

   df = ffai.history_to_dataframe()
   df = ffai.clean_history_to_dataframe()
   df = ffai.ordered_history_to_dataframe()
   df = ffai.search_history(text="error", model="gpt-4o")
   df = ffai.get_model_stats_df()
   df = ffai.get_prompt_name_stats_df()
   df = ffai.get_response_length_stats()
   df = ffai.interaction_counts_by_date()

Persistence
-----------

.. code-block:: python

   ffai.persist_all_histories()  # write Parquet to configured directory

Client conversation history
---------------------------

.. code-block:: python

   ffai.get_client_conversation_history()
   ffai.set_client_conversation_history(history)
   ffai.add_client_message("user", "Hello")
   ffai.clear_conversation()
