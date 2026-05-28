from ffai.ConversationHistory import ConversationHistory


class TestConversationHistoryInit:
    def test_starts_empty(self):
        ch = ConversationHistory()
        assert ch.turns == []


class TestAddTurnAssistant:
    def test_single_assistant_turn(self):
        ch = ConversationHistory()
        ch.add_turn_assistant("Hello there")
        assert len(ch.turns) == 1
        assert ch.turns[0]["role"] == "assistant"
        assert ch.turns[0]["content"][0]["type"] == "text"
        assert ch.turns[0]["content"][0]["text"] == "Hello there"

    def test_multiple_assistant_turns(self):
        ch = ConversationHistory()
        ch.add_turn_assistant("First")
        ch.add_turn_assistant("Second")
        assert len(ch.turns) == 2
        assert ch.turns[0]["content"][0]["text"] == "First"
        assert ch.turns[1]["content"][0]["text"] == "Second"


class TestAddTurnUser:
    def test_single_user_turn(self):
        ch = ConversationHistory()
        ch.add_turn_user("Hi")
        assert len(ch.turns) == 1
        assert ch.turns[0]["role"] == "user"
        assert ch.turns[0]["content"][0]["text"] == "Hi"

    def test_consecutive_user_turns_merge(self):
        ch = ConversationHistory()
        ch.add_turn_user("Line one")
        ch.add_turn_user("Line two")
        assert len(ch.turns) == 1
        assert ch.turns[0]["content"][0]["text"] == "Line one\nLine two"

    def test_user_after_assistant_starts_new_turn(self):
        ch = ConversationHistory()
        ch.add_turn_assistant("Reply")
        ch.add_turn_user("Follow-up")
        assert len(ch.turns) == 2
        assert ch.turns[0]["role"] == "assistant"
        assert ch.turns[1]["role"] == "user"
        assert ch.turns[1]["content"][0]["text"] == "Follow-up"


class TestGetTurns:
    def test_empty_returns_empty_list(self):
        ch = ConversationHistory()
        assert ch.get_turns() == []

    def test_returns_copy_not_reference(self):
        ch = ConversationHistory()
        ch.add_turn_user("Hello")
        turns = ch.get_turns()
        turns[0]["content"][0]["text"] = "modified"
        assert ch.turns[0]["content"][0]["text"] == "Hello"

    def test_user_turns_are_deep_copied(self):
        ch = ConversationHistory()
        ch.add_turn_user("Original")
        turns = ch.get_turns()
        turns[0]["content"].append({"type": "text", "text": "extra"})
        assert len(ch.turns[0]["content"]) == 1

    def test_assistant_turns_are_same_object(self):
        ch = ConversationHistory()
        ch.add_turn_assistant("Response")
        turns = ch.get_turns()
        assert turns[0] is ch.turns[0]

    def test_roundtrip_preserves_content(self):
        ch = ConversationHistory()
        ch.add_turn_user("Question")
        ch.add_turn_assistant("Answer")
        ch.add_turn_user("Follow-up")
        turns = ch.get_turns()
        assert len(turns) == 3
        assert turns[0]["role"] == "user"
        assert turns[0]["content"][0]["text"] == "Question"
        assert turns[1]["role"] == "assistant"
        assert turns[1]["content"][0]["text"] == "Answer"
        assert turns[2]["role"] == "user"
        assert turns[2]["content"][0]["text"] == "Follow-up"


class TestShimImport:
    def test_import_from_shim_module(self):
        from ffai.core.history.conversation import ConversationHistory as Direct

        assert ConversationHistory is Direct


from ffai.core.history.ordered import OrderedPromptHistory  # noqa: E402


class TestOrderedPromptHistoryTuplePromptName:
    def test_single_element_tuple_collapses_to_string(self):
        oph = OrderedPromptHistory()
        oph.add_interaction(
            model="gpt-4",
            prompt="hello",
            response="world",
            prompt_name=(("my_prompt", {}),),  # type: ignore[arg-type]
        )
        names = oph.get_all_prompt_names()
        assert names == ["my_prompt"]

    def test_multi_element_tuple_stringified(self):
        oph = OrderedPromptHistory()
        oph.add_interaction(
            model="gpt-4",
            prompt="hello",
            response="world",
            prompt_name=(("alpha", {}), ("beta", {})),  # type: ignore[arg-type]
        )
        names = oph.get_all_prompt_names()
        assert len(names) == 1
        assert names[0] == "('alpha', 'beta')"


class TestOrderedPromptHistoryMergeHistories:
    def test_merge_combines_and_resequences(self):
        oph1 = OrderedPromptHistory()
        oph1.add_interaction(model="m1", prompt="p1", response="r1", prompt_name="a")
        oph2 = OrderedPromptHistory()
        oph2.add_interaction(model="m2", prompt="p2", response="r2", prompt_name="b")
        oph1.merge_histories(oph2)
        all_interactions = oph1.get_all_interactions()
        assert len(all_interactions) == 2
        assert all_interactions[0].sequence_number < all_interactions[1].sequence_number

    def test_merge_preserves_prompt_names(self):
        oph1 = OrderedPromptHistory()
        oph1.add_interaction(model="m1", prompt="p1", response="r1", prompt_name="a")
        oph2 = OrderedPromptHistory()
        oph2.add_interaction(model="m2", prompt="p2", response="r2", prompt_name="b")
        oph1.merge_histories(oph2)
        assert "a" in oph1.get_all_prompt_names()
        assert "b" in oph1.get_all_prompt_names()


class TestOrderedPromptHistoryGetByModelAndPromptName:
    def test_filters_by_model_and_prompt_name(self):
        oph = OrderedPromptHistory()
        oph.add_interaction(model="gpt-4", prompt="p", response="r1", prompt_name="x")
        oph.add_interaction(model="claude", prompt="p", response="r2", prompt_name="x")
        result = oph.get_interactions_by_model_and_prompt_name("gpt-4", "x")
        assert len(result) == 1
        assert result[0].model == "gpt-4"

    def test_no_match_returns_empty(self):
        oph = OrderedPromptHistory()
        oph.add_interaction(model="gpt-4", prompt="p", response="r", prompt_name="x")
        assert oph.get_interactions_by_model_and_prompt_name("gpt-4", "y") == []


class TestOrderedPromptHistoryGetInteractionByPrompt:
    def test_returns_interaction_by_prompt_text(self):
        oph = OrderedPromptHistory()
        oph.add_interaction(model="m", prompt="find me", response="answer", prompt_name=None)
        result = oph.get_interaction_by_prompt("find me")
        assert result is not None
        assert result.response == "answer"

    def test_no_match_returns_none(self):
        oph = OrderedPromptHistory()
        assert oph.get_interaction_by_prompt("nonexistent") is None


class TestOrderedPromptHistoryGetFormattedResponsesCycle:
    def test_cycle_prevention(self):
        oph = OrderedPromptHistory()
        oph.add_interaction(
            model="m",
            prompt="pa",
            response="ra",
            prompt_name="a",
            history=["b"],
        )
        oph.add_interaction(
            model="m",
            prompt="pb",
            response="rb",
            prompt_name="b",
            history=["a"],
        )
        result = oph.get_formatted_responses(["a"])
        assert "ra" in result
        assert "rb" in result


class TestOrderedPromptHistoryGetFormattedResponsesHistoryChain:
    def test_history_chain_resolves_recursively(self):
        oph = OrderedPromptHistory()
        oph.add_interaction(
            model="m",
            prompt="pb",
            response="rb",
            prompt_name="B",
        )
        oph.add_interaction(
            model="m",
            prompt="pa",
            response="ra",
            prompt_name="A",
            history=["B"],
        )
        result = oph.get_formatted_responses(["A"])
        rb_pos = result.index("rb")
        ra_pos = result.index("ra")
        assert rb_pos < ra_pos


class TestOrderedPromptHistoryGetAllPromptNamesNoAttr:
    def test_missing_prompt_dict_returns_empty(self):
        oph = OrderedPromptHistory()
        del oph.prompt_dict
        assert oph.get_all_prompt_names() == []
