from __future__ import annotations

from ffai.rag.prompts import DEFAULT_RAG_PROMPT


class TestDefaultRAGPrompt:
    def test_contains_context_placeholder(self):
        assert "{context}" in DEFAULT_RAG_PROMPT

    def test_contains_question_placeholder(self):
        assert "{question}" in DEFAULT_RAG_PROMPT

    def test_format_map_produces_valid_string(self):
        result = DEFAULT_RAG_PROMPT.format_map({"context": "some context", "question": "what?"})
        assert "some context" in result
        assert "what?" in result
        assert "{context}" not in result
        assert "{question}" not in result

    def test_instructions_mention_not_making_up_info(self):
        assert "Do not make up" in DEFAULT_RAG_PROMPT
