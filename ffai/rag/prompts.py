"""Define the default RAG prompt template."""

from __future__ import annotations

DEFAULT_RAG_PROMPT = (
    "Answer the question based on the following context.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "If the context does not contain enough information to answer the "
    "question, say so. Do not make up information."
)
