# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

from .agent.agent_loop import AgentLoop
from .agent.agent_result import AgentResult, ToolCallRecord
from .agent.response_validator import ResponseValidator, ValidationResult
from .core.async_client_base import AsyncFFAIClientBase
from .core.async_executor import AsyncGraphExecutor, GraphResult
from .core.client_base import FFAIClientBase
from .core.condition_evaluator import ConditionEvaluator
from .core.execution_result import ExecutionResult
from .core.graph import ExecutionGraph
from .core.history.conversation import ConversationHistory
from .core.history.ordered import OrderedPromptHistory
from .core.history.permanent import PermanentHistory
from .core.response_executor import ResponseExecutor
from .core.response_options import ResponseOptions
from .FFAI import FFAI
from .tools.tool_registry import ToolDefinition, ToolRegistry

__all__ = [
    "FFAI",
    "AgentLoop",
    "AgentResult",
    "AsyncFFAIClientBase",
    "AsyncGraphExecutor",
    "ConditionEvaluator",
    "ConversationHistory",
    "ExecutionGraph",
    "ExecutionResult",
    "FFAIClientBase",
    "GraphResult",
    "OrderedPromptHistory",
    "PermanentHistory",
    "ResponseExecutor",
    "ResponseOptions",
    "ResponseValidator",
    "ToolCallRecord",
    "ToolDefinition",
    "ToolRegistry",
    "ValidationResult",
]

import contextlib

with contextlib.suppress(ImportError):
    from .rag import (  # noqa: F401, I001
        ClientAdapter,
        DEFAULT_RAG_PROMPT,
        Embeddings,
        GenerationResult,
        QueryResult,
        RAG,
        SearchHit,
        TextChunk,
        litellm_generate_fn,
    )
