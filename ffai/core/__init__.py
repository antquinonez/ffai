# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Core FFAI infrastructure: client abstraction, prompt assembly, history, and export."""

from .async_client_base import AsyncFFAIClientBase
from .async_executor import AsyncGraphExecutor, GraphResult
from .client_base import FFAIClientBase
from .condition_evaluator import ConditionEvaluator
from .execution_state import ExecutionState
from .graph import (
    DependencyEdge,
    ExecutionGraph,
    build_execution_graph,
    build_execution_graph_with_edges,
    evaluate_condition,
    evaluate_condition_with_trace,
    get_ready_prompts,
    is_abort_trigger,
)
from .graph_execution_helpers import (
    build_graph_history_dict,
    check_abort_condition,
    resolve_graph_prompt,
    should_skip_for_failed_deps,
)
from .history import ConversationHistory, OrderedPromptHistory, PermanentHistory
from .history_exporter import HistoryExporter
from .prompt_builder import PromptBuilder
from .prompt_node import PromptNode
from .prompt_utils import extract_json_field, interpolate_prompt
from .response_context import ResponseContext
from .response_result import ResponseResult
from .response_utils import clean_response, extract_json
from .structured_output import StructuredOutputHandler, StructuredResult

__all__ = [
    "AsyncFFAIClientBase",
    "AsyncGraphExecutor",
    "ConditionEvaluator",
    "ConversationHistory",
    "DependencyEdge",
    "ExecutionGraph",
    "ExecutionState",
    "FFAIClientBase",
    "GraphResult",
    "HistoryExporter",
    "OrderedPromptHistory",
    "PermanentHistory",
    "PromptBuilder",
    "PromptNode",
    "ResponseContext",
    "ResponseResult",
    "StructuredOutputHandler",
    "StructuredResult",
    "build_execution_graph",
    "build_execution_graph_with_edges",
    "build_graph_history_dict",
    "check_abort_condition",
    "clean_response",
    "evaluate_condition",
    "evaluate_condition_with_trace",
    "extract_json",
    "extract_json_field",
    "get_ready_prompts",
    "interpolate_prompt",
    "is_abort_trigger",
    "resolve_graph_prompt",
    "should_skip_for_failed_deps",
]
