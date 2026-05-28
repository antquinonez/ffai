# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

"""Async DAG executor for topological-parallel prompt execution.

Runs prompts level-by-level using ``asyncio.gather`` per level. Prompts
on the same level execute concurrently; levels execute sequentially.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .graph import build_execution_graph_with_edges
from .graph_execution_helpers import (
    check_abort_condition,
    should_skip_for_failed_deps,
)
from .prompt_node import PromptNode
from .response_result import ResponseResult

logger = logging.getLogger(__name__)


@dataclass
class GraphResult:
    """Aggregate result from executing a prompt dependency graph.

    Attributes:
        results: Mapping of prompt_name to ``ResponseResult``.
        success_count: Number of prompts that succeeded.
        failed_count: Number of prompts that failed.
        skipped_count: Number of prompts skipped by conditions.
        aborted: Whether execution was aborted.
        aborted_count: Number of prompts skipped due to abort.

    """

    results: dict[str, ResponseResult] = field(default_factory=dict)
    success_count: int = 0
    failed_count: int = 0
    skipped_count: int = 0
    aborted: bool = False
    aborted_count: int = 0


class AsyncGraphExecutor:
    """Execute a prompt DAG with topological-parallel async calls.

    Args:
        executor_fn: Async callable that takes prompt kwargs and returns
            a ``ResponseResult``.
        max_concurrency: Maximum number of concurrent API calls, enforced
            by an ``asyncio.Semaphore``.
        prompt_resolver: Optional callback ``(prompt_spec, results_by_name)
            -> (resolved_prompt, interpolated_names)``. When provided, each
            node's prompt is resolved before execution. When ``None``, prompts
            are sent as-is.

    """

    def __init__(
        self,
        executor_fn: Callable[..., Awaitable[ResponseResult]],
        max_concurrency: int = 10,
        prompt_resolver: Callable[
            [dict[str, Any], dict[str, dict[str, Any]]], tuple[str, set[str]]
        ]
        | None = None,
    ) -> None:
        self._executor_fn = executor_fn
        self._max_concurrency = max_concurrency
        self._prompt_resolver = prompt_resolver

    async def execute(self, prompts: list[dict[str, Any]]) -> GraphResult:
        """Build and execute a prompt dependency graph.

        Args:
            prompts: List of prompt dicts with keys: prompt_name, prompt,
                history, condition, abort_condition.

        Returns:
            ``GraphResult`` with per-prompt results and aggregate counts.

        Raises:
            ValueError: If a dependency cycle is detected.

        """
        graph = build_execution_graph_with_edges(prompts)

        levels: dict[int, list[PromptNode]] = defaultdict(list)
        for node in graph.nodes.values():
            levels[node.level].append(node)

        results_by_name: dict[str, dict[str, Any]] = {}
        result_map: dict[str, ResponseResult] = {}
        aborted = False
        aborted_count = 0

        sem = asyncio.Semaphore(self._max_concurrency)

        for level in range(graph.max_level + 1):
            if aborted:
                remaining = levels.get(level, [])
                for node in remaining:
                    name = node.get_prompt_name() or str(node.sequence)
                    result_map[name] = ResponseResult(
                        response=None,
                        status="skipped",
                        condition_trace="Skipped: execution aborted",
                    )
                    results_by_name[name] = {
                        "status": "skipped",
                        "response": "",
                        "attempts": 0,
                        "error": "",
                        "has_response": False,
                    }
                    aborted_count += 1
                continue

            nodes = levels.get(level, [])
            if not nodes:
                continue

            tasks = [
                self._execute_node(node, results_by_name, graph.nodes, sem)
                for node in nodes
            ]
            node_results = await asyncio.gather(*tasks, return_exceptions=True)

            for node, outcome in zip(nodes, node_results):
                name = node.get_prompt_name() or str(node.sequence)
                if isinstance(outcome, Exception):
                    result_map[name] = ResponseResult(
                        response=None,
                        status="failed",
                        condition_error=str(outcome),
                    )
                    results_by_name[name] = {
                        "status": "failed",
                        "response": "",
                        "attempts": 1,
                        "error": str(outcome),
                        "has_response": False,
                    }
                else:
                    r = outcome
                    assert isinstance(r, ResponseResult)
                    result_map[name] = r
                    results_by_name[name] = {
                        "status": r.status,
                        "response": str(r.response) if r.response else "",
                        "attempts": 1,
                        "error": "",
                        "has_response": r.response is not None,
                    }

            for node in nodes:
                name = node.get_prompt_name() or str(node.sequence)
                r = result_map.get(name)
                if r is None or r.status != "success":
                    continue
                should_abort, trace, _error = check_abort_condition(
                    node.prompt, results_by_name
                )
                if should_abort:
                    logger.info(
                        f"Abort condition triggered by '{name}': {trace}"
                    )
                    aborted = True
                    break

        skipped_count = sum(
            1 for r in result_map.values() if r.status == "skipped"
        )

        return GraphResult(
            results=result_map,
            success_count=sum(
                1 for r in result_map.values() if r.status == "success"
            ),
            failed_count=sum(
                1 for r in result_map.values() if r.status == "failed"
            ),
            skipped_count=skipped_count,
            aborted=aborted,
            aborted_count=aborted_count,
        )

    async def _execute_node(
        self,
        node: PromptNode,
        results_by_name: dict[str, dict[str, Any]],
        nodes: dict[int, PromptNode],
        sem: asyncio.Semaphore,
    ) -> ResponseResult:
        """Execute a single node with condition check, failure propagation,
        prompt resolution, and semaphore.

        Args:
            node: The prompt node to execute.
            results_by_name: Results from prior levels for condition evaluation.
            nodes: All graph nodes for dependency lookup.
            sem: Concurrency-limiting semaphore.

        Returns:
            ``ResponseResult`` from the executor function.

        """
        prompt_spec = node.prompt
        condition = prompt_spec.get("condition")

        skip, skip_reason = should_skip_for_failed_deps(
            node, results_by_name, nodes
        )
        if skip:
            return ResponseResult(
                response=None,
                status="skipped",
                condition_trace=skip_reason,
            )

        if condition:
            from .graph import evaluate_condition_with_trace

            should_execute, _, error, trace = evaluate_condition_with_trace(
                prompt_spec, results_by_name
            )
            if error:
                return ResponseResult(
                    response=None,
                    status="failed",
                    condition_error=error,
                )
            if not should_execute:
                return ResponseResult(
                    response=None,
                    status="skipped",
                    condition_trace=trace,
                )

        if self._prompt_resolver is not None:
            spec_copy = dict(prompt_spec)
            resolved_prompt, _ = self._prompt_resolver(spec_copy, results_by_name)
            spec_copy["prompt"] = resolved_prompt
        else:
            spec_copy = prompt_spec

        async with sem:
            return await self._executor_fn(**spec_copy)
