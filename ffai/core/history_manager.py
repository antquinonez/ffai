# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT
# Contact: antquinonez@farfiner.com

from __future__ import annotations

import threading
from typing import Any

import polars as pl

from .history.ordered import OrderedPromptHistory
from .history.permanent import PermanentHistory
from .history.recorder import HistoryRecorder
from .history_exporter import HistoryExporter
from .response_context import ResponseContext


class HistoryManager:
    def __init__(
        self,
        recorder: HistoryRecorder,
        context: ResponseContext,
        permanent: PermanentHistory,
        ordered: OrderedPromptHistory,
        exporter: HistoryExporter,
    ) -> None:
        self._recorder = recorder
        self._context = context
        self._permanent = permanent
        self._ordered = ordered
        self._exporter = exporter

    @property
    def raw(self) -> list[dict[str, Any]]:
        return self._recorder.history

    @raw.setter
    def raw(self, value: list[dict[str, Any]]) -> None:
        self._recorder.history = value

    @property
    def clean(self) -> list[dict[str, Any]]:
        return self._recorder.clean_history

    @property
    def prompt_attr_history(self) -> list[dict[str, Any]]:
        return self._context.prompt_attr_history

    @prompt_attr_history.setter
    def prompt_attr_history(self, value: list[dict[str, Any]]) -> None:
        self._context.prompt_attr_history = value

    @property
    def lock(self) -> threading.Lock | None:
        return self._context.history_lock

    @property
    def permanent(self) -> PermanentHistory:
        return self._permanent

    @property
    def ordered(self) -> OrderedPromptHistory:
        return self._ordered

    def get_interaction_history(self) -> list[dict[str, Any]]:
        return self.raw

    def get_clean_interaction_history(self) -> list[dict[str, Any]]:
        return self.clean

    def get_prompt_attr_history(self) -> list[dict[str, Any]]:
        return self.prompt_attr_history

    def get_all_interactions(self) -> list[Any]:
        return self._ordered.get_all_interactions()

    def get_latest_interaction_by_prompt_name(self, prompt_name: str) -> dict[str, Any] | None:
        matching = [e for e in self._recorder.history if e.get("prompt_name") == prompt_name]
        return matching[-1] if matching else None

    def get_last_n_interactions(self, n: int) -> list[dict[str, Any]]:
        all_interactions = self._ordered.get_all_interactions()
        return [i.to_dict() for i in all_interactions[-n:]]

    def get_interaction(self, sequence_number: int) -> dict[str, Any] | None:
        all_interactions = self._ordered.get_all_interactions()
        interaction = next(
            (i for i in all_interactions if i.sequence_number == sequence_number), None
        )
        return interaction.to_dict() if interaction else None

    def get_model_interactions(self, model: str) -> list[dict[str, Any]]:
        all_interactions = self._ordered.get_all_interactions()
        return [i.to_dict() for i in all_interactions if i.model == model]

    def get_interactions_by_prompt_name(self, prompt_name: str) -> list[dict[str, Any]]:
        return [
            i.to_dict() for i in self._ordered.get_interactions_by_prompt_name(prompt_name)
        ]

    def get_latest_interaction(self) -> dict[str, Any] | None:
        all_interactions = self._ordered.get_all_interactions()
        return all_interactions[-1].to_dict() if all_interactions else None

    def get_prompt_history(self) -> list[str]:
        return [i.prompt for i in self._ordered.get_all_interactions()]

    def get_response_history(self) -> list[str]:
        return [i.response for i in self._ordered.get_all_interactions()]

    def get_model_usage_stats(self) -> dict[str, int]:
        usage_stats: dict[str, int] = {}
        for interaction in self._ordered.get_all_interactions():
            usage_stats[interaction.model] = usage_stats.get(interaction.model, 0) + 1
        return usage_stats

    def get_prompt_name_usage_stats(self) -> dict[str, int]:
        return self._ordered.get_prompt_name_usage_stats()

    def get_prompt_dict(self) -> dict[str, list[dict[str, Any]]]:
        return self._ordered.to_dict()

    def get_latest_responses_by_prompt_names(
        self, prompt_names: list[str]
    ) -> dict[str, dict[str, str]]:
        return self._ordered.get_latest_responses_by_prompt_names(prompt_names)

    def get_formatted_responses(self, prompt_names: list[str]) -> str:
        return self._ordered.get_formatted_responses(prompt_names)

    def _convert_unix_seconds_to_datetime(self, df: pl.DataFrame) -> pl.DataFrame:
        return self._exporter._convert_unix_seconds_to_datetime(df)

    def history_to_dataframe(self) -> pl.DataFrame:
        return self._exporter.history_to_dataframe()

    def clean_history_to_dataframe(self) -> pl.DataFrame:
        return self._exporter.clean_history_to_dataframe()

    def prompt_attr_history_to_dataframe(self) -> pl.DataFrame:
        return self._exporter.prompt_attr_history_to_dataframe()

    def ordered_history_to_dataframe(self) -> pl.DataFrame:
        return self._exporter.ordered_history_to_dataframe()

    def search_history(
        self,
        text: str | None = None,
        prompt_name: str | None = None,
        model: str | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> pl.DataFrame:
        return self._exporter.search_history(
            text=text,
            prompt_name=prompt_name,
            model=model,
            start_time=start_time,
            end_time=end_time,
        )

    def get_model_stats_df(self) -> pl.DataFrame:
        return self._exporter.get_model_stats_df(self.get_model_usage_stats())

    def get_prompt_name_stats_df(self) -> pl.DataFrame:
        return self._exporter.get_prompt_name_stats_df(self.get_prompt_name_usage_stats())

    def get_response_length_stats(self) -> pl.DataFrame:
        return self._exporter.get_response_length_stats()

    def interaction_counts_by_date(self) -> pl.DataFrame:
        return self._exporter.interaction_counts_by_date()

    def persist_all_histories(self) -> bool:
        return self._exporter.persist_all_histories()
