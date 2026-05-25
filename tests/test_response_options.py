# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT

import pytest

from src.core.response_options import ResponseOptions


class TestResponseOptionsConstruction:
    def test_default_construction(self):
        opts = ResponseOptions()
        assert opts.model is None
        assert opts.system_instructions is None
        assert opts.response_format is None
        assert opts.response_model is None
        assert opts.condition is None
        assert opts.abort_condition is None
        assert opts.strict is False
        assert opts.history is None
        assert opts.dependencies is None

    def test_all_fields_set(self):
        opts = ResponseOptions(
            model="gpt-4",
            system_instructions="Be helpful",
            response_format={"type": "json_object"},
            response_model=str,
            condition='{{x.status}} == "success"',
            abort_condition="True",
            strict=True,
            history=["a", "b"],
            dependencies=["c"],
        )
        assert opts.model == "gpt-4"
        assert opts.system_instructions == "Be helpful"
        assert opts.response_format == {"type": "json_object"}
        assert opts.response_model is str
        assert opts.condition == '{{x.status}} == "success"'
        assert opts.abort_condition == "True"
        assert opts.strict is True
        assert opts.history == ["a", "b"]
        assert opts.dependencies == ["c"]

    def test_frozen_raises_on_mutation(self):
        opts = ResponseOptions(model="gpt-4")
        with pytest.raises(AttributeError):
            opts.model = "claude-3"  # type: ignore[reportAttributeAccessIssue]

    def test_frozen_raises_on_new_attribute(self):
        opts = ResponseOptions()
        with pytest.raises(AttributeError):
            opts.temperature = 0.3  # type: ignore[reportAttributeAccessIssue]


class TestFromDict:
    def test_maps_known_keys(self):
        d = {"model": "gpt-4", "condition": "True", "strict": True, "history": ["a"]}
        opts = ResponseOptions.from_dict(d)
        assert opts.model == "gpt-4"
        assert opts.condition == "True"
        assert opts.strict is True
        assert opts.history == ["a"]

    def test_ignores_prompt_and_prompt_name(self):
        d = {"prompt": "Hello", "prompt_name": "greeting", "model": "gpt-4"}
        opts = ResponseOptions.from_dict(d)
        assert opts.model == "gpt-4"
        assert not hasattr(opts, "prompt")
        assert not hasattr(opts, "prompt_name")

    def test_ignores_sequence(self):
        d = {"sequence": 0, "prompt_name": "x", "model": "gpt-4"}
        opts = ResponseOptions.from_dict(d)
        assert opts.model == "gpt-4"
        assert not hasattr(opts, "sequence")

    def test_ignores_unknown_keys(self):
        d = {"model": "gpt-4", "temperature": 0.3, "max_tokens": 100}
        opts = ResponseOptions.from_dict(d)
        assert opts.model == "gpt-4"
        assert not hasattr(opts, "temperature")

    def test_skips_none_values(self):
        d = {"model": None, "condition": "True"}
        opts = ResponseOptions.from_dict(d)
        assert opts.model is None
        assert opts.condition == "True"

    def test_empty_dict_returns_defaults(self):
        opts = ResponseOptions.from_dict({})
        assert opts.model is None
        assert opts.strict is False

    def test_all_nine_known_keys(self):
        d = {
            "model": "gpt-4",
            "system_instructions": "sys",
            "response_format": {"type": "json_object"},
            "response_model": str,
            "condition": "True",
            "abort_condition": "False",
            "strict": True,
            "history": ["a"],
            "dependencies": ["b"],
        }
        opts = ResponseOptions.from_dict(d)
        assert opts.model == "gpt-4"
        assert opts.system_instructions == "sys"
        assert opts.response_format == {"type": "json_object"}
        assert opts.response_model is str
        assert opts.condition == "True"
        assert opts.abort_condition == "False"
        assert opts.strict is True
        assert opts.history == ["a"]
        assert opts.dependencies == ["b"]

    def test_execute_graph_dict_roundtrip(self):
        spec = {
            "prompt_name": "analyze",
            "prompt": "Analyze {{fetch.response}}",
            "history": ["fetch"],
            "condition": '{{fetch.status}} == "success"',
            "response_model": str,
            "model": "gpt-4",
        }
        opts = ResponseOptions.from_dict(spec)
        assert opts.history == ["fetch"]
        assert opts.condition == '{{fetch.status}} == "success"'
        assert opts.response_model is str
        assert opts.model == "gpt-4"
