# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT


from src.core.condition_evaluator import (
    ConditionEvaluator,
    _parse_llm_json,
    _safe_json_get,
    _safe_json_has,
    _safe_json_type,
)


class TestConditionEvaluatorBasic:
    def test_equality_string(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "ok", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.status}} == "success"')
        assert result is True
        assert error is None

    def test_inequality_string(self):
        evaluator = ConditionEvaluator({"step": {"status": "failed", "response": "", "attempts": 1, "error": "timeout", "has_response": False}})
        result, error = evaluator.evaluate('{{step.status}} != "success"')
        assert result is True

    def test_comparison_numbers(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "ok", "attempts": 5, "error": "", "has_response": True}})
        result, error = evaluator.evaluate("{{step.attempts}} > 3")
        assert result is True

    def test_boolean_and(self):
        evaluator = ConditionEvaluator({
            "a": {"status": "success", "response": "ok", "attempts": 1, "error": "", "has_response": True},
            "b": {"status": "success", "response": "ok", "attempts": 1, "error": "", "has_response": True},
        })
        result, error = evaluator.evaluate('{{a.status}} == "success" and {{b.status}} == "success"')
        assert result is True

    def test_boolean_or(self):
        evaluator = ConditionEvaluator({
            "a": {"status": "failed", "response": "", "attempts": 1, "error": "err", "has_response": False},
            "b": {"status": "success", "response": "ok", "attempts": 1, "error": "", "has_response": True},
        })
        result, error = evaluator.evaluate('{{a.status}} == "success" or {{b.status}} == "success"')
        assert result is True

    def test_not_operator(self):
        evaluator = ConditionEvaluator({"step": {"status": "failed", "response": "", "attempts": 1, "error": "err", "has_response": False}})
        result, error = evaluator.evaluate("not {{step.has_response}}")
        assert result is True

    def test_len_function(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "short", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate("len({{step.response}}) < 100")
        assert result is True

    def test_empty_condition(self):
        evaluator = ConditionEvaluator({})
        result, error = evaluator.evaluate("")
        assert result is True

    def test_unknown_prompt_name_raises(self):
        evaluator = ConditionEvaluator({})
        result, error = evaluator.evaluate('{{nonexistent.status}} == "success"')
        assert result is False
        assert error is not None

    def test_has_response_true(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "some text", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate("{{step.has_response}}")
        assert result is True

    def test_has_response_false(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "", "attempts": 1, "error": "", "has_response": False}})
        result, error = evaluator.evaluate("not {{step.has_response}}")
        assert result is True


class TestConditionEvaluatorFunctions:
    def test_json_get(self):
        evaluator = ConditionEvaluator({
            "step": {"status": "success", "response": '{"score": 8.5}', "attempts": 1, "error": "", "has_response": True}
        })
        result, error = evaluator.evaluate('json_get({{step.response}}, "score") > 8')
        assert result is True

    def test_json_has(self):
        evaluator = ConditionEvaluator({
            "step": {"status": "success", "response": '{"key": "value"}', "attempts": 1, "error": "", "has_response": True}
        })
        result, error = evaluator.evaluate('json_has({{step.response}}, "key")')
        assert result is True

    def test_string_contains(self):
        evaluator = ConditionEvaluator({
            "step": {"status": "success", "response": "error timeout occurred", "attempts": 1, "error": "", "has_response": True}
        })
        result, error = evaluator.evaluate('"error" in {{step.response}}')
        assert result is True

    def test_regex_match(self):
        evaluator = ConditionEvaluator({
            "step": {"status": "success", "response": "score: 85", "attempts": 1, "error": "", "has_response": True}
        })
        result, error = evaluator.evaluate('{{step.response}} % "score: \\\\d+"')
        assert result is True


class TestConditionEvaluatorTrace:
    def test_evaluate_with_trace(self):
        evaluator = ConditionEvaluator({
            "step": {"status": "success", "response": "ok", "attempts": 1, "error": "", "has_response": True}
        })
        result, error, trace = evaluator.evaluate_with_trace('{{step.status}} == "success"')
        assert result is True
        assert trace is not None
        assert '"success"' in trace


class TestConditionEvaluatorExtractNames:
    def test_extract_referenced_names(self):
        names = ConditionEvaluator.extract_referenced_names("{{fetch.status}} == \"success\"")
        assert ("fetch", "status") in names

    def test_extract_multiple_names(self):
        names = ConditionEvaluator.extract_referenced_names("{{a.status}} == \"success\" and {{b.status}} == \"success\"")
        assert ("a", "status") in names
        assert ("b", "status") in names

    def test_extract_empty(self):
        names = ConditionEvaluator.extract_referenced_names("")
        assert names == []


class TestConditionEvaluatorValidateSyntax:
    def test_valid_syntax(self):
        valid, error = ConditionEvaluator.validate_syntax('{{a.status}} == "success"')
        assert valid is True
        assert error is None

    def test_invalid_syntax(self):
        valid, error = ConditionEvaluator.validate_syntax("{{a.status}} === invalid")
        assert valid is False
        assert error is not None


class TestConditionEvaluatorComputedProperties:
    def test_status_defaults_to_pending(self):
        evaluator = ConditionEvaluator({"step": {"status": "", "response": "", "attempts": 0, "error": "", "has_response": False}})
        result, error = evaluator.evaluate('{{step.status}} == "pending"')
        assert result is True

    def test_response_none_becomes_empty(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": None, "attempts": 1, "error": "", "has_response": False}})
        result, error = evaluator.evaluate('{{step.response}} == ""')
        assert result is True

    def test_error_property(self):
        evaluator = ConditionEvaluator({"step": {"status": "failed", "response": "", "attempts": 3, "error": "timeout", "has_response": False}})
        result, error = evaluator.evaluate('{{step.error}} == "timeout"')
        assert result is True


class TestParseLlmJson:
    def test_dict_passthrough(self):
        assert _parse_llm_json({"a": 1}) == {"a": 1}

    def test_list_passthrough(self):
        assert _parse_llm_json([1, 2]) == [1, 2]

    def test_empty_string_returns_none(self):
        assert _parse_llm_json("") is None

    def test_non_string_returns_none(self):
        assert _parse_llm_json(42) is None  # type: ignore[arg-type]

    def test_whitespace_only_returns_none(self):
        assert _parse_llm_json("   ") is None


class TestSafeJsonGet:
    def test_valid_path(self):
        assert _safe_json_get('{"a": 1}', "a") == 1

    def test_nested_path(self):
        assert _safe_json_get('{"a": {"b": 2}}', "a.b") == 2

    def test_missing_key_returns_default(self):
        assert _safe_json_get('{"a": 1}', "b", "fallback") == "fallback"

    def test_unparseable_returns_default(self):
        assert _safe_json_get("not json", "key", "default") == "default"

    def test_dict_input(self):
        assert _safe_json_get({"a": 1}, "a") == 1

    def test_array_index(self):
        assert _safe_json_get('[10, 20, 30]', "[1]") == 20


class TestSafeJsonHas:
    def test_existing_key(self):
        assert _safe_json_has('{"a": 1}', "a") is True

    def test_missing_key(self):
        assert _safe_json_has('{"a": 1}', "b") is False

    def test_empty_string(self):
        assert _safe_json_has("", "key") is False

    def test_unparseable(self):
        assert _safe_json_has("not json", "key") is False


class TestSafeJsonType:
    def test_boolean_type(self):
        assert _safe_json_type('{"a": true}', "a") == "boolean"

    def test_number_type(self):
        assert _safe_json_type('{"a": 42}', "a") == "number"

    def test_string_type(self):
        assert _safe_json_type('{"a": "hello"}', "a") == "string"

    def test_array_type(self):
        assert _safe_json_type('{"a": [1, 2]}', "a") == "array"

    def test_object_type(self):
        assert _safe_json_type('{"a": {"b": 1}}', "a") == "object"

    def test_null_data(self):
        assert _safe_json_type('null', "") == "null"

    def test_unparseable(self):
        assert _safe_json_type("bad json", "x") == "null"


class TestConditionEvaluatorStringMethods:
    def test_lower(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "HELLO", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.response}}.lower() == "hello"')
        assert result is True

    def test_upper(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.response}}.upper() == "HELLO"')
        assert result is True

    def test_strip(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "  hi  ", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.response}}.strip() == "hi"')
        assert result is True

    def test_startswith(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello world", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.response}}.startswith("hello")')
        assert result is True

    def test_endswith(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello world", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.response}}.endswith("world")')
        assert result is True

    def test_contains_method(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello world", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('"hello" in {{step.response}}')
        assert result is True

    def test_split(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "a,b,c", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.response}}.split(",")[1] == "b"')
        assert result is True


class TestConditionEvaluatorNotIn:
    def test_not_in_list(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "c", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.response}} not in ["a", "b"]')
        assert result is True

    def test_not_in_false(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "a", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.response}} not in ["a", "b"]')
        assert result is False


class TestConditionEvaluatorTernary:
    def test_true_branch(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "ok", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('("yes" if {{step.has_response}} else "no") == "yes"')
        assert result is True

    def test_false_branch(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "", "attempts": 1, "error": "", "has_response": False}})
        result, error = evaluator.evaluate('("yes" if {{step.has_response}} else "no") == "no"')
        assert result is True


class TestConditionEvaluatorArithmetic:
    def test_addition(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "", "attempts": 5, "error": "", "has_response": True}})
        result, error = evaluator.evaluate("{{step.attempts}} + 5 == 10")
        assert result is True

    def test_subtraction(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "", "attempts": 5, "error": "", "has_response": True}})
        result, error = evaluator.evaluate("{{step.attempts}} - 1 == 4")
        assert result is True

    def test_multiplication(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "", "attempts": 5, "error": "", "has_response": True}})
        result, error = evaluator.evaluate("{{step.attempts}} * 2 == 10")
        assert result is True

    def test_division(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "", "attempts": 5, "error": "", "has_response": True}})
        result, error = evaluator.evaluate("{{step.attempts}} / 2 == 2.5")
        assert result is True


class TestConditionEvaluatorLiterals:
    def test_true_literal(self):
        evaluator = ConditionEvaluator({})
        result, error = evaluator.evaluate("True")
        assert result is True

    def test_false_literal(self):
        evaluator = ConditionEvaluator({})
        result, error = evaluator.evaluate("false")
        assert result is False

    def test_none_literal(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": None, "attempts": 1, "error": "", "has_response": False}})
        result, error = evaluator.evaluate('{{step.response}} == ""')
        assert result is True


class TestConditionEvaluatorJsonFunctions:
    def test_json_keys(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": '{"a": 1, "b": 2}', "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('json_keys({{step.response}}) == ["a", "b"]')
        assert result is True

    def test_json_parse(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": '{"score": 8}', "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('json_parse({{step.response}})["score"] == 8')
        assert result is True

    def test_json_type_function(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": '{"score": 8}', "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('json_type({{step.response}}, "score") == "number"')
        assert result is True


class TestConditionEvaluatorSubscript:
    def test_dict_subscript(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": '{"a": 1}', "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('json_parse({{step.response}})["a"] == 1')
        assert result is True


class TestConditionEvaluatorErrorPaths:
    def test_syntax_error_returns_false(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate("{{step.status}} === invalid")
        assert result is False
        assert error is not None
        assert "Syntax error" in error

    def test_empty_condition_with_trace(self):
        evaluator = ConditionEvaluator({})
        result, error, trace = evaluator.evaluate_with_trace("")
        assert result is True
        assert error is None
        assert trace is None

    def test_whitespace_condition_with_trace(self):
        evaluator = ConditionEvaluator({})
        result, error, trace = evaluator.evaluate_with_trace("   ")
        assert result is True
        assert error is None
        assert trace is None

    def test_validate_syntax_empty(self):
        valid, error = ConditionEvaluator.validate_syntax("")
        assert valid is True
        assert error is None


class TestConditionEvaluatorRegexMatchOperator:
    def test_regex_match_true(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.response}} % "hel.*"')
        assert result is True

    def test_regex_match_false(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.response}} % "xyz.*"')
        assert result is False


class TestConditionEvaluatorInOperator:
    def test_in_list(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "b", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.response}} in ["a", "b", "c"]')
        assert result is True

    def test_in_string(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('"ell" in {{step.response}}')
        assert result is True

    def test_in_dict_keys(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "a", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.response}} in {"a": 1, "b": 2}')
        assert result is True


class TestConditionEvaluatorAdditionalCoverage:
    def test_resolve_display_trace_unknown_name(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "ok", "attempts": 1, "error": "", "has_response": True}})
        trace = evaluator._resolve_display_trace("{{missing.status}}")
        assert trace == "{{missing.status}}"

    def test_value_to_literal_none(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._value_to_literal(None) == '""'

    def test_value_to_literal_bool(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._value_to_literal(True) == "True"
        assert evaluator._value_to_literal(False) == "False"

    def test_value_to_literal_numeric(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._value_to_literal(42) == "42"
        assert evaluator._value_to_literal(3.14) == "3.14"

    def test_value_to_literal_special_chars(self):
        evaluator = ConditionEvaluator({})
        result = evaluator._value_to_literal("line1\nline2\ttab")
        assert "\\n" in result
        assert "\\t" in result

    def test_value_to_literal_non_string(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._value_to_literal([1, 2]) == '"[1, 2]"'

    def test_value_to_display_json(self):
        evaluator = ConditionEvaluator({})
        result = evaluator._value_to_display('{"key": "value"}')
        assert "key" in result

    def test_value_to_display_none(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._value_to_display(None) == '""'

    def test_value_to_display_bool(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._value_to_display(True) == "True"

    def test_value_to_display_numeric(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._value_to_display(42) == "42"
        assert evaluator._value_to_display(3.14) == "3.14"

    def test_value_to_display_non_string(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._value_to_display([1, 2]) == '"[1, 2]"'

    def test_compute_property_has_response_bool(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._compute_property({"has_response": True}, "has_response", True) is True

    def test_compute_property_has_response_computed_true(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._compute_property({"response": "hello", "has_response": None}, "has_response", None) is True

    def test_compute_property_has_response_computed_false(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._compute_property({"response": "", "has_response": None}, "has_response", None) is False

    def test_compute_property_status_default(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._compute_property({"status": None}, "status", None) == "pending"

    def test_compute_property_response_none(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._compute_property({"response": None}, "response", None) == ""

    def test_compute_property_error_none(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._compute_property({"error": None}, "error", None) == ""

    def test_compute_property_error_value(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._compute_property({"error": "timeout"}, "error", "timeout") == "timeout"

    def test_compute_property_attempts_none(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._compute_property({"attempts": None}, "attempts", None) == 0

    def test_compute_property_attempts_value(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._compute_property({"attempts": 3}, "attempts", 3) == 3

    def test_compute_property_unknown(self):
        evaluator = ConditionEvaluator({})
        assert evaluator._compute_property({"custom": "val"}, "custom", "val") == "val"


class TestConditionEvaluatorAstErrors:
    def test_unknown_name(self):
        evaluator = ConditionEvaluator({})
        result, error = evaluator.evaluate("unknown_var")
        assert result is False
        assert error is not None

    def test_unsupported_node_type(self):
        evaluator = ConditionEvaluator({})
        result, error = evaluator.evaluate("lambda x: x")
        assert result is False
        assert error is not None

    def test_in_operator_non_string_left(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate("42 in {{step.response}}")
        assert result is False
        assert error is not None

    def test_in_operator_unsupported_right_type(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('len({{step.response}}) in 42')
        assert result is False
        assert error is not None

    def test_unknown_function(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate("unknown_func({{step.response}})")
        assert result is False
        assert error is not None

    def test_keyword_args_rejected(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('len({{step.response}}, key="val")')
        assert result is False
        assert error is not None

    def test_method_keyword_args_rejected(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.response}}.split(sep=",")')
        assert result is False
        assert error is not None

    def test_private_attribute_blocked(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate("{{step.response}}.__class__")
        assert result is False
        assert error is not None

    def test_unknown_string_method_blocked(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.response}}.format("x")')
        assert result is False
        assert error is not None

    def test_unknown_list_method_blocked(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('split({{step.response}}, ",").append("x")')
        assert result is False
        assert error is not None

    def test_unknown_dict_method_blocked(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": '{"a": 1}', "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('json_parse({{step.response}}).popitem()')
        assert result is False
        assert error is not None

    def test_attribute_on_unsupported_type(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate("len({{step.response}}).real")
        assert result is False
        assert error is not None

    def test_private_method_on_string(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.response}}.__len__()')
        assert result is False
        assert error is not None

    def test_private_method_on_list(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('split({{step.response}}, ",")._private()')
        assert result is False
        assert error is not None

    def test_private_method_on_dict(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": '{"a": 1}', "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('json_parse({{step.response}})._private()')
        assert result is False
        assert error is not None

    def test_unsupported_binary_operator(self):
        evaluator = ConditionEvaluator({})
        result, error = evaluator.evaluate("1 ** 2 == 1")
        assert result is False
        assert error is not None

    def test_matches_requires_strings(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate("42 % 7")
        assert result is False
        assert error is not None

    def test_invalid_regex_pattern(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.response}} % "[invalid"')
        assert result is False
        assert error is not None

    def test_subscript_slice_rejected(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('json_parse({{step.response}})[1:3]')
        assert result is False
        assert error is not None

    def test_subscript_missing_key(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": '{"a": 1}', "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('json_parse({{step.response}})["missing"]')
        assert result is False
        assert error is not None

    def test_method_on_unsupported_type(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate("abs(1).something")
        assert result is False
        assert error is not None

    def test_unsupported_comparison_operator(self):
        evaluator = ConditionEvaluator({})
        result, error = evaluator.evaluate("1 is 1")
        assert result is False
        assert error is not None

    def test_unknown_list_attribute(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "a,b", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('split({{step.response}}, ",").sort()')
        assert result is False
        assert error is not None

    def test_unknown_dict_attribute(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": '{"a": 1}', "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('json_parse({{step.response}}).items()')
        assert result is False
        assert error is not None

    def test_attribute_on_unsupported_value(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate("(1 + 1).something")
        assert result is False
        assert error is not None


class TestConditionEvaluatorMethodCalls:
    def test_list_method_count(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "a,b,c", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('split({{step.response}}, ",").count("a") == 1')
        assert result is True

    def test_list_method_index(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "a,b,c", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('split({{step.response}}, ",").index("b") == 1')
        assert result is True

    def test_dict_method_get(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": '{"a": 1, "b": 2}', "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('json_parse({{step.response}}).get("a") == 1')
        assert result is True

    def test_dict_attribute_keys(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": '{"x": 1}', "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('json_parse({{step.response}}).get("x") == 1')
        assert result is True

    def test_dict_attribute_values(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": '{"x": 1}', "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('json_parse({{step.response}}).get("missing") == None')
        assert result is True

    def test_string_attribute_returns_unbound(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('{{step.response}}.upper')
        assert error is None


class TestConditionEvaluatorBuiltins:
    def test_json_values_function(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": '{"x": 1, "y": 2}', "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('json_values({{step.response}}) == [1, 2]')
        assert result is True

    def test_json_get_default_function(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": '{"a": 1}', "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('json_get_default({{step.response}}, "b", 99) == 99')
        assert result is True

    def test_is_null_function(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": None, "attempts": 1, "error": "", "has_response": False}})
        result, error = evaluator.evaluate('is_empty({{step.response}})')
        assert result is True

    def test_is_empty_function(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "", "attempts": 1, "error": "", "has_response": False}})
        result, error = evaluator.evaluate('is_empty({{step.response}})')
        assert result is True

    def test_is_empty_list(self):
        evaluator = ConditionEvaluator({})
        result, error = evaluator.evaluate('is_empty([])')
        assert result is True

    def test_is_empty_dict(self):
        evaluator = ConditionEvaluator({})
        result, error = evaluator.evaluate('is_empty({})')
        assert result is True

    def test_float_conversion(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "3.14", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('float({{step.response}}) > 3.0')
        assert result is True

    def test_bool_conversion(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('bool({{step.response}}) == True')
        assert result is True

    def test_int_conversion(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "42", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('int({{step.response}}) == 42')
        assert result is True

    def test_str_conversion(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, error = evaluator.evaluate('str({{step.attempts}}) == "1"')
        assert result is True

    def test_abs_function(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "", "attempts": 5, "error": "", "has_response": True}})
        result, error = evaluator.evaluate("abs({{step.attempts}}) == 5")
        assert result is True

    def test_round_function(self):
        evaluator = ConditionEvaluator({})
        result, error = evaluator.evaluate("round(3.7) == 4")
        assert result is True

    def test_min_max_functions(self):
        evaluator = ConditionEvaluator({})
        result, _ = evaluator.evaluate("min(1, 2) == 1")
        assert result is True
        result, _ = evaluator.evaluate("max(1, 2) == 2")
        assert result is True

    def test_string_functions(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "  Hello  ", "attempts": 1, "error": "", "has_response": True}})
        result, _ = evaluator.evaluate('lower({{step.response}}) == "  hello  "')
        assert result is True
        result, _ = evaluator.evaluate('upper({{step.response}}) == "  HELLO  "')
        assert result is True
        result, _ = evaluator.evaluate('trim({{step.response}}) == "Hello"')
        assert result is True

    def test_replace_function(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello world", "attempts": 1, "error": "", "has_response": True}})
        result, _ = evaluator.evaluate('replace({{step.response}}, "world", "earth") == "hello earth"')
        assert result is True

    def test_count_function(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "banana", "attempts": 1, "error": "", "has_response": True}})
        result, _ = evaluator.evaluate('count({{step.response}}, "a") == 3')
        assert result is True

    def test_find_function(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, _ = evaluator.evaluate('find({{step.response}}, "ll") == 2')
        assert result is True

    def test_rfind_function(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, _ = evaluator.evaluate('rfind({{step.response}}, "l") == 3')
        assert result is True

    def test_slice_function(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "hello", "attempts": 1, "error": "", "has_response": True}})
        result, _ = evaluator.evaluate('slice({{step.response}}, 0, 3) == "hel"')
        assert result is True

    def test_none_conversions(self):
        evaluator = ConditionEvaluator({})
        result, _ = evaluator.evaluate('int(None) == 0')
        assert result is True
        result, _ = evaluator.evaluate('float(None) == 0.0')
        assert result is True
        result, _ = evaluator.evaluate('str(None) == ""')
        assert result is True
        result, _ = evaluator.evaluate('bool(None) == False')
        assert result is True

    def test_none_string_functions(self):
        evaluator = ConditionEvaluator({})
        result, _ = evaluator.evaluate('split(None) == []')
        assert result is True
        result, _ = evaluator.evaluate('replace(None, "a", "b") == ""')
        assert result is True
        result, _ = evaluator.evaluate('count(None, "a") == 0')
        assert result is True
        result, _ = evaluator.evaluate('find(None, "a") != 0')
        assert result is True
        result, _ = evaluator.evaluate('rfind(None, "a") != 0')
        assert result is True
        result, _ = evaluator.evaluate('slice(None) == ""')
        assert result is True

    def test_json_parse_empty(self):
        evaluator = ConditionEvaluator({})
        result, _ = evaluator.evaluate('json_parse("") == {}')
        assert result is True

    def test_json_keys_non_dict(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "not json", "attempts": 1, "error": "", "has_response": True}})
        result, _ = evaluator.evaluate('json_keys({{step.response}}) == []')
        assert result is True

    def test_json_values_non_dict(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "not json", "attempts": 1, "error": "", "has_response": True}})
        result, _ = evaluator.evaluate('json_values({{step.response}}) == []')
        assert result is True

    def test_split_no_sep(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "a b c", "attempts": 1, "error": "", "has_response": True}})
        result, _ = evaluator.evaluate('split({{step.response}}) == ["a", "b", "c"]')
        assert result is True

    def test_rsplit_function(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "a,b,c", "attempts": 1, "error": "", "has_response": True}})
        result, _ = evaluator.evaluate('rsplit({{step.response}}, ",")[0] == "a"')
        assert result is True

    def test_lstrip_rstrip(self):
        evaluator = ConditionEvaluator({"step": {"status": "success", "response": "  hello  ", "attempts": 1, "error": "", "has_response": True}})
        result, _ = evaluator.evaluate('lstrip({{step.response}}) == "hello  "')
        assert result is True
        result, _ = evaluator.evaluate('rstrip({{step.response}}) == "  hello"')
        assert result is True

    def test_is_null_none(self):
        evaluator = ConditionEvaluator({})
        result, _ = evaluator.evaluate('is_null(None)')
        assert result is True
