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
