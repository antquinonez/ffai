# Copyright (c) 2025 Antonio Quinonez / Far Finer LLC
# SPDX-License-Identifier: MIT


from src.core.condition_evaluator import ConditionEvaluator


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
