import json

import pytest

from src.tools.tool_registry import ToolDefinition, ToolRegistry


class TestToolDefinition:
    def test_to_openai_tool(self):
        tool = ToolDefinition(
            name="calculate",
            description="Perform a calculation",
            parameters={
                "type": "object",
                "properties": {"expr": {"type": "string"}},
                "required": ["expr"],
            },
        )
        schema = tool.to_openai_tool()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "calculate"
        assert schema["function"]["description"] == "Perform a calculation"
        assert "properties" in schema["function"]["parameters"]

    def test_to_dict(self):
        tool = ToolDefinition(
            name="search",
            description="Search documents",
            parameters={"type": "object", "properties": {}},
            implementation="python:mymod.search",
            enabled=False,
        )
        d = tool.to_dict()
        assert d["name"] == "search"
        assert d["implementation"] == "python:mymod.search"
        assert d["enabled"] is False

    def test_from_dict(self):
        data = {
            "name": "calc",
            "description": "Calculate",
            "parameters": {"type": "object", "properties": {"x": {"type": "number"}}},
            "implementation": "",
            "enabled": True,
        }
        tool = ToolDefinition.from_dict(data)
        assert tool.name == "calc"
        assert tool.parameters == {"type": "object", "properties": {"x": {"type": "number"}}}

    def test_from_dict_parameters_as_json_string(self):
        data = {
            "name": "calc",
            "description": "Calculate",
            "parameters": '{"type": "object"}',
        }
        tool = ToolDefinition.from_dict(data)
        assert tool.parameters == {"type": "object"}

    def test_from_dict_parameters_none(self):
        data = {"name": "calc", "description": "Calculate", "parameters": None}
        tool = ToolDefinition.from_dict(data)
        assert tool.parameters == {}

    def test_from_dict_invalid_json_parameters(self):
        data = {"name": "calc", "description": "Calculate", "parameters": "not json"}
        tool = ToolDefinition.from_dict(data)
        assert "properties" in tool.parameters

    def test_default_values(self):
        tool = ToolDefinition(name="t", description="d")
        assert tool.parameters == {}
        assert tool.implementation == ""
        assert tool.enabled is True


class TestToolRegistry:
    def test_register_and_get(self):
        registry = ToolRegistry()
        tool = ToolDefinition(name="calc", description="Calculate")
        registry.register(tool)
        assert registry.get_tool("calc") is tool

    def test_register_duplicate_raises(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="calc", description="Calculate"))
        with pytest.raises(ValueError, match="already registered"):
            registry.register(ToolDefinition(name="calc", description="Calculate 2"))

    def test_get_unknown_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="not found"):
            registry.get_tool("missing")

    def test_has_tool(self):
        registry = ToolRegistry()
        assert not registry.has_tool("calc")
        registry.register(ToolDefinition(name="calc", description="Calculate"))
        assert registry.has_tool("calc")

    def test_get_registered_names(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="a", description="A"))
        registry.register(ToolDefinition(name="b", description="B"))
        assert sorted(registry.get_registered_names()) == ["a", "b"]

    def test_get_enabled_names(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="a", description="A", enabled=True))
        registry.register(ToolDefinition(name="b", description="B", enabled=False))
        assert registry.get_enabled_names() == ["a"]

    def test_get_tools_schema(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="calc",
                description="Calculate",
                parameters={"type": "object", "properties": {}},
            )
        )
        schemas = registry.get_tools_schema(["calc"])
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "calc"

    def test_get_tools_schema_skips_disabled(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="a", description="A", enabled=True))
        registry.register(ToolDefinition(name="b", description="B", enabled=False))
        schemas = registry.get_tools_schema(["a", "b"])
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "a"

    def test_get_tools_schema_skips_unknown(self):
        registry = ToolRegistry()
        schemas = registry.get_tools_schema(["nonexistent"])
        assert schemas == []

    def test_execute_tool_with_executor(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="echo", description="Echo"))
        registry.register_executor("echo", lambda args: json.dumps(args))

        result = registry.execute_tool("echo", {"msg": "hello"})
        assert json.loads(result) == {"msg": "hello"}

    def test_execute_tool_unknown_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.execute_tool("missing", {})

    def test_execute_tool_disabled_raises(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="calc", description="Calculate", enabled=False))
        with pytest.raises(RuntimeError, match="disabled"):
            registry.execute_tool("calc", {})

    def test_execute_tool_no_executor_raises(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="calc", description="Calculate"))
        with pytest.raises(RuntimeError, match="No executor"):
            registry.execute_tool("calc", {})

    def test_execute_tool_error_raises(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="fail", description="Fail"))
        registry.register_executor("fail", lambda args: (_ for _ in ()).throw(ValueError("boom")))

        with pytest.raises(RuntimeError, match="execution failed"):
            registry.execute_tool("fail", {})

    def test_execute_tool_non_string_result(self):
        registry = ToolRegistry()
        registry.register(ToolDefinition(name="num", description="Return number"))
        registry.register_executor("num", lambda args: 42)  # type: ignore[arg-type]
        result = registry.execute_tool("num", {})
        assert result == "42"

    def test_register_executor_unknown_tool_raises(self):
        registry = ToolRegistry()
        with pytest.raises(ValueError, match="unknown tool"):
            registry.register_executor("missing", lambda args: "ok")

    def test_execute_tool_python_implementation(self):
        registry = ToolRegistry()
        registry.register(
            ToolDefinition(
                name="json_dumps",
                description="JSON dumps",
                implementation="python:json.dumps",
            )
        )
        result = registry.execute_tool("json_dumps", {"key": "val"})
        assert result == '{"key": "val"}'

    def test_load_python_callable_invalid_path(self):
        result = ToolRegistry.load_python_callable("no_dot_path")
        assert result is None

    def test_load_python_callable_nonexistent(self):
        result = ToolRegistry.load_python_callable("nonexistent_module.func")
        assert result is None
