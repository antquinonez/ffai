import json

import pytest

from ffai.agent.agent_loop import AgentLoop
from ffai.agent.agent_result import AgentResult
from ffai.Clients.FFMistralSmall import FFMistralSmall
from ffai.tools.tool_registry import ToolDefinition, ToolRegistry

pytestmark = pytest.mark.integration


WEATHER_TOOL = ToolDefinition(
    name="get_weather",
    description="Get the current weather for a city",
    parameters={
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name"},
        },
        "required": ["city"],
    },
)

CALCULATOR_TOOL = ToolDefinition(
    name="calculate",
    description="Evaluate a mathematical expression",
    parameters={
        "type": "object",
        "properties": {
            "expression": {"type": "string", "description": "Math expression to evaluate"},
        },
        "required": ["expression"],
    },
)


def _make_registry_with_weather():
    registry = ToolRegistry()
    registry.register(WEATHER_TOOL)
    registry.register_executor("get_weather", lambda args: json.dumps({
        "city": args.get("city", "unknown"),
        "temperature": 22,
        "condition": "sunny",
    }))
    return registry


def _make_registry_with_calculator():
    registry = ToolRegistry()
    registry.register(CALCULATOR_TOOL)
    registry.register_executor("calculate", lambda args: str(eval(args.get("expression", "0"))))  # noqa: S307
    return registry


class TestAgentLoopWeatherTool:
    @pytest.fixture(autouse=True)
    def setup(self, ffmistralsmall_client: FFMistralSmall):
        self.client = ffmistralsmall_client
        self.registry = _make_registry_with_weather()

    def test_single_tool_call(self):
        loop = AgentLoop(self.client, self.registry, max_rounds=3)
        result = loop.execute(
            prompt="What is the weather in Paris?",
            tools=["get_weather"],
        )
        assert isinstance(result, AgentResult)
        assert result.status == "success"
        assert len(result.tool_calls) >= 1
        assert result.tool_calls[0].tool_name == "get_weather"
        assert "paris" in result.tool_calls[0].arguments.get("city", "").lower()
        assert result.tool_calls[0].result is not None
        assert result.tool_calls[0].error is None
        assert result.total_rounds >= 1
        assert result.total_llm_calls >= 1

    def test_final_response_references_tool_result(self):
        loop = AgentLoop(self.client, self.registry, max_rounds=3)
        result = loop.execute(
            prompt="What is the current weather in Berlin?",
            tools=["get_weather"],
        )
        assert result.status == "success"
        response_lower = result.response.lower()
        has_weather_info = any(
            w in response_lower
            for w in ("22", "sunny", "weather", "berlin", "temperature")
        )
        assert has_weather_info

    def test_tool_result_contains_expected_data(self):
        loop = AgentLoop(self.client, self.registry, max_rounds=3)
        result = loop.execute(
            prompt="Get the weather in Tokyo",
            tools=["get_weather"],
        )
        tc = result.tool_calls[0]
        result_data = json.loads(tc.result)
        assert result_data["city"] == "Tokyo"
        assert result_data["temperature"] == 22
        assert result_data["condition"] == "sunny"


class TestAgentLoopCalculatorTool:
    @pytest.fixture(autouse=True)
    def setup(self, ffmistralsmall_client: FFMistralSmall):
        self.client = ffmistralsmall_client
        self.registry = _make_registry_with_calculator()

    def test_calculator_execution(self):
        loop = AgentLoop(self.client, self.registry, max_rounds=3)
        result = loop.execute(
            prompt="What is 17 * 23?",
            tools=["calculate"],
        )
        assert result.status == "success"
        assert len(result.tool_calls) >= 1
        assert result.tool_calls[0].tool_name == "calculate"

    def test_calculator_result_accuracy(self):
        loop = AgentLoop(self.client, self.registry, max_rounds=3)
        result = loop.execute(
            prompt="Use the calculator to compute 100 + 200",
            tools=["calculate"],
        )
        tc = result.tool_calls[0]
        assert tc.result == "300"


class TestAgentLoopEdgeCases:
    @pytest.fixture(autouse=True)
    def setup(self, ffmistralsmall_client: FFMistralSmall):
        self.client = ffmistralsmall_client
        self.registry = _make_registry_with_weather()

    def test_no_tools_falls_back_to_plain_generation(self):
        registry = ToolRegistry()
        registry.register(WEATHER_TOOL)
        loop = AgentLoop(self.client, registry, max_rounds=3)
        result = loop.execute(
            prompt="Say hello",
            tools=["nonexistent_tool"],
        )
        assert result.status == "success"
        assert result.total_rounds == 1

    def test_tool_timeout_raises_error(self):
        import time

        registry = ToolRegistry()
        slow_tool = ToolDefinition(
            name="slow_tool",
            description="A tool that takes too long",
            parameters={"type": "object", "properties": {}},
        )
        registry.register(slow_tool)
        registry.register_executor("slow_tool", lambda args: (time.sleep(5), "done")[1])

        loop = AgentLoop(self.client, registry, max_rounds=2, tool_timeout=0.1)
        result = loop.execute(
            prompt="Use the slow tool",
            tools=["slow_tool"],
        )
        timeout_errors = [tc for tc in result.tool_calls if tc.error and "timed out" in tc.error]
        assert len(timeout_errors) >= 1

    def test_max_rounds_exceeded(self):
        registry = ToolRegistry()
        echo_tool = ToolDefinition(
            name="echo",
            description="Echo back the input",
            parameters={"type": "object", "properties": {"text": {"type": "string"}}},
        )
        registry.register(echo_tool)
        registry.register_executor("echo", lambda args: args.get("text", ""))

        loop = AgentLoop(self.client, registry, max_rounds=1)
        result = loop.execute(
            prompt="Keep calling the echo tool repeatedly with 'hello'",
            tools=["echo"],
        )
        assert result.status in ("success", "max_rounds_exceeded")
        assert result.total_rounds <= 1


class TestAgentLoopHistoryInteraction:
    @pytest.fixture(autouse=True)
    def setup(self, ffmistralsmall_client: FFMistralSmall):
        self.client = ffmistralsmall_client
        self.registry = _make_registry_with_weather()

    def test_tool_call_recorded_in_client_history(self):
        loop = AgentLoop(self.client, self.registry, max_rounds=3)
        result = loop.execute(
            prompt="What is the weather in Madrid?",
            tools=["get_weather"],
        )
        assert len(result.tool_calls) >= 1
        history = self.client.get_conversation_history()
        tool_messages = [m for m in history if m.get("role") == "tool"]
        assert len(tool_messages) >= 1

    def test_tool_call_has_duration(self):
        loop = AgentLoop(self.client, self.registry, max_rounds=3)
        result = loop.execute(
            prompt="Get weather for Paris",
            tools=["get_weather"],
        )
        assert len(result.tool_calls) >= 1
        assert result.tool_calls[0].duration_ms >= 0
