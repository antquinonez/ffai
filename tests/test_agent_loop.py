import time
from unittest.mock import MagicMock

from ffai.agent.agent_loop import AgentLoop
from ffai.agent.agent_result import AgentResult, ToolCallRecord
from ffai.core.client_base import FFAIClientBase
from ffai.tools.tool_registry import ToolDefinition, ToolRegistry


def _make_mock_client(responses):
    """Create a mock client that returns responses in sequence."""
    client = MagicMock(spec=FFAIClientBase)
    client.model = "test-model"
    client.generate_response = MagicMock(side_effect=responses)
    client.get_conversation_history = MagicMock(return_value=[])
    client.add_tool_result = MagicMock()
    client.set_conversation_history = MagicMock()
    return client


def _make_registry_with_tools():
    """Create a registry with a simple echo tool."""
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo arguments",
            parameters={"type": "object", "properties": {"msg": {"type": "string"}}},
        )
    )
    registry.register_executor("echo", lambda args: args.get("msg", ""))
    return registry


class TestAgentLoopBasic:
    def test_no_tools_falls_back_to_single_prompt(self):
        client = _make_mock_client(["Hello!"])
        registry = ToolRegistry()
        loop = AgentLoop(client, registry)

        result = loop.execute(prompt="Hi", tools=[])

        assert result.response == "Hello!"
        assert result.total_rounds == 1
        assert result.total_llm_calls == 1
        assert result.status == "success"
        assert result.tool_calls == []

    def test_no_valid_schemas_falls_back(self):
        client = _make_mock_client(["Response"])
        registry = ToolRegistry()
        loop = AgentLoop(client, registry)

        result = loop.execute(prompt="Hi", tools=["nonexistent"])

        assert result.response == "Response"
        assert result.status == "success"

    def test_single_round_no_tool_calls(self):
        client = _make_mock_client(["Final answer"])
        client.get_conversation_history = MagicMock(return_value=[
            {"role": "assistant", "content": "Final answer"}
        ])

        registry = _make_registry_with_tools()
        loop = AgentLoop(client, registry, max_rounds=3)

        result = loop.execute(prompt="Hi", tools=["echo"])

        assert result.response == "Final answer"
        assert result.total_rounds == 1
        assert result.total_llm_calls == 1
        assert result.status == "success"

    def test_single_tool_call(self):
        client = _make_mock_client(["", "Final answer"])
        call_count = [0]

        def mock_history():
            call_count[0] += 1
            if call_count[0] == 1:
                return [
                    {"role": "assistant", "content": "", "tool_calls": [
                        {
                            "id": "tc_1",
                            "function": {"name": "echo", "arguments": '{"msg": "hello"}'},
                        }
                    ]}
                ]
            return [{"role": "assistant", "content": "Final answer"}]

        client.get_conversation_history = MagicMock(side_effect=mock_history)

        registry = _make_registry_with_tools()
        loop = AgentLoop(client, registry, max_rounds=3)

        result = loop.execute(prompt="Say hello", tools=["echo"])

        assert result.response == "Final answer"
        assert result.tool_calls_count == 1
        assert result.tool_calls[0].tool_name == "echo"
        assert result.tool_calls[0].result == "hello"
        assert result.total_rounds == 2
        assert result.status == "success"

    def test_max_rounds_exceeded(self):
        client = _make_mock_client(["tool_resp"] * 5)
        round_num = [0]

        def mock_history():
            round_num[0] += 1
            return [
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": f"tc_{round_num[0]}", "function": {"name": "echo", "arguments": '{}'}}
                ]}
            ]

        client.get_conversation_history = MagicMock(side_effect=mock_history)

        registry = _make_registry_with_tools()
        loop = AgentLoop(client, registry, max_rounds=2)

        result = loop.execute(prompt="Go", tools=["echo"])

        assert result.status == "max_rounds_exceeded"
        assert result.total_rounds == 2

    def test_tool_error_abort(self):
        client = _make_mock_client(["", "final"])
        call_count = [0]

        def mock_history():
            call_count[0] += 1
            if call_count[0] == 1:
                return [
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "tc_1", "function": {"name": "fail_tool", "arguments": '{}'}}
                    ]}
                ]
            return [{"role": "assistant", "content": "final"}]

        client.get_conversation_history = MagicMock(side_effect=mock_history)

        registry = ToolRegistry()
        registry.register(ToolDefinition(name="fail_tool", description="Fails"))
        registry.register_executor("fail_tool", lambda args: (_ for _ in ()).throw(RuntimeError("boom")))

        loop = AgentLoop(client, registry, max_rounds=3, continue_on_tool_error=False)

        result = loop.execute(prompt="Go", tools=["fail_tool"])

        assert result.status == "failed"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].error is not None

    def test_tool_error_continue(self):
        client = _make_mock_client(["", "final"])
        call_count = [0]

        def mock_history():
            call_count[0] += 1
            if call_count[0] == 1:
                return [
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "tc_1", "function": {"name": "fail_tool", "arguments": '{}'}}
                    ]}
                ]
            return [{"role": "assistant", "content": "final"}]

        client.get_conversation_history = MagicMock(side_effect=mock_history)

        registry = ToolRegistry()
        registry.register(ToolDefinition(name="fail_tool", description="Fails"))
        registry.register_executor("fail_tool", lambda args: (_ for _ in ()).throw(RuntimeError("boom")))

        loop = AgentLoop(client, registry, max_rounds=3, continue_on_tool_error=True)

        result = loop.execute(prompt="Go", tools=["fail_tool"])

        assert result.status == "success"
        assert result.response == "final"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].error is not None

    def test_llm_failure_on_first_call(self):
        client = MagicMock(spec=FFAIClientBase)
        client.generate_response = MagicMock(side_effect=RuntimeError("API down"))

        registry = _make_registry_with_tools()
        loop = AgentLoop(client, registry)

        result = loop.execute(prompt="Hi", tools=["echo"])

        assert result.status == "failed"
        assert result.total_llm_calls == 1

    def test_kwargs_stripped_of_prompt_name_and_history(self):
        client = _make_mock_client(["ok"])
        client.get_conversation_history = MagicMock(return_value=[
            {"role": "assistant", "content": "ok"}
        ])

        registry = _make_registry_with_tools()
        loop = AgentLoop(client, registry)

        loop.execute(
            prompt="Hi",
            tools=["echo"],
            prompt_name="test",
            history=["prev"],
            temperature=0.5,
        )

        call_kwargs = client.generate_response.call_args
        assert "prompt_name" not in call_kwargs.kwargs
        assert "history" not in call_kwargs.kwargs


class TestAgentResult:
    def test_tool_calls_count(self):
        result = AgentResult(
            response="done",
            tool_calls=[
                ToolCallRecord(round=1, tool_name="a"),
                ToolCallRecord(round=2, tool_name="b"),
            ],
        )
        assert result.tool_calls_count == 2

    def test_last_tool_name(self):
        result = AgentResult(
            tool_calls=[
                ToolCallRecord(round=1, tool_name="a"),
                ToolCallRecord(round=2, tool_name="b"),
            ],
        )
        assert result.last_tool_name == "b"

    def test_last_tool_name_empty(self):
        result = AgentResult()
        assert result.last_tool_name == ""

    def test_failed_tool_calls(self):
        result = AgentResult(
            tool_calls=[
                ToolCallRecord(round=1, tool_name="a"),
                ToolCallRecord(round=2, tool_name="b", error="fail"),
            ],
        )
        failed = result.failed_tool_calls
        assert len(failed) == 1
        assert failed[0].tool_name == "b"

    def test_to_dict_roundtrip(self):
        original = AgentResult(
            response="test",
            tool_calls=[ToolCallRecord(round=1, tool_name="echo", tool_call_id="tc1")],
            total_rounds=2,
            total_llm_calls=3,
            status="success",
        )
        d = original.to_dict()
        restored = AgentResult.from_dict(d)
        assert restored.response == "test"
        assert restored.tool_calls_count == 1
        assert restored.total_rounds == 2
        assert restored.total_llm_calls == 3
        assert restored.status == "success"


class TestToolCallRecord:
    def test_to_dict_roundtrip(self):
        original = ToolCallRecord(
            round=1,
            tool_name="search",
            tool_call_id="tc_123",
            arguments={"query": "test"},
            result="found",
            duration_ms=150.5,
            error=None,
        )
        d = original.to_dict()
        restored = ToolCallRecord.from_dict(d)
        assert restored.round == 1
        assert restored.tool_name == "search"
        assert restored.arguments == {"query": "test"}
        assert restored.error is None

    def test_from_dict_defaults(self):
        record = ToolCallRecord.from_dict({})
        assert record.round == 0
        assert record.tool_name == ""
        assert record.arguments == {}


class TestAgentLoopEdgeCases:
    def test_tool_args_invalid_json_falls_back_to_empty_dict(self):
        client = _make_mock_client(["", "done"])
        call_count = [0]

        def mock_history():
            call_count[0] += 1
            if call_count[0] == 1:
                return [
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "tc_1", "function": {"name": "echo", "arguments": "not json{"}}
                    ]}
                ]
            return [{"role": "assistant", "content": "done"}]

        client.get_conversation_history = MagicMock(side_effect=mock_history)

        registry = _make_registry_with_tools()
        loop = AgentLoop(client, registry, max_rounds=3)

        result = loop.execute(prompt="Go", tools=["echo"])

        assert result.status == "success"
        assert result.tool_calls[0].arguments == {}

    def test_no_assistant_message_returns_empty(self):
        client = _make_mock_client(["done"])
        client.get_conversation_history = MagicMock(return_value=[
            {"role": "user", "content": "hi"}
        ])

        registry = _make_registry_with_tools()
        loop = AgentLoop(client, registry, max_rounds=3)

        assert loop._extract_tool_calls() == []

    def test_history_read_exception_returns_empty(self):
        client = _make_mock_client(["done"])
        client.get_conversation_history = MagicMock(side_effect=RuntimeError("history broken"))

        registry = _make_registry_with_tools()
        loop = AgentLoop(client, registry, max_rounds=3)

        assert loop._extract_tool_calls() == []

    def test_tool_execution_timeout(self):
        client = _make_mock_client(["", "done"])
        call_count = [0]

        def mock_history():
            call_count[0] += 1
            if call_count[0] == 1:
                return [
                    {"role": "assistant", "content": "", "tool_calls": [
                        {"id": "tc_1", "function": {"name": "slow_tool", "arguments": '{}'}}
                    ]}
                ]
            return [{"role": "assistant", "content": "done"}]

        client.get_conversation_history = MagicMock(side_effect=mock_history)

        registry = ToolRegistry()
        registry.register(ToolDefinition(name="slow_tool", description="Slow tool"))
        registry.register_executor("slow_tool", lambda args: (time.sleep(0.5), "late")[1])

        loop = AgentLoop(client, registry, max_rounds=3, tool_timeout=0.05)

        result = loop.execute(prompt="Go", tools=["slow_tool"])

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].error is not None
        assert "timed out" in result.tool_calls[0].error
