"""Tests for the tool router and registry."""

from __future__ import annotations

import pytest

from agent.model import FunctionCall
from agent.tool_registry import ToolRegistry
from agent.tools.base import BaseTool, ToolCategory, ToolParameter, ToolResult, ToolSchema


class EchoTool(BaseTool):
    """Simple test tool that echoes input."""

    schema = ToolSchema(
        name="echo",
        description="Echo the input message back",
        category=ToolCategory.UTILITY,
        parameters=[
            ToolParameter(name="message", type="string", description="Message to echo", required=True),
        ],
        tags=["test", "echo"],
    )

    async def execute(self, **kwargs) -> ToolResult:
        message = kwargs.get("message", "")
        return ToolResult(success=True, data={"echo": message})


class AddTool(BaseTool):
    schema = ToolSchema(
        name="add",
        description="Add two numbers",
        category=ToolCategory.UTILITY,
        parameters=[
            ToolParameter(name="a", type="integer", description="First number", required=True),
            ToolParameter(name="b", type="integer", description="Second number", required=True),
        ],
        tags=["math", "add"],
    )

    async def execute(self, **kwargs) -> ToolResult:
        a = int(kwargs.get("a", 0))
        b = int(kwargs.get("b", 0))
        return ToolResult(success=True, data={"result": a + b})


class TestFunctionCallParsing:
    def test_json_single_call(self):
        text = '[{"name": "echo", "arguments": {"message": "hello"}}]'
        calls = FunctionCall.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "echo"
        assert calls[0].arguments == {"message": "hello"}

    def test_json_multiple_calls(self):
        text = '[{"name": "echo", "arguments": {"message": "hello"}}, {"name": "add", "arguments": {"a": 1, "b": 2}}]'
        calls = FunctionCall.parse(text)
        assert len(calls) == 2
        assert calls[0].name == "echo"
        assert calls[1].name == "add"
        assert calls[1].arguments == {"a": 1, "b": 2}

    def test_json_empty_args(self):
        text = '[{"name": "list_tools", "arguments": {}}]'
        calls = FunctionCall.parse(text)
        assert len(calls) == 1
        assert calls[0].arguments == {}

    def test_no_calls(self):
        calls = FunctionCall.parse("No function calls here")
        assert len(calls) == 0

    def test_plain_text_no_calls(self):
        calls = FunctionCall.parse("I don't have a tool for that request.")
        assert len(calls) == 0

    def test_legacy_single_call(self):
        text = "<start_function_call>call:echo{message:<escape>hello<escape>}<end_function_call>"
        calls = FunctionCall.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "echo"
        assert calls[0].arguments == {"message": "hello"}

    def test_legacy_empty_params(self):
        text = "<start_function_call>call:echo{}<end_function_call>"
        calls = FunctionCall.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "echo"
        assert calls[0].arguments == {}

    def test_json_embedded_in_text(self):
        text = 'Here are the tools: [{"name": "echo", "arguments": {"message": "hi"}}]'
        calls = FunctionCall.parse(text)
        assert len(calls) == 1
        assert calls[0].name == "echo"


class TestToolRegistry:
    def setup_method(self):
        self.registry = ToolRegistry()
        self.echo = EchoTool()
        self.add = AddTool()
        self.registry.register(self.echo)
        self.registry.register(self.add)

    def test_registration(self):
        assert self.registry.tool_count == 2
        assert "echo" in self.registry.tool_names
        assert "add" in self.registry.tool_names

    def test_get_tool(self):
        tool = self.registry.get("echo")
        assert tool is self.echo

    def test_get_missing(self):
        assert self.registry.get("nonexistent") is None

    def test_unregister(self):
        assert self.registry.unregister("echo") is True
        assert self.registry.tool_count == 1
        assert self.registry.unregister("echo") is False

    @pytest.mark.asyncio
    async def test_execute(self):
        result = await self.registry.execute("echo", message="test")
        assert result.success
        assert result.data == {"echo": "test"}

    @pytest.mark.asyncio
    async def test_execute_missing(self):
        result = await self.registry.execute("nonexistent")
        assert not result.success
        assert "not found" in result.error

    def test_search(self):
        results = self.registry.search("echo")
        assert len(results) == 1
        assert results[0].name == "echo"

    def test_search_by_tag(self):
        results = self.registry.search("math")
        assert len(results) == 1
        assert results[0].name == "add"

    def test_list_tools(self):
        tools = self.registry.list_tools()
        assert len(tools) == 2
        names = {t["name"] for t in tools}
        assert names == {"echo", "add"}

    def test_function_schemas(self):
        schemas = self.registry.get_function_schemas()
        assert len(schemas) == 2
        for s in schemas:
            assert s["type"] == "function"
            assert "name" in s["function"]

    def test_a2a_skills(self):
        skills = self.registry.get_a2a_skills()
        assert len(skills) == 2

    def test_mcp_tools(self):
        mcp_tools = self.registry.get_mcp_tools()
        assert len(mcp_tools) == 2
        assert "echo" in mcp_tools


class TestToolSchema:
    def test_to_function_schema(self):
        schema = ToolSchema(
            name="test",
            description="A test tool",
            parameters=[
                ToolParameter(name="input", type="string", description="Input text", required=True),
            ],
        )
        func_schema = schema.to_function_schema()
        assert func_schema["type"] == "function"
        assert func_schema["function"]["name"] == "test"
        assert "input" in func_schema["function"]["parameters"]["properties"]
        assert func_schema["function"]["parameters"]["required"] == ["input"]

    def test_to_a2a_skill(self):
        schema = ToolSchema(
            name="test_tool",
            description="A test",
            tags=["test"],
        )
        skill = schema.to_a2a_skill()
        assert skill["id"] == "test_tool"
        assert skill["name"] == "Test Tool"

    def test_to_mcp_tool(self):
        schema = ToolSchema(
            name="test",
            description="test",
            parameters=[
                ToolParameter(name="a", type="string", required=True),
                ToolParameter(name="b", type="integer", required=False),
            ],
        )
        mcp = schema.to_mcp_tool()
        assert mcp["type"] == "object"
        assert "a" in mcp["properties"]
        assert mcp["required"] == ["a"]
