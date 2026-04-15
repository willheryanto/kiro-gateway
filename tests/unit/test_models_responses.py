# -*- coding: utf-8 -*-

"""
Tests for Responses API Pydantic models.

Covers request parsing, validation, edge cases, and malformed inputs.
"""

import pytest
from pydantic import ValidationError

from kiro.models_responses import (
    ResponsesApiRequest,
    ResponseInputItemParam,
    ResponseToolFunction,
    ResponseReasoning,
    ResponseInputMessage,
    ResponseInputFunctionCall,
    ResponseInputFunctionCallOutput,
)


# =============================================================================
# ResponsesApiRequest - Basic Parsing
# =============================================================================

class TestResponsesApiRequestBasic:
    """Tests for basic request parsing and defaults."""

    def test_minimal_request_with_model_only(self):
        """Minimal valid request requires only model."""
        req = ResponsesApiRequest(model="claude-sonnet-4")
        assert req.model == "claude-sonnet-4"
        assert req.stream is True  # default
        assert req.input is None
        assert req.instructions is None
        assert req.tools is None

    def test_request_with_string_input(self):
        """Input can be a plain string."""
        req = ResponsesApiRequest(model="claude-sonnet-4", input="Hello world")
        assert req.input == "Hello world"

    def test_request_with_list_input(self):
        """Input can be a list of items."""
        items = [
            {"type": "message", "role": "user", "content": "Hello"},
            {"type": "message", "role": "assistant", "content": "Hi there"},
        ]
        req = ResponsesApiRequest(model="claude-sonnet-4", input=items)
        assert isinstance(req.input, list)
        assert len(req.input) == 2

    def test_request_with_instructions(self):
        """Instructions field is parsed correctly."""
        req = ResponsesApiRequest(
            model="claude-sonnet-4",
            instructions="You are a helpful assistant.",
            input="Hello",
        )
        assert req.instructions == "You are a helpful assistant."

    def test_request_with_tools(self):
        """Tools list is parsed correctly."""
        tools = [
            {"type": "function", "name": "get_weather", "parameters": {"type": "object"}},
        ]
        req = ResponsesApiRequest(model="claude-sonnet-4", tools=tools)
        assert len(req.tools) == 1

    def test_request_with_reasoning(self):
        """Reasoning configuration is parsed."""
        req = ResponsesApiRequest(
            model="claude-sonnet-4",
            reasoning={"effort": "high", "summary": "auto"},
        )
        assert req.reasoning is not None
        assert req.reasoning.effort == "high"
        assert req.reasoning.summary == "auto"

    def test_stream_defaults_to_true(self):
        """Stream should default to True (Codex always streams)."""
        req = ResponsesApiRequest(model="claude-sonnet-4")
        assert req.stream is True

    def test_stream_can_be_false(self):
        """Stream can be explicitly set to False."""
        req = ResponsesApiRequest(model="claude-sonnet-4", stream=False)
        assert req.stream is False

    def test_extra_fields_are_allowed(self):
        """Unknown fields should not cause validation errors."""
        req = ResponsesApiRequest(
            model="claude-sonnet-4",
            input="Hello",
            unknown_field="some_value",
            another_field=42,
        )
        assert req.model == "claude-sonnet-4"


# =============================================================================
# ResponsesApiRequest - Complex Inputs
# =============================================================================

class TestResponsesApiRequestComplexInputs:
    """Tests for complex input item structures."""

    def test_function_call_items(self):
        """Function call items in input list."""
        items = [
            {"type": "message", "role": "user", "content": "What's the weather?"},
            {
                "type": "function_call",
                "call_id": "call_abc",
                "name": "get_weather",
                "arguments": '{"location": "NYC"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_abc",
                "output": "Sunny, 72°F",
            },
        ]
        req = ResponsesApiRequest(model="claude-sonnet-4", input=items)
        assert len(req.input) == 3

    def test_reasoning_items_in_input(self):
        """Reasoning items from previous turns."""
        items = [
            {"type": "message", "role": "user", "content": "Solve this"},
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking..."}]},
            {"type": "message", "role": "assistant", "content": "The answer is 42"},
        ]
        req = ResponsesApiRequest(model="claude-sonnet-4", input=items)
        assert len(req.input) == 3

    def test_content_as_list_of_blocks(self):
        """Message content can be a list of content blocks."""
        items = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image"},
                    {"type": "input_image", "image_url": "data:image/png;base64,abc123"},
                ],
            },
        ]
        req = ResponsesApiRequest(model="claude-sonnet-4", input=items)
        assert isinstance(req.input[0]["content"], list)

    def test_previous_response_id(self):
        """Previous response ID for continuation."""
        req = ResponsesApiRequest(
            model="claude-sonnet-4",
            input="Continue",
            previous_response_id="resp_abc123",
        )
        assert req.previous_response_id == "resp_abc123"

    def test_service_tier(self):
        """Service tier preference."""
        req = ResponsesApiRequest(
            model="claude-sonnet-4",
            input="Hello",
            service_tier="fast",
        )
        assert req.service_tier == "fast"


# =============================================================================
# ResponsesApiRequest - Validation Edge Cases
# =============================================================================

class TestResponsesApiRequestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_missing_model_raises_error(self):
        """Model is required."""
        with pytest.raises(ValidationError):
            ResponsesApiRequest()

    def test_empty_string_model(self):
        """Empty string model is technically valid at Pydantic level."""
        req = ResponsesApiRequest(model="")
        assert req.model == ""

    def test_none_input(self):
        """None input is valid."""
        req = ResponsesApiRequest(model="claude-sonnet-4", input=None)
        assert req.input is None

    def test_empty_list_input(self):
        """Empty list input is rejected by validator."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="input must not be an empty list"):
            ResponsesApiRequest(model="claude-sonnet-4", input=[])

    def test_empty_tools_list(self):
        """Empty tools list is valid."""
        req = ResponsesApiRequest(model="claude-sonnet-4", tools=[])
        assert req.tools == []

    def test_max_output_tokens(self):
        """Max output tokens field."""
        req = ResponsesApiRequest(model="claude-sonnet-4", max_output_tokens=4096)
        assert req.max_output_tokens == 4096

    def test_temperature_and_top_p(self):
        """Temperature and top_p fields."""
        req = ResponsesApiRequest(model="claude-sonnet-4", temperature=0.7, top_p=0.9)
        assert req.temperature == 0.7
        assert req.top_p == 0.9


# =============================================================================
# ResponseInputItemParam
# =============================================================================

class TestResponseInputItemParam:
    """Tests for the generic input item model."""

    def test_message_item(self):
        """Parse a message input item."""
        item = ResponseInputItemParam(
            type="message", role="user", content="Hello"
        )
        assert item.type == "message"
        assert item.role == "user"
        assert item.content == "Hello"

    def test_function_call_item(self):
        """Parse a function call input item."""
        item = ResponseInputItemParam(
            type="function_call",
            call_id="call_1",
            name="get_weather",
            arguments='{"city": "NYC"}',
        )
        assert item.type == "function_call"
        assert item.name == "get_weather"
        assert item.call_id == "call_1"

    def test_function_call_output_item(self):
        """Parse a function call output item."""
        item = ResponseInputItemParam(
            type="function_call_output",
            call_id="call_1",
            output="Sunny, 72°F",
        )
        assert item.type == "function_call_output"
        assert item.output == "Sunny, 72°F"

    def test_extra_fields_allowed(self):
        """Unknown fields should not cause errors."""
        item = ResponseInputItemParam(
            type="message",
            role="user",
            content="Hello",
            custom_field="value",
        )
        assert item.type == "message"


# =============================================================================
# ResponseToolFunction
# =============================================================================

class TestResponseToolFunction:
    """Tests for tool function definitions."""

    def test_basic_tool(self):
        """Parse a basic tool definition."""
        tool = ResponseToolFunction(
            name="get_weather",
            description="Get weather for a location",
            parameters={"type": "object", "properties": {"city": {"type": "string"}}},
        )
        assert tool.name == "get_weather"
        assert tool.description == "Get weather for a location"

    def test_tool_without_description(self):
        """Tool without description is valid."""
        tool = ResponseToolFunction(name="do_something")
        assert tool.name == "do_something"
        assert tool.description is None

    def test_tool_with_strict(self):
        """Tool with strict mode."""
        tool = ResponseToolFunction(name="calc", strict=True)
        assert tool.strict is True


# =============================================================================
# ResponseReasoning
# =============================================================================

class TestResponseReasoning:
    """Tests for reasoning configuration."""

    def test_reasoning_with_effort(self):
        """Reasoning with effort level."""
        r = ResponseReasoning(effort="high")
        assert r.effort == "high"

    def test_reasoning_with_summary(self):
        """Reasoning with summary mode."""
        r = ResponseReasoning(summary="auto")
        assert r.summary == "auto"

    def test_reasoning_empty(self):
        """Empty reasoning is valid."""
        r = ResponseReasoning()
        assert r.effort is None
        assert r.summary is None
