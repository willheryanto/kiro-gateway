# -*- coding: utf-8 -*-

"""
Tests for Responses API converters.

Covers input item conversion, tool conversion, payload building,
edge cases, and malformed inputs.
"""

import pytest
from unittest.mock import patch

from kiro.converters_responses import (
    convert_responses_input_to_unified,
    convert_responses_tools_to_unified,
    build_kiro_payload_responses,
    _extract_text_from_response_content,
    _extract_images_from_response_content,
    _flush_tool_pairs,
)
from kiro.converters_core import UnifiedMessage, UnifiedTool
from kiro.models_responses import ResponsesApiRequest


# =============================================================================
# Text Extraction
# =============================================================================

class TestExtractTextFromResponseContent:
    """Tests for _extract_text_from_response_content."""

    def test_string_content(self):
        assert _extract_text_from_response_content("Hello") == "Hello"

    def test_none_content(self):
        assert _extract_text_from_response_content(None) == ""

    def test_input_text_blocks(self):
        content = [{"type": "input_text", "text": "Hello "}, {"type": "input_text", "text": "World"}]
        assert _extract_text_from_response_content(content) == "Hello World"

    def test_output_text_blocks(self):
        content = [{"type": "output_text", "text": "Response here"}]
        assert _extract_text_from_response_content(content) == "Response here"

    def test_mixed_block_types(self):
        content = [
            {"type": "input_text", "text": "A"},
            {"type": "input_image", "image_url": "data:image/png;base64,abc"},
            {"type": "text", "text": "B"},
        ]
        assert _extract_text_from_response_content(content) == "AB"

    def test_string_items_in_list(self):
        content = ["Hello", " ", "World"]
        assert _extract_text_from_response_content(content) == "Hello World"

    def test_empty_list(self):
        assert _extract_text_from_response_content([]) == ""

    def test_non_string_non_list(self):
        assert _extract_text_from_response_content(42) == "42"

    def test_blocks_without_text_key(self):
        content = [{"type": "unknown", "data": "something"}]
        assert _extract_text_from_response_content(content) == ""

    def test_block_with_text_key_no_type(self):
        """Block with 'text' key but no recognized type still extracts text."""
        content = [{"text": "fallback"}]
        assert _extract_text_from_response_content(content) == "fallback"


# =============================================================================
# Image Extraction
# =============================================================================

class TestExtractImagesFromResponseContent:
    """Tests for _extract_images_from_response_content."""

    def test_no_images(self):
        content = [{"type": "input_text", "text": "Hello"}]
        assert _extract_images_from_response_content(content) == []

    def test_data_url_image(self):
        content = [{"type": "input_image", "image_url": "data:image/jpeg;base64,/9j/abc"}]
        images = _extract_images_from_response_content(content)
        assert len(images) == 1
        assert images[0]["media_type"] == "image/jpeg"
        assert images[0]["data"] == "/9j/abc"

    def test_image_url_as_dict(self):
        content = [{"type": "input_image", "image_url": {"url": "data:image/png;base64,abc123"}}]
        images = _extract_images_from_response_content(content)
        assert len(images) == 1
        assert images[0]["media_type"] == "image/png"

    def test_non_data_url_ignored(self):
        """Non-data URLs are not extracted (we only support base64)."""
        content = [{"type": "input_image", "image_url": "https://example.com/img.jpg"}]
        assert _extract_images_from_response_content(content) == []

    def test_string_content_returns_empty(self):
        assert _extract_images_from_response_content("Hello") == []

    def test_none_content_returns_empty(self):
        assert _extract_images_from_response_content(None) == []

    def test_malformed_data_url(self):
        """Malformed data URL should not crash."""
        content = [{"type": "input_image", "image_url": "data:broken"}]
        images = _extract_images_from_response_content(content)
        assert images == []


# =============================================================================
# Input Conversion
# =============================================================================

class TestConvertResponsesInputToUnified:
    """Tests for convert_responses_input_to_unified."""

    def test_string_input(self):
        """String input becomes a single user message."""
        system, messages = convert_responses_input_to_unified("Hello world")
        assert system == ""
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].content == "Hello world"

    def test_string_input_with_instructions(self):
        """Instructions become system prompt."""
        system, messages = convert_responses_input_to_unified(
            "Hello", instructions="Be helpful"
        )
        assert system == "Be helpful"
        assert len(messages) == 1

    def test_simple_message_items(self):
        """Basic message items are converted."""
        items = [
            {"type": "message", "role": "user", "content": "Hello"},
            {"type": "message", "role": "assistant", "content": "Hi there"},
            {"type": "message", "role": "user", "content": "How are you?"},
        ]
        system, messages = convert_responses_input_to_unified(items)
        assert system == ""
        assert len(messages) == 3
        assert messages[0].role == "user"
        assert messages[1].role == "assistant"
        assert messages[2].role == "user"

    def test_system_message_extracted_to_system_prompt(self):
        """System messages become part of system prompt."""
        items = [
            {"type": "message", "role": "system", "content": "You are helpful"},
            {"type": "message", "role": "user", "content": "Hello"},
        ]
        system, messages = convert_responses_input_to_unified(items)
        assert "You are helpful" in system
        assert len(messages) == 1
        assert messages[0].role == "user"

    def test_developer_message_extracted_to_system_prompt(self):
        """Developer messages become part of system prompt."""
        items = [
            {"type": "message", "role": "developer", "content": "Custom instructions"},
            {"type": "message", "role": "user", "content": "Hello"},
        ]
        system, messages = convert_responses_input_to_unified(items)
        assert "Custom instructions" in system
        assert len(messages) == 1

    def test_instructions_combined_with_system_messages(self):
        """Instructions and system messages are combined."""
        items = [
            {"type": "message", "role": "system", "content": "Extra context"},
            {"type": "message", "role": "user", "content": "Hello"},
        ]
        system, messages = convert_responses_input_to_unified(
            items, instructions="Base instructions"
        )
        assert "Base instructions" in system
        assert "Extra context" in system

    def test_function_call_and_output(self):
        """Function calls and outputs are paired correctly."""
        items = [
            {"type": "message", "role": "user", "content": "What's the weather?"},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "NYC"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Sunny, 72°F",
            },
            {"type": "message", "role": "assistant", "content": "It's sunny!"},
        ]
        system, messages = convert_responses_input_to_unified(items)
        assert len(messages) == 4  # user, assistant(tool_calls), user(tool_results), assistant

        # Check tool call
        assert messages[1].role == "assistant"
        assert messages[1].tool_calls is not None
        assert len(messages[1].tool_calls) == 1
        assert messages[1].tool_calls[0]["function"]["name"] == "get_weather"

        # Check tool result
        assert messages[2].role == "user"
        assert messages[2].tool_results is not None
        assert len(messages[2].tool_results) == 1
        assert messages[2].tool_results[0]["content"] == "Sunny, 72°F"

    def test_multiple_function_calls(self):
        """Multiple function calls are grouped together."""
        items = [
            {"type": "message", "role": "user", "content": "Compare weather"},
            {"type": "function_call", "call_id": "c1", "name": "get_weather", "arguments": '{"city": "NYC"}'},
            {"type": "function_call", "call_id": "c2", "name": "get_weather", "arguments": '{"city": "LA"}'},
            {"type": "function_call_output", "call_id": "c1", "output": "Sunny"},
            {"type": "function_call_output", "call_id": "c2", "output": "Cloudy"},
        ]
        system, messages = convert_responses_input_to_unified(items)
        # user, assistant(2 tool_calls), user(2 tool_results)
        assert len(messages) == 3
        assert len(messages[1].tool_calls) == 2
        assert len(messages[2].tool_results) == 2

    def test_reasoning_items_skipped(self):
        """Reasoning items are skipped."""
        items = [
            {"type": "message", "role": "user", "content": "Hello"},
            {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": "thinking"}]},
            {"type": "message", "role": "assistant", "content": "Hi"},
        ]
        system, messages = convert_responses_input_to_unified(items)
        assert len(messages) == 2

    def test_empty_input(self):
        """Empty input list returns empty messages."""
        system, messages = convert_responses_input_to_unified([])
        assert messages == []

    def test_none_input(self):
        """None input returns empty messages."""
        system, messages = convert_responses_input_to_unified(None)
        assert messages == []

    def test_unknown_item_types_skipped(self):
        """Unknown item types are silently skipped."""
        items = [
            {"type": "message", "role": "user", "content": "Hello"},
            {"type": "unknown_type", "data": "something"},
        ]
        system, messages = convert_responses_input_to_unified(items)
        assert len(messages) == 1

    def test_function_call_output_with_empty_output(self):
        """Empty function output gets placeholder."""
        items = [
            {"type": "function_call", "call_id": "c1", "name": "do_thing", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "c1", "output": ""},
        ]
        system, messages = convert_responses_input_to_unified(items)
        assert messages[1].tool_results[0]["content"] == "(empty result)"

    def test_content_with_image_blocks(self):
        """Image blocks in content are extracted."""
        items = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this"},
                    {"type": "input_image", "image_url": "data:image/png;base64,abc123"},
                ],
            },
        ]
        system, messages = convert_responses_input_to_unified(items)
        assert len(messages) == 1
        assert messages[0].content == "Describe this"
        assert messages[0].images is not None
        assert len(messages[0].images) == 1


# =============================================================================
# Tool Conversion
# =============================================================================

class TestConvertResponsesToolsToUnified:
    """Tests for convert_responses_tools_to_unified."""

    def test_basic_function_tool(self):
        """Basic function tool conversion."""
        tools = [
            {"type": "function", "name": "get_weather", "description": "Get weather", "parameters": {"type": "object"}},
        ]
        result = convert_responses_tools_to_unified(tools)
        assert len(result) == 1
        assert result[0].name == "get_weather"
        assert result[0].description == "Get weather"

    def test_tool_without_description(self):
        """Tool without description."""
        tools = [{"type": "function", "name": "do_thing"}]
        result = convert_responses_tools_to_unified(tools)
        assert len(result) == 1
        assert result[0].description is None

    def test_nested_function_format(self):
        """OpenAI Chat Completions style nested function format."""
        tools = [
            {"type": "function", "function": {"name": "calc", "description": "Calculate", "parameters": {}}},
        ]
        result = convert_responses_tools_to_unified(tools)
        assert len(result) == 1
        assert result[0].name == "calc"

    def test_client_side_tools_skipped(self):
        """Non-convertible client-side tools (web_search, local_shell, etc.) are skipped."""
        tools = [
            {"type": "web_search"},
            {"type": "local_shell"},
            {"type": "function", "name": "valid_tool"},
        ]
        result = convert_responses_tools_to_unified(tools)
        assert len(result) == 1
        assert result[0].name == "valid_tool"

    def test_custom_tool_converted_to_function(self):
        """Custom/freeform tools (like apply_patch) are converted to function tools."""
        tools = [
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "Use the apply_patch tool to edit files.",
                "format": {"type": "grammar", "syntax": "lark", "definition": "..."},
            },
            {"type": "function", "name": "valid_tool"},
        ]
        result = convert_responses_tools_to_unified(tools)
        assert len(result) == 2
        ap = result[0]
        assert ap.name == "apply_patch"
        assert ap.description == "Use the apply_patch tool to edit files."
        assert ap.input_schema is not None
        assert ap.input_schema["type"] == "object"
        assert "input" in ap.input_schema["properties"]
        assert ap.input_schema["properties"]["input"]["type"] == "string"
        assert ap.input_schema["required"] == ["input"]
        assert result[1].name == "valid_tool"

    def test_custom_tool_without_name_skipped(self):
        """Custom tools without a name are skipped."""
        tools = [
            {"type": "custom", "description": "no name tool"},
            {"type": "function", "name": "valid_tool"},
        ]
        result = convert_responses_tools_to_unified(tools)
        assert len(result) == 1
        assert result[0].name == "valid_tool"

    def test_custom_tool_minimal_fields(self):
        """Custom tool with only name gets converted with empty description."""
        tools = [{"type": "custom", "name": "my_tool"}]
        result = convert_responses_tools_to_unified(tools)
        assert len(result) == 1
        assert result[0].name == "my_tool"
        assert result[0].description == ""
        assert result[0].input_schema["required"] == ["input"]

    def test_unknown_tool_types_dropped(self):
        """Truly unknown tool types are dropped with a warning."""
        tools = [
            {"type": "totally_unknown", "name": "mystery"},
            {"type": "function", "name": "valid_tool"},
        ]
        result = convert_responses_tools_to_unified(tools)
        assert len(result) == 1
        assert result[0].name == "valid_tool"

    def test_none_tools(self):
        """None tools returns None."""
        assert convert_responses_tools_to_unified(None) is None

    def test_empty_tools(self):
        """Empty tools list returns None."""
        assert convert_responses_tools_to_unified([]) is None

    def test_non_dict_items_skipped(self):
        """Non-dict items in tools list are skipped."""
        tools = ["not_a_tool", {"type": "function", "name": "valid"}]
        result = convert_responses_tools_to_unified(tools)
        assert len(result) == 1

    def test_multiple_tools(self):
        """Multiple tools are all converted."""
        tools = [
            {"type": "function", "name": "tool_a", "description": "A"},
            {"type": "function", "name": "tool_b", "description": "B"},
            {"type": "function", "name": "tool_c", "description": "C"},
        ]
        result = convert_responses_tools_to_unified(tools)
        assert len(result) == 3


# =============================================================================
# Payload Building
# =============================================================================

class TestBuildKiroPayloadResponses:
    """Tests for build_kiro_payload_responses."""

    @patch("kiro.converters_responses.get_model_id_for_kiro", return_value="claude-sonnet-4")
    def test_basic_payload(self, mock_model_id):
        """Basic payload building from simple request."""
        request = ResponsesApiRequest(
            model="claude-sonnet-4",
            input=[{"type": "message", "role": "user", "content": "Hello"}],
        )
        payload = build_kiro_payload_responses(request, "conv-123", "arn:test")
        assert "conversationState" in payload
        assert payload["conversationState"]["conversationId"] == "conv-123"

    @patch("kiro.converters_responses.get_model_id_for_kiro", return_value="claude-sonnet-4")
    def test_payload_with_instructions(self, mock_model_id):
        """Instructions are included in the payload."""
        request = ResponsesApiRequest(
            model="claude-sonnet-4",
            instructions="Be concise",
            input=[{"type": "message", "role": "user", "content": "Hello"}],
        )
        payload = build_kiro_payload_responses(request, "conv-123", "arn:test")
        # Instructions should be in the system prompt, which gets prepended to content
        current_msg = payload["conversationState"]["currentMessage"]["userInputMessage"]
        assert "Be concise" in current_msg["content"]

    @patch("kiro.converters_responses.get_model_id_for_kiro", return_value="claude-sonnet-4")
    def test_payload_with_tools(self, mock_model_id):
        """Tools are included in the payload."""
        request = ResponsesApiRequest(
            model="claude-sonnet-4",
            input=[{"type": "message", "role": "user", "content": "Hello"}],
            tools=[{"type": "function", "name": "test_tool", "description": "A test"}],
        )
        payload = build_kiro_payload_responses(request, "conv-123", "arn:test")
        context = payload["conversationState"]["currentMessage"]["userInputMessage"].get("userInputMessageContext", {})
        assert "tools" in context

    @patch("kiro.converters_responses.get_model_id_for_kiro", return_value="claude-sonnet-4")
    def test_payload_with_string_input(self, mock_model_id):
        """String input is handled correctly."""
        request = ResponsesApiRequest(
            model="claude-sonnet-4",
            input="Just a simple question",
        )
        payload = build_kiro_payload_responses(request, "conv-123", "arn:test")
        current_msg = payload["conversationState"]["currentMessage"]["userInputMessage"]
        assert "simple question" in current_msg["content"]

    def test_payload_empty_input_raises(self):
        """Empty input should raise ValidationError at model level."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError, match="input must not be an empty list"):
            ResponsesApiRequest(model="claude-sonnet-4", input=[])

    @patch("kiro.converters_responses.get_model_id_for_kiro", return_value="claude-sonnet-4")
    def test_payload_with_history(self, mock_model_id):
        """Multi-turn conversation creates history."""
        request = ResponsesApiRequest(
            model="claude-sonnet-4",
            input=[
                {"type": "message", "role": "user", "content": "Hello"},
                {"type": "message", "role": "assistant", "content": "Hi there"},
                {"type": "message", "role": "user", "content": "How are you?"},
            ],
        )
        payload = build_kiro_payload_responses(request, "conv-123", "arn:test")
        assert "history" in payload["conversationState"]
        assert len(payload["conversationState"]["history"]) > 0


# ==================================================================================================
# Apply Patch Tool Injection Tests
# ==================================================================================================

