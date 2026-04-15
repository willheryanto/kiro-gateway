# -*- coding: utf-8 -*-

"""
Tests for Responses API streaming logic.

Covers SSE event generation, token-by-token streaming, thinking/reasoning,
tool calls, usage calculation, error handling, and edge cases.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from kiro.streaming_responses import (
    _generate_response_id,
    _generate_item_id,
    _generate_content_part_id,
    _sse_event,
    stream_kiro_to_responses_internal,
)
from kiro.streaming_core import KiroEvent, FirstTokenTimeoutError


# =============================================================================
# ID Generation
# =============================================================================

class TestIdGeneration:
    """Tests for ID generation functions."""

    def test_response_id_format(self):
        rid = _generate_response_id()
        assert rid.startswith("resp_")
        assert len(rid) > 5

    def test_item_id_format(self):
        iid = _generate_item_id()
        assert iid.startswith("item_")
        assert len(iid) > 5

    def test_content_part_id_format(self):
        cpid = _generate_content_part_id()
        assert cpid.startswith("cp_")
        assert len(cpid) > 3

    def test_ids_are_unique(self):
        ids = {_generate_response_id() for _ in range(100)}
        assert len(ids) == 100

    def test_item_ids_are_unique(self):
        ids = {_generate_item_id() for _ in range(100)}
        assert len(ids) == 100


# =============================================================================
# SSE Event Formatting
# =============================================================================

class TestSseEvent:
    """Tests for _sse_event formatting."""

    def test_basic_event(self):
        result = _sse_event("response.created", {"response": {"id": "resp_1"}})
        assert result.startswith("event: response.created\n")
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_event_contains_type(self):
        result = _sse_event("response.output_text.delta", {"delta": "Hello"})
        data_line = result.split("data: ")[1].strip()
        parsed = json.loads(data_line)
        assert parsed["type"] == "response.output_text.delta"
        assert parsed["delta"] == "Hello"

    def test_event_unicode(self):
        result = _sse_event("response.output_text.delta", {"delta": "こんにちは"})
        data_line = result.split("data: ")[1].strip()
        parsed = json.loads(data_line)
        assert parsed["delta"] == "こんにちは"

    def test_event_empty_data(self):
        result = _sse_event("response.created", {})
        data_line = result.split("data: ")[1].strip()
        parsed = json.loads(data_line)
        assert parsed["type"] == "response.created"


# =============================================================================
# Stream Processing
# =============================================================================

class TestStreamKiroToResponsesInternal:
    """Tests for stream_kiro_to_responses_internal."""

    @pytest.mark.asyncio
    async def test_simple_text_stream(self):
        """Simple text content produces correct SSE events."""
        events = [
            KiroEvent(type="content", content="Hello"),
            KiroEvent(type="content", content=" World"),
        ]

        with patch("kiro.streaming_responses.parse_kiro_stream") as mock_parse:
            async def mock_gen(*args, **kwargs):
                for e in events:
                    yield e
            mock_parse.return_value = mock_gen()

            mock_response = MagicMock()
            mock_client = MagicMock()
            mock_cache = MagicMock()
            mock_cache.get_max_input_tokens.return_value = 200000
            mock_auth = MagicMock()

            chunks = []
            async for chunk in stream_kiro_to_responses_internal(
                mock_client, mock_response, "claude-sonnet-4",
                mock_cache, mock_auth,
            ):
                chunks.append(chunk)

        # Parse all events
        event_types = []
        for chunk in chunks:
            for line in chunk.strip().split("\n"):
                if line.startswith("event: "):
                    event_types.append(line[7:])

        assert "response.created" in event_types
        assert "response.output_item.added" in event_types
        assert "response.content_part.added" in event_types
        assert "response.output_text.delta" in event_types
        assert "response.output_text.done" in event_types
        assert "response.content_part.done" in event_types
        assert "response.output_item.done" in event_types
        assert "response.completed" in event_types

    @pytest.mark.asyncio
    async def test_text_deltas_contain_content(self):
        """Text delta events contain the actual content."""
        events = [
            KiroEvent(type="content", content="Hello"),
            KiroEvent(type="content", content=" World"),
        ]

        with patch("kiro.streaming_responses.parse_kiro_stream") as mock_parse:
            async def mock_gen(*args, **kwargs):
                for e in events:
                    yield e
            mock_parse.return_value = mock_gen()

            mock_response = MagicMock()
            mock_client = MagicMock()
            mock_cache = MagicMock()
            mock_cache.get_max_input_tokens.return_value = 200000
            mock_auth = MagicMock()

            deltas = []
            async for chunk in stream_kiro_to_responses_internal(
                mock_client, mock_response, "claude-sonnet-4",
                mock_cache, mock_auth,
            ):
                for line in chunk.strip().split("\n"):
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("type") == "response.output_text.delta":
                            deltas.append(data["delta"])

        assert deltas == ["Hello", " World"]

    @pytest.mark.asyncio
    async def test_thinking_stream(self):
        """Thinking content produces reasoning summary events."""
        events = [
            KiroEvent(type="thinking", thinking_content="Let me think..."),
            KiroEvent(type="content", content="The answer is 42"),
        ]

        with patch("kiro.streaming_responses.parse_kiro_stream") as mock_parse:
            async def mock_gen(*args, **kwargs):
                for e in events:
                    yield e
            mock_parse.return_value = mock_gen()

            mock_response = MagicMock()
            mock_client = MagicMock()
            mock_cache = MagicMock()
            mock_cache.get_max_input_tokens.return_value = 200000
            mock_auth = MagicMock()

            event_types = []
            async for chunk in stream_kiro_to_responses_internal(
                mock_client, mock_response, "claude-sonnet-4",
                mock_cache, mock_auth,
            ):
                for line in chunk.strip().split("\n"):
                    if line.startswith("event: "):
                        event_types.append(line[7:])

        assert "response.reasoning_summary_part.added" in event_types
        assert "response.reasoning_summary_text.delta" in event_types
        assert "response.reasoning_summary_text.done" in event_types
        assert "response.output_text.delta" in event_types

    @pytest.mark.asyncio
    async def test_tool_call_stream(self):
        """Tool calls produce function_call output items."""
        events = [
            KiroEvent(type="tool_use", tool_use={
                "id": "call_abc",
                "name": "get_weather",
                "arguments": '{"city": "NYC"}',
            }),
        ]

        with patch("kiro.streaming_responses.parse_kiro_stream") as mock_parse:
            async def mock_gen(*args, **kwargs):
                for e in events:
                    yield e
            mock_parse.return_value = mock_gen()

            mock_response = MagicMock()
            mock_client = MagicMock()
            mock_cache = MagicMock()
            mock_cache.get_max_input_tokens.return_value = 200000
            mock_auth = MagicMock()

            tool_items = []
            async for chunk in stream_kiro_to_responses_internal(
                mock_client, mock_response, "claude-sonnet-4",
                mock_cache, mock_auth,
            ):
                for line in chunk.strip().split("\n"):
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("type") == "response.output_item.done":
                            item = data.get("item", {})
                            if item.get("type") == "function_call":
                                tool_items.append(item)

        assert len(tool_items) == 1
        assert tool_items[0]["name"] == "get_weather"
        assert tool_items[0]["call_id"] == "call_abc"

    @pytest.mark.asyncio
    async def test_empty_stream(self):
        """Empty stream still produces created and completed events."""
        with patch("kiro.streaming_responses.parse_kiro_stream") as mock_parse:
            async def mock_gen(*args, **kwargs):
                return
                yield  # make it an async generator
            mock_parse.return_value = mock_gen()

            mock_response = MagicMock()
            mock_client = MagicMock()
            mock_cache = MagicMock()
            mock_cache.get_max_input_tokens.return_value = 200000
            mock_auth = MagicMock()

            event_types = []
            async for chunk in stream_kiro_to_responses_internal(
                mock_client, mock_response, "claude-sonnet-4",
                mock_cache, mock_auth,
            ):
                for line in chunk.strip().split("\n"):
                    if line.startswith("event: "):
                        event_types.append(line[7:])

        assert "response.created" in event_types
        assert "response.completed" in event_types
        # No text events since no content
        assert "response.output_text.delta" not in event_types

    @pytest.mark.asyncio
    async def test_usage_in_completed_event(self):
        """Completed event includes usage information."""
        events = [
            KiroEvent(type="content", content="Hello"),
            KiroEvent(type="usage", usage={"outputTokenCount": 5}),
            KiroEvent(type="context_usage", context_usage_percentage=10.0),
        ]

        with patch("kiro.streaming_responses.parse_kiro_stream") as mock_parse:
            async def mock_gen(*args, **kwargs):
                for e in events:
                    yield e
            mock_parse.return_value = mock_gen()

            mock_response = MagicMock()
            mock_client = MagicMock()
            mock_cache = MagicMock()
            mock_cache.get_max_input_tokens.return_value = 200000
            mock_auth = MagicMock()

            completed_data = None
            async for chunk in stream_kiro_to_responses_internal(
                mock_client, mock_response, "claude-sonnet-4",
                mock_cache, mock_auth,
            ):
                for line in chunk.strip().split("\n"):
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        if data.get("type") == "response.completed":
                            completed_data = data

        assert completed_data is not None
        usage = completed_data["response"]["usage"]
        assert "input_tokens" in usage
        assert "output_tokens" in usage
        assert "total_tokens" in usage

    @pytest.mark.asyncio
    async def test_first_token_timeout_propagated(self):
        """FirstTokenTimeoutError is propagated for retry logic."""
        with patch("kiro.streaming_responses.parse_kiro_stream") as mock_parse:
            async def mock_gen(*args, **kwargs):
                raise FirstTokenTimeoutError("timeout")
                yield  # make it an async generator
            mock_parse.return_value = mock_gen()

            mock_response = MagicMock()
            mock_client = MagicMock()
            mock_cache = MagicMock()
            mock_auth = MagicMock()

            with pytest.raises(FirstTokenTimeoutError):
                async for _ in stream_kiro_to_responses_internal(
                    mock_client, mock_response, "claude-sonnet-4",
                    mock_cache, mock_auth,
                ):
                    pass

    @pytest.mark.asyncio
    async def test_model_name_in_events(self):
        """Model name is included in created and completed events."""
        events = [KiroEvent(type="content", content="Hi")]

        with patch("kiro.streaming_responses.parse_kiro_stream") as mock_parse:
            async def mock_gen(*args, **kwargs):
                for e in events:
                    yield e
            mock_parse.return_value = mock_gen()

            mock_response = MagicMock()
            mock_client = MagicMock()
            mock_cache = MagicMock()
            mock_cache.get_max_input_tokens.return_value = 200000
            mock_auth = MagicMock()

            models_seen = []
            async for chunk in stream_kiro_to_responses_internal(
                mock_client, mock_response, "claude-opus-4.6",
                mock_cache, mock_auth,
            ):
                for line in chunk.strip().split("\n"):
                    if line.startswith("data: "):
                        data = json.loads(line[6:])
                        resp = data.get("response", {})
                        if "model" in resp:
                            models_seen.append(resp["model"])

        assert all(m == "claude-opus-4.6" for m in models_seen)
        assert len(models_seen) >= 2  # at least created + completed
