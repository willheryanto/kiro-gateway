# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/jwadow/kiro-gateway
# Copyright (C) 2025 Jwadow
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Streaming conversion from Kiro API format to the OpenAI Responses API.

Emits the SSE event sequence a Responses client expects:

    response.created
    response.output_text.delta        (repeated, incremental text)
    response.output_item.done         (assistant message, if any text)
    response.output_item.done         (one per function call)
    response.completed               (terminal, carries usage)

Two properties of this protocol drive the implementation:

1. ``response.completed`` is mandatory and terminal. A client that never
   receives it treats the turn as failed even if all the text arrived.
2. Tool calls must be delivered whole inside ``response.output_item.done``.
   Incremental ``response.function_call_arguments.delta`` events are ignored by
   the Codex CLI, so a call streamed only as fragments is never seen. Kiro
   yields complete tool calls anyway, so they are buffered until the stream
   drains and then emitted as single items.

Reasoning events are intentionally not emitted. Kiro has no reasoning channel;
the gateway's "thinking" text is extracted from prompt-injected tags and has no
relationship to the opaque, provider-signed ``encrypted_content`` that Responses
clients round-trip. The thinking parser still runs so those tags are stripped
from the visible output rather than leaking into the text deltas.

Uses streaming_core.py for parsing the Kiro stream into unified KiroEvent objects.
"""

import json
from typing import TYPE_CHECKING, Any, AsyncGenerator, Awaitable, Callable, Dict, List, Optional, Set

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro.config import FIRST_TOKEN_MAX_RETRIES, FIRST_TOKEN_TIMEOUT
from kiro.converters_responses import FREEFORM_TOOL_INPUT_KEY
from kiro.parsers import deduplicate_tool_calls, parse_bracket_tool_calls
from kiro.tokenizer import count_message_tokens, count_tokens, count_tools_tokens
from kiro.utils import generate_output_item_id, generate_response_id

from kiro.streaming_core import (
    FirstTokenTimeoutError,
    calculate_tokens_from_context_usage,
    parse_kiro_stream,
    stream_with_first_token_retry as stream_with_first_token_retry_core,
)

if TYPE_CHECKING:
    from kiro.auth import KiroAuthManager
    from kiro.cache import ModelInfoCache

try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


__all__ = [
    'format_responses_event',
    'stream_kiro_to_responses_internal',
    'stream_with_first_token_retry_responses',
]


def format_responses_event(event_type: str, payload: Dict[str, Any]) -> str:
    """
    Format one Responses SSE event.

    Both the ``event:`` name and the ``type`` field inside ``data:`` are written.
    Clients read the type from the JSON body, but the event name is part of the
    wire format that real Responses servers emit, so it is included too.

    Args:
        event_type: Event type, e.g. "response.completed".
        payload: Event body. A "type" key is added automatically.

    Returns:
        A complete SSE frame, terminated by a blank line.
    """
    body = {"type": event_type, **payload}
    return f"event: {event_type}\ndata: {json.dumps(body, ensure_ascii=False)}\n\n"


def _build_tool_call_item(
    tool_call: Dict[str, Any],
    freeform_tool_names: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Build a tool-call output item from a unified tool call.

    Tools the client declared as freeform ("custom") must come back as
    ``custom_tool_call`` items carrying raw ``input``; everything else as
    ``function_call`` items with a JSON ``arguments`` string.

    Args:
        tool_call: Unified tool call, as produced by the Kiro stream parser.
        freeform_tool_names: Names declared as freeform by the client.

    Returns:
        An item for response.output_item.done.
    """
    function = tool_call.get("function") or {}
    name = function.get("name") or tool_call.get("name") or ""
    arguments = function.get("arguments")
    if arguments is None:
        arguments = tool_call.get("input")

    # Clients parse `arguments` as a JSON string, so a dict has to be encoded
    # and an empty value must still be valid JSON.
    if isinstance(arguments, dict):
        arguments = json.dumps(arguments, ensure_ascii=False)
    elif not isinstance(arguments, str) or not arguments:
        arguments = "{}"

    if freeform_tool_names and name in freeform_tool_names:
        # Kiro was given a synthetic one-string schema for this tool, so unwrap
        # that field back into the raw text the client expects.
        raw_input = arguments
        try:
            decoded = json.loads(arguments)
            if isinstance(decoded, dict):
                raw_input = decoded.get(FREEFORM_TOOL_INPUT_KEY)
                if raw_input is None:
                    # Model ignored the synthetic key; fall back to a lone value.
                    values = [v for v in decoded.values() if isinstance(v, str)]
                    raw_input = values[0] if len(values) == 1 else arguments
        except (json.JSONDecodeError, TypeError):
            pass  # Not JSON - already raw text.

        return {
            "type": "custom_tool_call",
            "id": generate_output_item_id("ctc"),
            "call_id": tool_call.get("id") or "",
            "name": name,
            "input": raw_input if isinstance(raw_input, str) else str(raw_input),
        }

    return {
        "type": "function_call",
        "id": generate_output_item_id("fc"),
        "call_id": tool_call.get("id") or "",
        "name": name,
        "arguments": arguments,
    }


async def stream_kiro_to_responses_internal(
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    response_id: Optional[str] = None,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
    freeform_tool_names: Optional[Set[str]] = None,
) -> AsyncGenerator[str, None]:
    """
    Convert a Kiro stream to Responses API SSE events.

    Raises FirstTokenTimeoutError if the first token does not arrive within
    first_token_timeout seconds, so the retry wrapper can start a new request.

    Args:
        response: HTTP response carrying the Kiro stream.
        model: Model name for logging and token limits.
        model_cache: Model cache, for context-window size.
        auth_manager: Authentication manager.
        response_id: Response id to report. Generated when omitted.
        first_token_timeout: First-token timeout in seconds.
        request_messages: Original messages, for fallback token counting.
        request_tools: Original tools, for fallback token counting.
        freeform_tool_names: Tools the client declared as freeform ("custom"),
            which must be reported back as custom_tool_call items.

    Yields:
        SSE frames in Responses format.
    """
    response_id = response_id or generate_response_id()

    full_content = ""
    full_thinking_content = ""
    tool_calls_from_stream: List[Dict[str, Any]] = []
    metering_data = None
    context_usage_percentage = None
    message_item_id: Optional[str] = None

    # Emitted before reading upstream so the client's idle timer starts now
    # rather than when Kiro produces its first token.
    yield format_responses_event("response.created", {"response": {"id": response_id}})

    async for event in parse_kiro_stream(response, first_token_timeout):
        if event.type == "content" and event.content:
            # Open the message item before the first delta. Clients track an
            # "active item" and reject deltas that arrive without one. The item
            # is opened with empty content so the client does not seed text that
            # the deltas then repeat.
            if message_item_id is None:
                message_item_id = generate_output_item_id("msg")
                yield format_responses_event(
                    "response.output_item.added",
                    {
                        "item": {
                            "type": "message",
                            "role": "assistant",
                            "id": message_item_id,
                            "content": [],
                        }
                    },
                )

            full_content += event.content
            yield format_responses_event(
                "response.output_text.delta",
                {"delta": event.content, "item_id": message_item_id},
            )

        elif event.type == "thinking" and event.thinking_content:
            # Tracked for token accounting only; see the module docstring for
            # why no reasoning events are emitted.
            full_thinking_content += event.thinking_content

        elif event.type == "tool_use" and event.tool_use:
            # Buffered, not streamed: fragments would be ignored by the client.
            tool_calls_from_stream.append(event.tool_use)

        elif event.type == "usage" and event.usage:
            metering_data = event.usage

        elif event.type == "context_usage" and event.context_usage_percentage is not None:
            context_usage_percentage = event.context_usage_percentage

    # Some models emit tool calls as bracket syntax inside the text instead of
    # structured events.
    bracket_tool_calls = parse_bracket_tool_calls(full_content)
    all_tool_calls = deduplicate_tool_calls(tool_calls_from_stream + bracket_tool_calls)

    if full_content:
        # Reuses the id from output_item.added so the client closes the item it
        # already opened rather than seeing a second, unknown one.
        yield format_responses_event(
            "response.output_item.done",
            {
                "item": {
                    "type": "message",
                    "role": "assistant",
                    "id": message_item_id or generate_output_item_id("msg"),
                    "content": [{"type": "output_text", "text": full_content}],
                }
            },
        )

    for tool_call in all_tool_calls:
        yield format_responses_event(
            "response.output_item.done",
            {"item": _build_tool_call_item(tool_call, freeform_tool_names)},
        )

    completion_tokens = count_tokens(full_content + full_thinking_content)
    prompt_tokens, total_tokens, prompt_source, _total_source = calculate_tokens_from_context_usage(
        context_usage_percentage, completion_tokens, model_cache, model
    )

    # Fallback when Kiro reported no context usage.
    if prompt_source == "unknown" and request_messages:
        prompt_tokens = count_message_tokens(request_messages, apply_claude_correction=False)
        if request_tools:
            prompt_tokens += count_tools_tokens(request_tools, apply_claude_correction=False)
        total_tokens = prompt_tokens + completion_tokens

    logger.debug(
        f"Responses stream finished: {len(full_content)} chars, "
        f"{len(all_tool_calls)} tool calls, tokens in={prompt_tokens} out={completion_tokens}"
    )

    completed_payload = {
        "response": {
            "id": response_id,
            "usage": {
                "input_tokens": prompt_tokens,
                "output_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
            # No pending tool calls means the turn is finished.
            "end_turn": not all_tool_calls,
        }
    }
    if metering_data is not None:
        completed_payload["response"]["credits_used"] = metering_data

    final_frame = format_responses_event("response.completed", completed_payload)
    if debug_logger:
        debug_logger.log_modified_chunk(final_frame.encode('utf-8'))
    yield final_frame


def make_response_failed_frame(error: Exception, response_id: Optional[str] = None) -> str:
    """
    Build a response.failed frame for a mid-stream failure.

    Once SSE headers are flushed the status code can no longer be changed, so
    the failure has to be reported in-band.

    Args:
        error: The exception that ended the stream.
        response_id: Response id to report, if known.

    Returns:
        A response.failed SSE frame.
    """
    return format_responses_event(
        "response.failed",
        {
            "response": {
                "id": response_id or generate_response_id(),
                "status": "failed",
                "error": {"code": "server_error", "message": str(error)},
            }
        },
    )


async def stream_with_first_token_retry_responses(
    make_request: Callable[[], Awaitable[httpx.Response]],
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    initial_response: Optional[httpx.Response] = None,
    max_retries: int = FIRST_TOKEN_MAX_RETRIES,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_messages: Optional[list] = None,
    request_tools: Optional[list] = None,
    freeform_tool_names: Optional[Set[str]] = None,
) -> AsyncGenerator[str, None]:
    """
    Stream Responses SSE with automatic retry on first-token timeout.

    Wraps the generic retry helper from streaming_core with Responses-specific
    stream processing and error shapes.

    Args:
        make_request: Creates a new upstream request for retries.
        model: Model name.
        model_cache: Model cache.
        auth_manager: Authentication manager.
        initial_response: Pre-validated response to use on the first attempt.
        max_retries: Maximum number of attempts.
        first_token_timeout: First-token timeout in seconds.
        request_messages: Original messages, for fallback token counting.
        request_tools: Original tools, for fallback token counting.
        freeform_tool_names: Tools the client declared as freeform ("custom").

    Yields:
        SSE frames in Responses format.

    Raises:
        HTTPException: After every attempt has failed.
    """
    def create_http_error(status_code: int, error_text: str) -> HTTPException:
        """Create HTTPException for upstream HTTP errors."""
        return HTTPException(
            status_code=status_code,
            detail=f"Upstream API error: {error_text}"
        )

    def create_timeout_error(retries: int, timeout: float) -> HTTPException:
        """Create HTTPException when all retries time out."""
        return HTTPException(
            status_code=504,
            detail=f"Model did not respond within {timeout}s after {retries} attempts. Please try again."
        )

    # Each attempt reuses the same id so the client sees one logical response.
    response_id = generate_response_id()

    async def stream_processor(response: httpx.Response) -> AsyncGenerator[str, None]:
        """Process an upstream response and yield Responses SSE frames."""
        async for chunk in stream_kiro_to_responses_internal(
            response,
            model,
            model_cache,
            auth_manager,
            response_id=response_id,
            first_token_timeout=first_token_timeout,
            request_messages=request_messages,
            request_tools=request_tools,
            freeform_tool_names=freeform_tool_names,
        ):
            yield chunk

    async for chunk in stream_with_first_token_retry_core(
        make_request=make_request,
        stream_processor=stream_processor,
        initial_response=initial_response,
        max_retries=max_retries,
        first_token_timeout=first_token_timeout,
        on_http_error=create_http_error,
        on_all_retries_failed=create_timeout_error,
    ):
        yield chunk
