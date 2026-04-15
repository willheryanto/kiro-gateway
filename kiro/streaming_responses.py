# -*- coding: utf-8 -*-

"""
Streaming logic for converting Kiro stream to Responses API format.

Converts Kiro's AWS SSE stream into OpenAI Responses API SSE events:
- response.created
- response.output_item.added
- response.content_part.added
- response.output_text.delta (token-by-token text)
- response.output_text.done
- response.content_part.done
- response.output_item.done
- response.reasoning_summary_part.added
- response.reasoning_summary_text.delta
- response.reasoning_summary_text.done
- response.reasoning_summary_part.done
- response.completed

Uses streaming_core.py for parsing Kiro stream into unified KiroEvent objects.
"""

import json
import time
import uuid
from typing import TYPE_CHECKING, AsyncGenerator, Callable, Awaitable, Optional

import httpx
from fastapi import HTTPException
from loguru import logger

from kiro.config import (
    FIRST_TOKEN_TIMEOUT,
    FIRST_TOKEN_MAX_RETRIES,
    TRUNCATION_RECOVERY,
    FAKE_REASONING_HANDLING,
)
from kiro.tokenizer import count_tokens, count_tools_tokens
from kiro.parsers import parse_bracket_tool_calls, deduplicate_tool_calls

from kiro.streaming_core import (
    parse_kiro_stream,
    FirstTokenTimeoutError,
    KiroEvent,
    calculate_tokens_from_context_usage,
    stream_with_first_token_retry as stream_with_first_token_retry_core,
)

if TYPE_CHECKING:
    from kiro.auth import KiroAuthManager
    from kiro.cache import ModelInfoCache

try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


def _generate_response_id() -> str:
    """Generate a unique response ID in Responses API format."""
    return f"resp_{uuid.uuid4().hex[:24]}"


def _generate_item_id() -> str:
    """Generate a unique item ID."""
    return f"item_{uuid.uuid4().hex[:24]}"


def _generate_content_part_id() -> str:
    """Generate a unique content part ID."""
    return f"cp_{uuid.uuid4().hex[:24]}"


def _sse_event(event_type: str, data: dict) -> str:
    """
    Format a Responses API SSE event.

    Args:
        event_type: The event type (e.g. response.output_text.delta)
        data: The event payload

    Returns:
        Formatted SSE string with event type and data lines
    """
    data["type"] = event_type
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def stream_kiro_to_responses_internal(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_input: Optional[list] = None,
    request_tools: Optional[list] = None,
    conversation_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Internal generator for converting Kiro stream to Responses API SSE format.

    Parses AWS SSE stream and converts events to Responses API events,
    enabling token-by-token streaming in Codex CLI.

    Args:
        client: HTTP client (for connection management)
        response: HTTP response with data stream
        model: Model name to include in response
        model_cache: Model cache for getting token limits
        auth_manager: Authentication manager
        first_token_timeout: First token wait timeout (seconds)
        request_input: Original request input items (for fallback token counting)
        request_tools: Original request tools (for fallback token counting)
        conversation_id: Stable conversation ID for truncation recovery

    Yields:
        Strings in SSE format with Responses API events

    Raises:
        FirstTokenTimeoutError: If first token not received within timeout
    """
    response_id = _generate_response_id()
    output_item_id = _generate_item_id()
    content_part_id = _generate_content_part_id()
    reasoning_item_id = _generate_item_id()
    reasoning_part_id = _generate_content_part_id()

    full_content = ""
    full_thinking_content = ""
    metering_data = None
    context_usage_percentage = None
    streaming_error_occurred = False
    text_started = False
    reasoning_started = False
    reasoning_closed = False
    reasoning_header_sent = False
    tool_calls_from_stream = []

    try:
        # Emit response.created
        yield _sse_event(
            "response.created",
            {
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": int(time.time()),
                    "model": model,
                    "status": "in_progress",
                    "output": [],
                },
            },
        )

        async for event in parse_kiro_stream(response, first_token_timeout):
            if event.type == "content" and event.content:
                full_content += event.content

                # Output index shifts by 1 if reasoning was emitted first
                text_output_index = 1 if reasoning_started else 0

                if not text_started:
                    text_started = True

                    # Close reasoning before text starts (Codex TUI needs
                    # reasoning fully closed before text output_item arrives)
                    if reasoning_started and not reasoning_closed:
                        reasoning_closed = True
                        full_thinking_with_header = "**Thinking**\n\n" + full_thinking_content
                        yield _sse_event(
                            "response.reasoning_summary_text.done",
                            {
                                "output_index": 0,
                                "summary_index": 0,
                                "text": full_thinking_with_header,
                            },
                        )
                        yield _sse_event(
                            "response.reasoning_summary_part.done",
                            {
                                "output_index": 0,
                                "summary_index": 0,
                                "part": {
                                    "type": "summary_text",
                                    "text": full_thinking_with_header,
                                },
                            },
                        )
                        yield _sse_event(
                            "response.output_item.done",
                            {
                                "output_index": 0,
                                "item": {
                                    "id": reasoning_item_id,
                                    "type": "reasoning",
                                    "summary": [
                                        {"type": "summary_text", "text": full_thinking_with_header},
                                    ],
                                },
                            },
                        )
                    # Emit output_item.added
                    yield _sse_event(
                        "response.output_item.added",
                        {
                            "output_index": text_output_index,
                            "item": {
                                "id": output_item_id,
                                "type": "message",
                                "role": "assistant",
                                "status": "in_progress",
                                "content": [],
                            },
                        },
                    )
                    # Emit content_part.added
                    yield _sse_event(
                        "response.content_part.added",
                        {
                            "output_index": text_output_index,
                            "content_index": 0,
                            "part": {
                                "type": "output_text",
                                "text": "",
                            },
                        },
                    )

                # Emit text delta (token-by-token)
                yield _sse_event(
                    "response.output_text.delta",
                    {
                        "output_index": text_output_index,
                        "content_index": 0,
                        "delta": event.content,
                    },
                )

                if debug_logger:
                    debug_logger.log_modified_chunk(event.content.encode("utf-8"))

            elif event.type == "thinking" and event.thinking_content:
                full_thinking_content += event.thinking_content

                # Respect FAKE_REASONING_HANDLING config
                if FAKE_REASONING_HANDLING == "remove":
                    # Silently consume thinking content without emitting
                    pass
                elif FAKE_REASONING_HANDLING == "as_reasoning_content":
                    # Emit as reasoning events (Responses API native format)
                    # Codex CLI requires an output_item.added with type=reasoning
                    # BEFORE any reasoning_summary deltas (sets active_item)
                    if not reasoning_started:
                        reasoning_started = True
                        # Emit reasoning output item first
                        yield _sse_event(
                            "response.output_item.added",
                            {
                                "output_index": 0,
                                "item": {
                                    "id": reasoning_item_id,
                                    "type": "reasoning",
                                    "summary": [],
                                },
                            },
                        )
                        yield _sse_event(
                            "response.reasoning_summary_part.added",
                            {
                                "output_index": 0,
                                "summary_index": 0,
                            },
                        )

                    # Codex TUI expects reasoning to start with **Header**


                    # to show the "Thinking" shimmer and render the block visibly.
                    # Without it, the block becomes transcript_only (invisible).
                    thinking_delta = event.thinking_content
                    if not reasoning_header_sent:
                        reasoning_header_sent = True
                        thinking_delta = "**Thinking**\n\n" + thinking_delta

                    yield _sse_event(
                        "response.reasoning_summary_text.delta",
                        {
                            "output_index": 0,
                            "summary_index": 0,
                            "delta": thinking_delta,
                        },
                    )
                else:
                    # "pass" / "strip_tags" — include as regular text content
                    full_content += event.thinking_content

                    if not text_started:
                        text_started = True
                        yield _sse_event(
                            "response.output_item.added",
                            {
                                "output_index": 0,
                                "item": {
                                    "id": output_item_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "status": "in_progress",
                                    "content": [],
                                },
                            },
                        )
                        yield _sse_event(
                            "response.content_part.added",
                            {
                                "output_index": 0,
                                "content_index": 0,
                                "part": {"type": "output_text", "text": ""},
                            },
                        )

                    yield _sse_event(
                        "response.output_text.delta",
                        {
                            "output_index": 0,
                            "content_index": 0,
                            "delta": event.thinking_content,
                        },
                    )

            elif event.type == "tool_use" and event.tool_use:
                tool_calls_from_stream.append(event.tool_use)

            elif event.type == "usage" and event.usage:
                metering_data = event.usage

            elif event.type == "context_usage" and event.context_usage_percentage:
                context_usage_percentage = event.context_usage_percentage

        # --- Stream finished, emit closing events ---

        # Close reasoning if not already closed (e.g. no text content followed)
        if reasoning_started and not reasoning_closed:
            reasoning_closed = True
            full_thinking_with_header = "**Thinking**\n\n" + full_thinking_content
            yield _sse_event(
                "response.reasoning_summary_text.done",
                {
                    "output_index": 0,
                    "summary_index": 0,
                    "text": full_thinking_with_header,
                },
            )
            yield _sse_event(
                "response.reasoning_summary_part.done",
                {
                    "output_index": 0,
                    "summary_index": 0,
                    "part": {
                        "type": "summary_text",
                        "text": full_thinking_with_header,
                    },
                },
            )
            yield _sse_event(
                "response.output_item.done",
                {
                    "output_index": 0,
                    "item": {
                        "id": reasoning_item_id,
                        "type": "reasoning",
                        "summary": [
                            {"type": "summary_text", "text": full_thinking_with_header},
                        ],
                    },
                },
            )

        # Close text content if started
        text_output_index = 1 if reasoning_started else 0
        if text_started:
            yield _sse_event(
                "response.output_text.done",
                {
                    "output_index": text_output_index,
                    "content_index": 0,
                    "text": full_content,
                },
            )
            yield _sse_event(
                "response.content_part.done",
                {
                    "output_index": text_output_index,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": full_content,
                    },
                },
            )
            yield _sse_event(
                "response.output_item.done",
                {
                    "output_index": text_output_index,
                    "item": {
                        "id": output_item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {
                                "type": "output_text",
                                "text": full_content,
                            }
                        ],
                    },
                },
            )

        # Filter out tool calls with empty names — Kiro API sometimes streams
        # tool_use events where the name is empty. Forwarding these causes
        # "Improperly formed request" errors on the next turn.
        if tool_calls_from_stream:
            valid_tool_calls = [
                tc for tc in tool_calls_from_stream
                if (tc.get("function") or {}).get("name") or tc.get("name")
            ]
            dropped = len(tool_calls_from_stream) - len(valid_tool_calls)
            if dropped > 0:
                logger.warning(
                    f"Filtered {dropped} tool_call(s) with empty names from Responses API stream"
                )
            tool_calls_from_stream = valid_tool_calls

        # Track completion signals for truncation detection
        received_usage = metering_data is not None
        received_context_usage = context_usage_percentage is not None
        stream_completed_normally = received_usage or received_context_usage

        # Check bracket-style tool calls in full content
        bracket_tool_calls = parse_bracket_tool_calls(full_content)
        all_tool_calls = tool_calls_from_stream + bracket_tool_calls
        all_tool_calls = deduplicate_tool_calls(all_tool_calls)

        # Detect content truncation (missing completion signals)
        content_was_truncated = (
            not stream_completed_normally
            and len(full_content) > 0
            and not all_tool_calls  # Don't confuse with tool call truncation
        )

        if content_was_truncated:
            logger.error(
                f"Content truncated by Kiro API: stream ended without completion signals, "
                f"length={len(full_content)} chars. "
                f"{'Model will be notified automatically about truncation.' if TRUNCATION_RECOVERY else 'Set TRUNCATION_RECOVERY=true in .env to auto-notify model about truncation.'}"
            )

        # Save truncation info for recovery (tracked by stable identifiers)
        from kiro.truncation_recovery import should_inject_recovery
        from kiro.truncation_state import save_tool_truncation, save_content_truncation

        if should_inject_recovery():
            truncated_count = 0
            for tc in all_tool_calls:
                if tc.get("_truncation_detected"):
                    save_tool_truncation(
                        tool_call_id=tc["id"],
                        tool_name=tc["function"]["name"],
                        truncation_info=tc["_truncation_info"],
                    )
                    truncated_count += 1

            if content_was_truncated:
                save_content_truncation(full_content)

            if truncated_count > 0 or content_was_truncated:
                logger.info(
                    f"Truncation detected: {truncated_count} tool(s), "
                    f"content={content_was_truncated}. Will be handled when client sends next request."
                )

        # Emit tool call items if any (use all_tool_calls which includes bracket-parsed ones)
        for idx, tc in enumerate(all_tool_calls):
            tc_item_id = _generate_item_id()
            tc_call_id = tc.get("id", f"call_{uuid.uuid4().hex[:8]}")
            tc_func = tc.get("function", {})
            tc_name = tc_func.get("name", "") if tc_func else tc.get("name", "")
            tc_args = (
                tc_func.get("arguments", "{}") if tc_func else tc.get("arguments", "{}")
            )
            if isinstance(tc_args, dict):
                tc_args = json.dumps(tc_args)

            output_index = 1 + idx if text_started else idx
            yield _sse_event(
                "response.output_item.added",
                {
                    "output_index": output_index,
                    "item": {
                        "id": tc_item_id,
                        "type": "function_call",
                        "call_id": tc_call_id,
                        "name": tc_name,
                        "arguments": tc_args,
                        "status": "completed",
                    },
                },
            )
            yield _sse_event(
                "response.output_item.done",
                {
                    "output_index": output_index,
                    "item": {
                        "id": tc_item_id,
                        "type": "function_call",
                        "call_id": tc_call_id,
                        "name": tc_name,
                        "arguments": tc_args,
                        "status": "completed",
                    },
                },
            )

        # Calculate usage
        completion_tokens = count_tokens(full_content + full_thinking_content)
        prompt_tokens = 0
        total_tokens = completion_tokens
        prompt_source = "unknown"
        total_source = "unknown"

        if metering_data:
            metering_tokens = metering_data.get("outputTokenCount", 0)
            if metering_tokens > 0:
                completion_tokens = metering_tokens

        if context_usage_percentage and context_usage_percentage > 0:
            prompt_tokens, total_tokens, prompt_source, total_source = (
                calculate_tokens_from_context_usage(
                    context_usage_percentage, completion_tokens, model_cache, model
                )
            )
        else:
            total_tokens = prompt_tokens + completion_tokens

        # Fallback: estimate prompt_tokens from tools if Kiro didn't provide context_usage
        if prompt_source == "unknown" and request_tools:
            try:
                tool_dicts = [
                    t.model_dump() if hasattr(t, "model_dump") else t
                    for t in request_tools
                ]
                prompt_tokens += count_tools_tokens(
                    tool_dicts, apply_claude_correction=False
                )
                total_tokens = prompt_tokens + completion_tokens
                prompt_source = "tiktoken"
                total_source = "tiktoken"
            except Exception as e:
                logger.debug(f"Failed to count tool tokens: {e}")

        # Build output list for completed event
        completed_output = []
        if text_started:
            completed_output.append({
                "id": output_item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": full_content}],
            })
        for idx, tc in enumerate(all_tool_calls):
            tc_func = tc.get("function") or {}
            completed_output.append({
                "id": _generate_item_id(),
                "type": "function_call",
                "call_id": tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                "name": tc_func.get("name", "") if tc_func else tc.get("name", ""),
                "arguments": tc_func.get("arguments", "{}") if tc_func else tc.get("arguments", "{}"),
                "status": "completed",
            })

        # Emit response.completed
        yield _sse_event(
            "response.completed",
            {
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": int(time.time()),
                    "model": model,
                    "status": "completed",
                    "output": completed_output,
                    "usage": {
                        "input_tokens": prompt_tokens,
                        "output_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "input_tokens_details": {
                            "cached_tokens": 0,
                        },
                        "output_tokens_details": {
                            "reasoning_tokens": count_tokens(full_thinking_content)
                            if full_thinking_content
                            else 0,
                        },
                        **({"credits_used": metering_data} if metering_data else {}),
                    },
                },
            },
        )

    except FirstTokenTimeoutError:
        raise
    except GeneratorExit:
        # Client disconnected — normal, not an error
        logger.debug("Client disconnected (GeneratorExit) during Responses API streaming")
        streaming_error_occurred = True
        raise
    except Exception as e:
        streaming_error_occurred = True
        error_msg = str(e) if str(e) else "(empty message)"
        logger.error("Error during Responses API streaming: {}", error_msg)

        # Emit response.failed so client can close cleanly
        yield _sse_event(
            "response.failed",
            {
                "response": {
                    "id": response_id,
                    "object": "response",
                    "created_at": int(time.time()),
                    "model": model,
                    "status": "failed",
                    "error": {
                        "type": "server_error",
                        "message": error_msg,
                    },
                },
            },
        )
        raise
    finally:
        # Always close the upstream response to prevent connection leaks
        try:
            await response.aclose()
        except Exception as close_error:
            logger.debug(f"Error closing response: {close_error}")

        if streaming_error_occurred:
            logger.debug("Responses API streaming completed with error")
        else:
            logger.debug("Responses API streaming completed successfully")


# NOTE: stream_kiro_to_responses pass-through wrapper removed (was dead code).
# Routes should call stream_with_first_token_retry (for streaming)
# or stream_kiro_to_responses_internal (for non-streaming collection) directly.


async def stream_with_first_token_retry(
    make_request: Callable[[], Awaitable[httpx.Response]],
    client: httpx.AsyncClient,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    max_retries: int = FIRST_TOKEN_MAX_RETRIES,
    first_token_timeout: float = FIRST_TOKEN_TIMEOUT,
    request_input: Optional[list] = None,
    request_tools: Optional[list] = None,
    conversation_id: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Streaming with automatic retry on first token timeout for Responses API.

    Args:
        make_request: Function to create new HTTP request
        client: HTTP client
        model: Model name
        model_cache: Model cache
        auth_manager: Authentication manager
        max_retries: Maximum number of attempts
        first_token_timeout: First token wait timeout (seconds)
        request_input: Original request input items
        request_tools: Original request tools

    Yields:
        Strings in Responses API SSE format

    Raises:
        HTTPException: After exhausting all attempts
    """

    def create_http_error(status_code: int, error_text: str) -> HTTPException:
        return HTTPException(
            status_code=status_code,
            detail=f"Upstream API error: {error_text}",
        )

    def create_timeout_error(retries: int, timeout: float) -> HTTPException:
        return HTTPException(
            status_code=504,
            detail=f"Model did not respond within {timeout}s after {retries} attempts. Please try again.",
        )

    async def stream_processor(resp: httpx.Response) -> AsyncGenerator[str, None]:
        async for chunk in stream_kiro_to_responses_internal(
            client,
            resp,
            model,
            model_cache,
            auth_manager,
            first_token_timeout=first_token_timeout,
            request_input=request_input,
            request_tools=request_tools,
            conversation_id=conversation_id,
        ):
            yield chunk

    async for chunk in stream_with_first_token_retry_core(
        make_request=make_request,
        stream_processor=stream_processor,
        max_retries=max_retries,
        first_token_timeout=first_token_timeout,
        on_http_error=create_http_error,
        on_all_retries_failed=create_timeout_error,
    ):
        yield chunk


async def collect_responses_response(
    client: httpx.AsyncClient,
    response: httpx.Response,
    model: str,
    model_cache: "ModelInfoCache",
    auth_manager: "KiroAuthManager",
    request_input: Optional[list] = None,
    request_tools: Optional[list] = None,
) -> dict:
    """
    Collect full response from streaming for non-streaming Responses API mode.

    Args:
        client: HTTP client
        response: HTTP response with stream
        model: Model name
        model_cache: Model cache
        auth_manager: Authentication manager
        request_input: Original request input items
        request_tools: Original request tools

    Returns:
        Dictionary with full response in Responses API format
    """
    response_id = _generate_response_id()
    full_content = ""
    full_thinking = ""
    tool_calls = []
    final_usage = None

    async for chunk_str in stream_kiro_to_responses_internal(
        client,
        response,
        model,
        model_cache,
        auth_manager,
        request_input=request_input,
        request_tools=request_tools,
    ):
        # Parse SSE events
        for line in chunk_str.split("\n"):
            if not line.startswith("data: "):
                continue
            data_str = line[6:].strip()
            if not data_str:
                continue
            try:
                event_data = json.loads(data_str)
                event_type = event_data.get("type", "")

                if event_type == "response.output_text.delta":
                    full_content += event_data.get("delta", "")
                elif event_type == "response.reasoning_summary_text.delta":
                    full_thinking += event_data.get("delta", "")
                elif event_type == "response.completed":
                    resp = event_data.get("response", {})
                    final_usage = resp.get("usage")
                elif event_type == "response.output_item.done":
                    item = event_data.get("item", {})
                    if item.get("type") == "function_call":
                        tool_calls.append(item)
            except json.JSONDecodeError:
                continue

    # Build output items
    output = []
    if full_content:
        output_item_id = _generate_item_id()
        output.append(
            {
                "id": output_item_id,
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": full_content}],
            }
        )
    for tc in tool_calls:
        output.append(tc)

    return {
        "id": response_id,
        "object": "response",
        "created_at": int(time.time()),
        "model": model,
        "status": "completed",
        "output": output,
        "usage": final_usage
        or {
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
        },
    }
