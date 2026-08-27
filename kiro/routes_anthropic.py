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
FastAPI routes for Anthropic Messages API.

Contains the /v1/messages endpoint compatible with Anthropic's Messages API.

Reference: https://docs.anthropic.com/en/api/messages
"""

import json
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, Response, Security, Header
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from loguru import logger

from kiro.config import PROXY_API_KEY
from kiro.models_anthropic import (
    AnthropicMessagesRequest,
    AnthropicCountTokensRequest,
    AnthropicMessagesResponse,
    AnthropicErrorResponse,
    AnthropicErrorDetail,
)
from kiro.auth import KiroAuthManager, AuthType
from kiro.cache import ModelInfoCache
from kiro.converters_anthropic import anthropic_to_kiro
from kiro.streaming_anthropic import (
    stream_kiro_to_anthropic,
    collect_anthropic_response,
    stream_with_first_token_retry_anthropic,
)
from kiro.tokenizer import estimate_request_tokens
from kiro.config import WEB_SEARCH_ENABLED
from kiro.mcp_tools import handle_native_web_search
from kiro.route_orchestrator import (
    ERROR_INVALID_REQUEST,
    RouteFormat,
    run_legacy_single_account,
    run_with_account_failover,
)

# Import debug_logger
try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


# --- Security scheme ---
# Anthropic uses x-api-key header instead of Authorization: Bearer
anthropic_api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
# Also support Authorization: Bearer for compatibility
auth_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_anthropic_api_key(
    x_api_key: Optional[str] = Security(anthropic_api_key_header),
    authorization: Optional[str] = Security(auth_header)
) -> bool:
    """
    Verify API key for Anthropic API.
    
    Supports two authentication methods:
    1. x-api-key header (Anthropic native)
    2. Authorization: Bearer header (for compatibility)
    
    Args:
        x_api_key: Value from x-api-key header
        authorization: Value from Authorization header
    
    Returns:
        True if key is valid
    
    Raises:
        HTTPException: 401 if key is invalid or missing
    """
    # Check x-api-key first (Anthropic native)
    if x_api_key and x_api_key == PROXY_API_KEY:
        return True
    
    # Fall back to Authorization: Bearer
    if authorization and authorization == f"Bearer {PROXY_API_KEY}":
        return True
    
    logger.warning("Access attempt with invalid API key (Anthropic endpoint)")
    raise HTTPException(
        status_code=401,
        detail={
            "type": "error",
            "error": {
                "type": "authentication_error",
                "message": "Invalid or missing API key. Use x-api-key header or Authorization: Bearer."
            }
        }
    )


# --- Router ---
router = APIRouter(tags=["Anthropic API"])




# ==================================================================================================
# Route format definition (used by the shared orchestrator)
# ==================================================================================================

def _prepare_tokenizer_data(request_data: AnthropicMessagesRequest) -> dict:
    """
    Serialize request data for fallback token counting.

    Args:
        request_data: Validated Anthropic messages request.

    Returns:
        Keyword arguments for the streaming and collecting helpers.
    """
    if isinstance(request_data.system, list):
        system_for_tokenizer = [
            b.model_dump() if hasattr(b, "model_dump") else b for b in request_data.system
        ]
    else:
        system_for_tokenizer = request_data.system

    return {
        "request_messages": [msg.model_dump() for msg in request_data.messages],
        "request_tools": (
            [tool.model_dump() for tool in request_data.tools] if request_data.tools else None
        ),
        "request_system": system_for_tokenizer,
    }


def _run_stream(
    make_request,
    http_client,
    model: str,
    model_cache,
    auth_manager,
    initial_response,
    tokenizer_data: dict,
):
    """Stream a Kiro response as Anthropic SSE events."""
    return stream_with_first_token_retry_anthropic(
        make_request=make_request,
        model=model,
        model_cache=model_cache,
        auth_manager=auth_manager,
        initial_response=initial_response,
        **tokenizer_data,
    )


async def _collect_response(
    http_client,
    response,
    model: str,
    model_cache,
    auth_manager,
    tokenizer_data: dict,
) -> dict:
    """Collect a full Kiro response into an Anthropic messages body."""
    return await collect_anthropic_response(
        response,
        model,
        model_cache,
        auth_manager,
        **tokenizer_data,
    )


def _make_error(kind: str, status_code: int, message: str) -> Response:
    """
    Build an Anthropic-style error response.

    This endpoint always returns an error body rather than raising, so clients
    receive a well-formed Anthropic error envelope.

    Args:
        kind: One of the ERROR_* constants from kiro.route_orchestrator.
        status_code: HTTP status code to report.
        message: User-facing error message.

    Returns:
        JSONResponse with an Anthropic error envelope.
    """
    error_type = "invalid_request_error" if kind == ERROR_INVALID_REQUEST else "api_error"
    return JSONResponse(
        status_code=status_code,
        content={
            "type": "error",
            "error": {
                "type": error_type,
                "message": message
            }
        }
    )


def _make_stream_error_frame(error: Exception) -> str:
    """Build an Anthropic SSE error event for a mid-stream failure."""
    payload = {"type": "error", "error": {"type": "api_error", "message": str(error)}}
    return f'event: error\ndata: {json.dumps(payload)}\n\n'


def _build_route_format() -> RouteFormat:
    """Build the RouteFormat describing the Anthropic Messages endpoint."""
    return RouteFormat(
        endpoint_label="/v1/messages",
        build_payload=anthropic_to_kiro,
        prepare_tokenizer_data=_prepare_tokenizer_data,
        run_stream=_run_stream,
        make_error=_make_error,
        make_stream_error_frame=_make_stream_error_frame,
        # Anthropic ends the stream cleanly after the error event rather than
        # propagating, so the client sees a complete SSE stream.
        reraise_stream_error=False,
        collect_response=_collect_response,
        streaming_headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )

@router.post("/v1/messages", dependencies=[Depends(verify_anthropic_api_key)])
async def messages(
    request: Request,
    request_data: AnthropicMessagesRequest,
    anthropic_version: Optional[str] = Header(None, alias="anthropic-version")
):
    """
    Anthropic Messages API endpoint.
    
    Compatible with Anthropic's /v1/messages endpoint.
    Accepts requests in Anthropic format and translates them to Kiro API.
    
    Required headers:
    - x-api-key: Your API key (or Authorization: Bearer)
    - anthropic-version: API version (optional, for compatibility)
    - Content-Type: application/json
    
    Args:
        request: FastAPI Request for accessing app.state
        request_data: Request in Anthropic MessagesRequest format
        anthropic_version: Anthropic API version header (optional)
    
    Returns:
        StreamingResponse for streaming mode (SSE)
        JSONResponse for non-streaming mode
    
    Raises:
        HTTPException: On validation or API errors
    """
    logger.info(f"Request to /v1/messages (model={request_data.model}, stream={request_data.stream})")
    
    if anthropic_version:
        logger.debug(f"Anthropic-Version header: {anthropic_version}")
    
    # Note: prepare_new_request() and log_request_body() are now called by DebugLoggerMiddleware
    # This ensures debug logging works even for requests that fail Pydantic validation (422 errors)
    
    # Check for truncation recovery opportunities
    from kiro.truncation_state import get_tool_truncation, get_content_truncation
    from kiro.truncation_recovery import generate_truncation_tool_result, generate_truncation_user_message
    from kiro.models_anthropic import AnthropicMessage
    
    modified_messages = []
    tool_results_modified = 0
    content_notices_added = 0
    
    for msg in request_data.messages:
        # Check if this is a user message with tool_result blocks
        if msg.role == "user" and msg.content and isinstance(msg.content, list):
            modified_content_blocks = []
            has_modifications = False
            
            for block in msg.content:
                # Handle both dict and Pydantic objects (ToolResultContentBlock)
                if isinstance(block, dict):
                    block_type = block.get("type")
                    tool_use_id = block.get("tool_use_id")
                    original_content = block.get("content", "")
                elif hasattr(block, "type"):
                    block_type = block.type
                    tool_use_id = getattr(block, "tool_use_id", None)
                    original_content = getattr(block, "content", "")
                else:
                    modified_content_blocks.append(block)
                    continue
                
                if block_type == "tool_result" and tool_use_id:
                    truncation_info = get_tool_truncation(tool_use_id)
                    if truncation_info:
                        # Modify tool_result content to include truncation notice
                        synthetic = generate_truncation_tool_result(
                            tool_name=truncation_info.tool_name,
                            tool_use_id=tool_use_id,
                            truncation_info=truncation_info.truncation_info
                        )
                        # Prepend truncation notice to original content
                        modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{original_content}"
                        
                        # Create modified block (handle both dict and Pydantic)
                        if isinstance(block, dict):
                            modified_block = block.copy()
                            modified_block["content"] = modified_content
                        else:
                            # Pydantic object - use model_copy
                            modified_block = block.model_copy(update={"content": modified_content})
                        
                        modified_content_blocks.append(modified_block)
                        tool_results_modified += 1
                        has_modifications = True
                        logger.debug(f"Modified tool_result for {tool_use_id} to include truncation notice")
                        continue
                
                modified_content_blocks.append(block)
            
            # Create NEW AnthropicMessage object if modifications were made (Pydantic immutability)
            if has_modifications:
                modified_msg = msg.model_copy(update={"content": modified_content_blocks})
                modified_messages.append(modified_msg)
                continue  # Skip normal append since we already added modified version
        
        # Check if this is an assistant message with truncated content
        if msg.role == "assistant" and msg.content:
            # Extract text content for hash check
            text_content = ""
            if isinstance(msg.content, str):
                text_content = msg.content
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text_content += block.get("text", "")
            
            if text_content:
                truncation_info = get_content_truncation(text_content)
                if truncation_info:
                    # Add this message first
                    modified_messages.append(msg)
                    # Then add synthetic user message about truncation
                    synthetic_user_msg = AnthropicMessage(
                        role="user",
                        content=[{"type": "text", "text": generate_truncation_user_message()}]
                    )
                    modified_messages.append(synthetic_user_msg)
                    content_notices_added += 1
                    logger.debug(f"Added truncation notice after assistant message (hash: {truncation_info.message_hash})")
                    continue  # Skip normal append since we already added it
        
        modified_messages.append(msg)
    
    if tool_results_modified > 0 or content_notices_added > 0:
        request_data.messages = modified_messages
        logger.info(f"Truncation recovery: modified {tool_results_modified} tool_result(s), added {content_notices_added} content notice(s)")
    
    # ==============================================================================
    # WebSearch Support - Path B: Auto-Injection (MCP Tool Emulation)
    # ==============================================================================
    
    # Auto-inject web_search tool if enabled (Path B - MCP emulation)
    if WEB_SEARCH_ENABLED:
        if request_data.tools is None:
            request_data.tools = []
        
        # Check if web_search already exists (by name)
        has_ws = any(
            getattr(tool, "name", "") == "web_search"
            for tool in request_data.tools
        )
        
        if not has_ws:
            from kiro.models_anthropic import AnthropicTool
            web_search_tool = AnthropicTool(
                name="web_search",
                description="Search the web for current information. Use when you need up-to-date data from the internet.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"}
                    },
                    "required": ["query"]
                }
            )
            request_data.tools.append(web_search_tool)
            logger.debug("Auto-injected web_search tool for MCP emulation (Path B)")
    
    # ==============================================================================
    # WebSearch Support - Path A: Native Anthropic (Early Return)
    # ==============================================================================
    
    # Check for native Anthropic server-side tool (Path A)
    # This works ALWAYS, regardless of WEB_SEARCH_ENABLED setting
    if request_data.tools:
        for tool in request_data.tools:
            tool_type = getattr(tool, "type", None)
            if tool_type and tool_type.startswith("web_search"):
                # Path A: Early return, direct MCP call
                # Get auth_manager from first available account (no failover needed for early return)
                account = request.app.state.account_manager.get_first_account()
                if not account.auth_manager:
                    logger.error("No initialized accounts available for native web_search")
                    return JSONResponse(
                        status_code=503,
                        content={
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": "No initialized accounts available"
                            }
                        }
                    )
                auth_manager = account.auth_manager
                
                logger.info("Detected native Anthropic web_search (Path A), routing to MCP API")
                return await handle_native_web_search(request, request_data, auth_manager, api_format="anthropic")
    
    # ==============================================================================
    # Account System: Account System Failover or Legacy Mode
    # ==============================================================================

    fmt = _build_route_format()

    if request.app.state.account_system:
        return await run_with_account_failover(request, request_data, fmt)

    return await run_legacy_single_account(request, request_data, fmt)


@router.post("/v1/messages/count_tokens", dependencies=[Depends(verify_anthropic_api_key)])
async def count_tokens_endpoint(
    request: Request,
    request_data: AnthropicCountTokensRequest,
):
    """
    Anthropic Count Tokens API endpoint.
    
    Returns estimated token count for the given request payload.
    Used by Claude Code to decide when to trigger conversation compaction.
    
    Uses the same fallback estimation as Anthropic streaming (message_start event),
    since Kiro API only provides accurate token counts after request completion.
    This endpoint is called BEFORE the actual request, so we cannot use Kiro's
    contextUsagePercentage (which is only available after generation completes).
    
    Args:
        request: FastAPI Request for accessing app.state
        request_data: Request in Anthropic MessagesRequest format
    
    Returns:
        JSONResponse with {"input_tokens": int}
    
    Raises:
        HTTPException: 401 if authentication fails (handled by dependency)
    """
    logger.info(f"Request to /v1/messages/count_tokens (model={request_data.model}, messages={len(request_data.messages)})")
    
    # Prepare data for tokenizer (same format as streaming message_start)
    messages_for_tokenizer = [msg.model_dump() for msg in request_data.messages]
    tools_for_tokenizer = [tool.model_dump() for tool in request_data.tools] if request_data.tools else None
    
    # Handle system prompt (can be string or list of content blocks)
    if isinstance(request_data.system, list):
        system_for_tokenizer = [b.model_dump() if hasattr(b, "model_dump") else b for b in request_data.system]
    else:
        system_for_tokenizer = request_data.system
    
    # Use the SAME estimation logic as Anthropic streaming message_start
    request_token_stats = estimate_request_tokens(
        messages=messages_for_tokenizer,
        tools=tools_for_tokenizer,
        system_prompt=system_for_tokenizer,
        apply_claude_correction=True  # CRITICAL: Enable correction for Claude models
    )
    
    input_tokens = request_token_stats["total_tokens"]
    
    logger.info(f"Token count estimate: {input_tokens} tokens")
    
    return JSONResponse(content={"input_tokens": input_tokens})
