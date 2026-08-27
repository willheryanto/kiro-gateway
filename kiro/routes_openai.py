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
FastAPI routes for Kiro Gateway.

Contains all API endpoints:
- / and /health: Health check
- /v1/models: Models list
- /v1/chat/completions: Chat completions
"""

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response, Security
from fastapi.responses import JSONResponse
from fastapi.security import APIKeyHeader
from loguru import logger

from kiro.config import (
    PROXY_API_KEY,
    APP_VERSION,
)
from kiro.models_openai import (
    OpenAIModel,
    ModelList,
    ChatCompletionRequest,
)
from kiro.auth import KiroAuthManager, AuthType
from kiro.cache import ModelInfoCache
from kiro.model_resolver import ModelResolver
from kiro.converters_openai import build_kiro_payload
from kiro.streaming_openai import stream_kiro_to_openai, collect_stream_response, stream_with_first_token_retry
from kiro.config import WEB_SEARCH_ENABLED
from kiro.mcp_tools import handle_native_web_search
from kiro.route_orchestrator import (
    ERROR_UPSTREAM,
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
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)


async def verify_api_key(auth_header: str = Security(api_key_header)) -> bool:
    """
    Verify API key in Authorization header.
    
    Expects format: "Bearer {PROXY_API_KEY}"
    
    Args:
        auth_header: Authorization header value
    
    Returns:
        True if key is valid
    
    Raises:
        HTTPException: 401 if key is invalid or missing
    """
    if not auth_header or auth_header != f"Bearer {PROXY_API_KEY}":
        logger.warning("Access attempt with invalid API key.")
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True


# --- Router ---
router = APIRouter()


@router.get("/")
async def root():
    """
    Health check endpoint.
    
    Returns:
        Status and application version
    """
    return {
        "status": "ok",
        "message": "Kiro Gateway is running",
        "version": APP_VERSION
    }


@router.get("/health")
async def health():
    """
    Detailed health check.
    
    Returns:
        Status, timestamp and version
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "version": APP_VERSION
    }

@router.get("/v1/models", response_model=ModelList, dependencies=[Depends(verify_api_key)])
async def get_models(request: Request):
    """
    Return list of available models.
    
    Models are loaded at startup (blocking) and cached.
    This endpoint returns the cached list.
    
    Args:
        request: FastAPI Request for accessing app.state
    
    Returns:
        ModelList with available models in consistent format (with dots)
    """
    logger.info("Request to /v1/models")
    
    # Get available models based on mode
    if request.app.state.account_system:
        # Account system: collect models from all initialized accounts
        available_model_ids = request.app.state.account_manager.get_all_available_models()
    else:
        # Legacy: use resolver from first account
        account = request.app.state.account_manager.get_first_account()
        available_model_ids = account.model_resolver.get_available_models()
    
    # Build OpenAI-compatible model list
    openai_models = [
        OpenAIModel(
            id=model_id,
            owned_by="anthropic",
            description="Claude model via Kiro API"
        )
        for model_id in available_model_ids
    ]
    
    return ModelList(data=openai_models)




# ==================================================================================================
# Route format definition (used by the shared orchestrator)
# ==================================================================================================

def _prepare_tokenizer_data(request_data: ChatCompletionRequest) -> dict:
    """
    Serialize request data for fallback token counting.

    Args:
        request_data: Validated OpenAI chat completion request.

    Returns:
        Keyword arguments for the streaming and collecting helpers.
    """
    return {
        "request_messages": [msg.model_dump() for msg in request_data.messages],
        "request_tools": (
            [tool.model_dump() for tool in request_data.tools] if request_data.tools else None
        ),
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
    """Stream a Kiro response as OpenAI chat completion chunks."""
    return stream_with_first_token_retry(
        make_request=make_request,
        client=http_client.client,
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
    """Collect a full Kiro response into an OpenAI chat completion body."""
    return await collect_stream_response(
        http_client.client,
        response,
        model,
        model_cache,
        auth_manager,
        **tokenizer_data,
    )


def _make_error(kind: str, status_code: int, message: str) -> Response:
    """
    Build an OpenAI-style error result.

    Upstream Kiro errors are returned as a JSON body so the client sees Kiro's
    original status code. Every other failure is raised as an HTTPException,
    matching this endpoint's long-standing behaviour.

    Args:
        kind: One of the ERROR_* constants from kiro.route_orchestrator.
        status_code: HTTP status code to report.
        message: User-facing error message.

    Returns:
        JSONResponse for upstream Kiro errors.

    Raises:
        HTTPException: For all other error kinds.
    """
    if kind == ERROR_UPSTREAM:
        return JSONResponse(
            status_code=status_code,
            content={
                "error": {
                    "message": message,
                    "type": "kiro_api_error",
                    "code": status_code
                }
            }
        )
    raise HTTPException(status_code=status_code, detail=message)


def _build_route_format() -> RouteFormat:
    """Build the RouteFormat describing the OpenAI Chat Completions endpoint."""
    return RouteFormat(
        endpoint_label="/v1/chat/completions",
        build_payload=build_kiro_payload,
        prepare_tokenizer_data=_prepare_tokenizer_data,
        run_stream=_run_stream,
        make_error=_make_error,
        make_stream_error_frame=lambda _e: "data: [DONE]\n\n",
        reraise_stream_error=True,
        collect_response=_collect_response,
    )

@router.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)])
async def chat_completions(request: Request, request_data: ChatCompletionRequest):
    """
    Chat completions endpoint - compatible with OpenAI API.
    
    Accepts requests in OpenAI format and translates them to Kiro API.
    Supports streaming and non-streaming modes.
    
    Args:
        request: FastAPI Request for accessing app.state
        request_data: Request in OpenAI ChatCompletionRequest format
    
    Returns:
        StreamingResponse for streaming mode
        JSONResponse for non-streaming mode
    
    Raises:
        HTTPException: On validation or API errors
    """
    logger.info(f"Request to /v1/chat/completions (model={request_data.model}, stream={request_data.stream})")
    
    # Note: prepare_new_request() and log_request_body() are now called by DebugLoggerMiddleware
    # This ensures debug logging works even for requests that fail Pydantic validation (422 errors)
    
    # Check for truncation recovery opportunities
    from kiro.truncation_state import get_tool_truncation, get_content_truncation
    from kiro.truncation_recovery import generate_truncation_tool_result, generate_truncation_user_message
    from kiro.models_openai import ChatMessage
    
    modified_messages = []
    tool_results_modified = 0
    content_notices_added = 0
    
    for msg in request_data.messages:
        # Check if this is a tool_result for a truncated tool call
        if msg.role == "tool" and msg.tool_call_id:
            truncation_info = get_tool_truncation(msg.tool_call_id)
            if truncation_info:
                # Modify tool_result content to include truncation notice
                synthetic = generate_truncation_tool_result(
                    tool_name=truncation_info.tool_name,
                    tool_use_id=msg.tool_call_id,
                    truncation_info=truncation_info.truncation_info
                )
                # Prepend truncation notice to original content
                modified_content = f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{msg.content}"
                
                # Create NEW ChatMessage object (Pydantic immutability)
                modified_msg = msg.model_copy(update={"content": modified_content})
                modified_messages.append(modified_msg)
                tool_results_modified += 1
                logger.debug(f"Modified tool_result for {msg.tool_call_id} to include truncation notice")
                continue  # Skip normal append since we already added modified version
        
        # Check if this is an assistant message with truncated content
        if msg.role == "assistant" and msg.content and isinstance(msg.content, str):
            truncation_info = get_content_truncation(msg.content)
            if truncation_info:
                # Add this message first
                modified_messages.append(msg)
                # Then add synthetic user message about truncation
                synthetic_user_msg = ChatMessage(
                    role="user",
                    content=generate_truncation_user_message()
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
        
        # Check if web_search already exists
        has_ws = any(
            getattr(tool, "type", None) == "function" and
            getattr(getattr(tool, "function", None), "name", None) == "web_search"
            for tool in request_data.tools
        )
        
        if not has_ws:
            from kiro.models_openai import Tool, ToolFunction
            web_search_tool = Tool(
                type="function",
                function=ToolFunction(
                    name="web_search",
                    description="Search the web for current information. Use when you need up-to-date data from the internet.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            }
                        },
                        "required": ["query"]
                    }
                )
            )
            request_data.tools.append(web_search_tool)
            logger.debug("Auto-injected web_search tool for MCP emulation (Path B)")
    
    # ==============================================================================
    # Account System: Account System Failover or Legacy Mode
    # ==============================================================================

    fmt = _build_route_format()

    if request.app.state.account_system:
        return await run_with_account_failover(request, request_data, fmt)

    return await run_legacy_single_account(request, request_data, fmt)
