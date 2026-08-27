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
OpenAI Responses API route for Kiro Gateway.

Exposes POST /v1/responses, the protocol used by newer OpenAI clients such as
the Codex CLI. Streaming only: those clients always request a stream, so there
is no single-JSON response path.

Authentication reuses the OpenAI bearer-token dependency, so the same
PROXY_API_KEY works for /v1/chat/completions and /v1/responses.
"""

from typing import Any, Dict, List

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import JSONResponse
from loguru import logger

from kiro.converters_responses import (
    collect_freeform_tool_names,
    collect_request_tools,
    convert_responses_input_to_unified,
    responses_to_kiro,
)
from kiro.models_responses import ResponsesRequest
from kiro.route_orchestrator import (
    ERROR_INVALID_REQUEST,
    RouteFormat,
    run_legacy_single_account,
    run_with_account_failover,
)
from kiro.routes_openai import verify_api_key
from kiro.streaming_responses import (
    make_response_failed_frame,
    stream_with_first_token_retry_responses,
)

router = APIRouter(tags=["OpenAI Responses API"])


# ==================================================================================================
# Route format definition (used by the shared orchestrator)
# ==================================================================================================

def _prepare_tokenizer_data(request_data: ResponsesRequest) -> dict:
    """
    Build the per-request keyword arguments for the streaming helper.

    Covers fallback token counting plus the freeform-tool names the emitter
    needs. Responses input items are converted to the role/content shape the
    tokenizer understands: its content-block handling expects Anthropic-style
    "text", "tool_use" and "tool_result" blocks, not the Responses "input_text"
    and peer-item layout.

    Args:
        request_data: Validated Responses request.

    Returns:
        Keyword arguments for the streaming helper.
    """
    _system_prompt, unified_messages = convert_responses_input_to_unified(
        request_data.input,
        request_data.instructions,
    )

    messages: List[Dict[str, Any]] = []
    for msg in unified_messages:
        blocks: List[Dict[str, Any]] = []
        if msg.content:
            blocks.append({"type": "text", "text": msg.content})
        for tool_call in msg.tool_calls or []:
            function = tool_call.get("function") or {}
            blocks.append({
                "type": "tool_use",
                "id": tool_call.get("id", ""),
                "name": function.get("name", ""),
                "input": function.get("arguments", ""),
            })
        for tool_result in msg.tool_results or []:
            blocks.append(tool_result)
        messages.append({"role": msg.role, "content": blocks})

    # Not request_data.tools: responses-lite clients leave that field empty and
    # carry definitions in an additional_tools input item instead, which would
    # undercount prompt tokens and lose the freeform-tool names.
    tools = collect_request_tools(request_data)

    return {
        "request_messages": messages,
        "request_tools": tools,
        "freeform_tool_names": set(collect_freeform_tool_names(tools)),
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
    """Stream a Kiro response as Responses API SSE events."""
    return stream_with_first_token_retry_responses(
        make_request=make_request,
        model=model,
        model_cache=model_cache,
        auth_manager=auth_manager,
        initial_response=initial_response,
        **tokenizer_data,
    )


def _make_error(kind: str, status_code: int, message: str) -> Response:
    """
    Build an OpenAI-style error response.

    Responses clients expect the same envelope as Chat Completions and read
    ``error.code`` to decide whether a failure is retryable.

    Args:
        kind: One of the ERROR_* constants from kiro.route_orchestrator.
        status_code: HTTP status code to report.
        message: User-facing error message.

    Returns:
        JSONResponse with an OpenAI error envelope.
    """
    error_type = "invalid_request_error" if kind == ERROR_INVALID_REQUEST else "kiro_api_error"
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": status_code,
            }
        }
    )


def _build_route_format() -> RouteFormat:
    """Build the RouteFormat describing the Responses endpoint."""
    return RouteFormat(
        endpoint_label="/v1/responses",
        build_payload=responses_to_kiro,
        prepare_tokenizer_data=_prepare_tokenizer_data,
        run_stream=_run_stream,
        make_error=_make_error,
        make_stream_error_frame=make_response_failed_frame,
        # End the stream cleanly after response.failed: an aborted connection
        # would surface to the client as "stream closed before response.completed"
        # instead of the actual error.
        reraise_stream_error=False,
        # Streaming-only endpoint - no non-streaming collector.
        collect_response=None,
    )


# ==================================================================================================
# Endpoint
# ==================================================================================================

@router.post("/v1/responses", dependencies=[Depends(verify_api_key)])
async def responses(request: Request, request_data: ResponsesRequest):
    """
    Responses API endpoint - compatible with the OpenAI Responses protocol.

    Accepts requests in Responses format and translates them to the Kiro API.
    Streaming only.

    Args:
        request: FastAPI Request, for app.state access.
        request_data: Request in Responses format.

    Returns:
        StreamingResponse carrying Responses SSE events.
    """
    logger.info(f"Request to /v1/responses (model={request_data.model}, stream={request_data.stream})")

    if not request_data.stream:
        logger.warning("Rejected /v1/responses request with stream=false")
        return _make_error(
            ERROR_INVALID_REQUEST,
            400,
            "This endpoint supports streaming only. Set \"stream\": true.",
        )

    fmt = _build_route_format()

    if request.app.state.account_system:
        return await run_with_account_failover(request, request_data, fmt)

    return await run_legacy_single_account(request, request_data, fmt)
