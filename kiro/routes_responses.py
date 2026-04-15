# -*- coding: utf-8 -*-

"""
FastAPI routes for Responses API (/v1/responses).

Provides the OpenAI Responses API endpoint used by Codex CLI.
Follows the same patterns as routes_openai.py but with the
Responses API wire format.
"""

import json
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Security
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import APIKeyHeader
from loguru import logger

from kiro.config import PROXY_API_KEY, PROFILE_ARN
from kiro.models_responses import ResponsesApiRequest
from kiro.auth import KiroAuthManager
from kiro.cache import ModelInfoCache
from kiro.model_resolver import ModelResolver
from kiro.converters_responses import build_kiro_payload_responses
from kiro.streaming_responses import (
    stream_with_first_token_retry,
    collect_responses_response,
)
from kiro.http_client import KiroHttpClient
from kiro.utils import generate_conversation_id

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
        logger.warning("Access attempt with invalid API key (Responses API).")
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return True


# --- Router ---
router = APIRouter()


@router.post("/v1/responses", dependencies=[Depends(verify_api_key)])
async def responses_endpoint(request: Request, request_data: ResponsesApiRequest):
    """
    Responses API endpoint - compatible with OpenAI Responses API.

    Accepts requests in Responses API format (used by Codex CLI)
    and translates them to Kiro API. Supports streaming and non-streaming.

    Args:
        request: FastAPI Request for accessing app.state
        request_data: Request in ResponsesApiRequest format

    Returns:
        StreamingResponse for streaming mode (SSE with Responses API events)
        JSONResponse for non-streaming mode

    Raises:
        HTTPException: On validation or API errors
    """
    logger.info(
        f"Request to /v1/responses (model={request_data.model}, stream={request_data.stream})"
    )

    auth_manager: KiroAuthManager = request.app.state.auth_manager
    model_cache: ModelInfoCache = request.app.state.model_cache

    # Generate conversation ID from input for stability
    input_for_id = request_data.input if isinstance(request_data.input, list) else []
    conversation_id = generate_conversation_id(
        [{"role": "user", "content": str(item)} for item in input_for_id[:3]]
        if input_for_id else None
    )

    # Check for truncation recovery opportunities (Responses API input items)
    from kiro.truncation_state import get_tool_truncation, get_content_truncation
    from kiro.truncation_recovery import generate_truncation_tool_result, generate_truncation_user_message

    if isinstance(request_data.input, list):
        modified_input = []
        tool_results_modified = 0
        content_notices_added = 0

        for item in request_data.input:
            if not isinstance(item, dict):
                modified_input.append(item)
                continue

            item_type = item.get("type", "")

            # Check function_call_output items for truncated tool calls
            if item_type == "function_call_output":
                call_id = item.get("call_id", item.get("id", ""))
                if call_id:
                    truncation_info = get_tool_truncation(call_id)
                    if truncation_info:
                        synthetic = generate_truncation_tool_result(
                            tool_name=truncation_info.tool_name,
                            tool_use_id=call_id,
                            truncation_info=truncation_info.truncation_info,
                        )
                        original_output = item.get("output", "")
                        modified_item = {**item, "output": f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{original_output}"}
                        modified_input.append(modified_item)
                        tool_results_modified += 1
                        logger.debug(f"Modified function_call_output for {call_id} to include truncation notice")
                        continue

            # Check assistant messages for truncated content
            if item_type == "message" and item.get("role") == "assistant":
                msg_content = item.get("content", "")
                if isinstance(msg_content, str) and msg_content:
                    truncation_info = get_content_truncation(msg_content)
                    if truncation_info:
                        modified_input.append(item)
                        # Inject synthetic user message about truncation
                        modified_input.append({
                            "type": "message",
                            "role": "user",
                            "content": generate_truncation_user_message(),
                        })
                        content_notices_added += 1
                        logger.debug(f"Added truncation notice after assistant message (hash: {truncation_info.message_hash})")
                        continue

            modified_input.append(item)

        if tool_results_modified > 0 or content_notices_added > 0:
            request_data.input = modified_input
            logger.info(
                f"Truncation recovery: modified {tool_results_modified} tool_result(s), "
                f"added {content_notices_added} content notice(s)"
            )

    # Build Kiro payload from Responses API request
    try:
        kiro_payload = build_kiro_payload_responses(
            request_data, conversation_id, PROFILE_ARN
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to build Kiro payload: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

    # Log payload for debugging
    if debug_logger:
        debug_logger.log_kiro_request_body(json.dumps(kiro_payload).encode("utf-8"))

    # Create per-request HTTP client for streaming
    http_client = KiroHttpClient(auth_manager)

    try:
        # Get Kiro API URL
        kiro_url = f"{auth_manager.api_host}/generateAssistantResponse"

        if request_data.stream:
            # Streaming mode — use first-token retry for resilience
            async def make_request():
                return await http_client.request_with_retry(
                    method="POST",
                    url=kiro_url,
                    json_data=kiro_payload,
                    stream=True,
                )

            async def stream_wrapper():
                streaming_error = None
                client_disconnected = False
                try:
                    async for chunk in stream_with_first_token_retry(
                        make_request=make_request,
                        client=http_client.client,
                        model=request_data.model,
                        model_cache=model_cache,
                        auth_manager=auth_manager,
                        request_input=request_data.input if isinstance(request_data.input, list) else None,
                        request_tools=request_data.tools,
                        conversation_id=conversation_id,
                    ):
                        yield chunk
                except GeneratorExit:
                    client_disconnected = True
                    logger.debug("Client disconnected during Responses API streaming")
                except Exception as e:
                    streaming_error = e
                    # Emit a terminal event so the client doesn't hang.
                    # Do NOT re-raise — the response has already started (HTTP 200 sent),
                    # so raising would cause "Caught handled exception, but response already started".
                    try:
                        error_data = {
                            "type": "response.failed",
                            "response": {
                                "status": "failed",
                                "error": {
                                    "type": "server_error",
                                    "message": str(e)[:200],
                                },
                            },
                        }
                        yield f"event: response.failed\ndata: {json.dumps(error_data, ensure_ascii=False)}\n\n"
                    except Exception:
                        pass  # Client already disconnected
                finally:
                    await http_client.close()
                    if streaming_error:
                        error_type = type(streaming_error).__name__
                        error_msg = str(streaming_error) if str(streaming_error) else "(empty)"
                        logger.error(
                            f"HTTP 500 - POST /v1/responses (streaming) - [{error_type}] {error_msg[:100]}"
                        )
                    elif client_disconnected:
                        logger.info("HTTP 200 - POST /v1/responses (streaming) - client disconnected")
                    else:
                        logger.info("HTTP 200 - POST /v1/responses (streaming) - completed")
                    if debug_logger:
                        if streaming_error:
                            debug_logger.flush_on_error(500, str(streaming_error))
                        else:
                            debug_logger.discard_buffers()

            # 1.6: Add SSE headers to prevent proxy/CDN buffering
            return StreamingResponse(
                stream_wrapper(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                },
            )

        else:
            # Non-streaming mode — make request and collect full response
            response = await http_client.request_with_retry(
                method="POST",
                url=kiro_url,
                json_data=kiro_payload,
                stream=True,
            )

            # Handle non-200 responses
            if response.status_code != 200:
                error_text = ""
                try:
                    error_content = await response.aread()
                    error_text = error_content.decode("utf-8", errors="replace")
                except Exception:
                    error_text = "Unknown error"

                await http_client.close()

                error_message = error_text
                try:
                    error_json = json.loads(error_text)
                    from kiro.kiro_errors import enhance_kiro_error
                    error_info = enhance_kiro_error(error_json)
                    error_message = error_info.user_message
                    logger.debug(
                        f"Original Kiro error: {error_info.original_message} (reason: {error_info.reason})"
                    )
                except (json.JSONDecodeError, KeyError):
                    pass

                logger.warning(
                    f"HTTP {response.status_code} - POST /v1/responses - {error_message[:100]}"
                )

                if debug_logger:
                    debug_logger.flush_on_error(response.status_code, error_message)

                return JSONResponse(
                    status_code=response.status_code,
                    content={
                        "error": {
                            "message": error_message,
                            "type": "kiro_api_error",
                            "code": str(response.status_code),
                        }
                    },
                )

            responses_response = await collect_responses_response(
                http_client.client,
                response,
                request_data.model,
                model_cache,
                auth_manager,
                request_input=request_data.input if isinstance(request_data.input, list) else None,
                request_tools=request_data.tools,
            )

            await http_client.close()
            logger.info("HTTP 200 - POST /v1/responses (non-streaming) - completed")

            if debug_logger:
                debug_logger.discard_buffers()

            return JSONResponse(content=responses_response)

    except HTTPException as e:
        await http_client.close()
        logger.error(f"HTTP {e.status_code} - POST /v1/responses - {e.detail}")
        if debug_logger:
            debug_logger.flush_on_error(e.status_code, str(e.detail))
        raise
    except Exception as e:
        await http_client.close()
        logger.error(f"Internal error in /v1/responses: {e}", exc_info=True)
        logger.error(f"HTTP 500 - POST /v1/responses - {str(e)[:100]}")
        if debug_logger:
            debug_logger.flush_on_error(500, str(e))
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
