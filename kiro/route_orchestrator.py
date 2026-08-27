"""
Shared request orchestration for API routes.

All client-facing routes (OpenAI Chat Completions, Anthropic Messages, OpenAI
Responses) follow the same flow once the client payload has been validated:

1. Pick an account (with failover, or a single account in legacy mode)
2. Build the Kiro payload via a format-specific converter
3. POST to Kiro's /generateAssistantResponse
4. Stream the response back, or collect it into a single JSON body
5. Classify upstream errors as FATAL (return to client) or RECOVERABLE (failover)

Only the *shapes* differ between formats: the error envelope, the converter, the
streaming emitter. Those are injected through :class:`RouteFormat`; everything
else lives here.

Both entry points (:func:`run_with_account_failover` and
:func:`run_legacy_single_account`) return a FastAPI ``Response``. Some formats
signal errors by raising ``HTTPException`` instead of returning a body - that is
the format's choice, made inside its ``make_error`` hook.
"""

import json
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Awaitable, Callable, Dict, Optional, Tuple

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from loguru import logger

from kiro.config import PROFILE_ARN
from kiro.http_client import KiroHttpClient
from kiro.utils import generate_conversation_id

try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


# Error kinds passed to RouteFormat.make_error. Each format decides whether to
# return a JSONResponse or raise HTTPException for a given kind, and which
# envelope to use.
ERROR_NO_ACCOUNTS = "no_accounts"          # No account available for the model
ERROR_ALL_FAILED = "all_failed"            # Failover exhausted every account
ERROR_NO_AUTH = "no_auth"                  # Legacy mode: no initialized account
ERROR_INVALID_REQUEST = "invalid_request"  # Converter raised ValueError
ERROR_UPSTREAM = "upstream"                # Kiro returned a non-200 status
ERROR_INTERNAL = "internal"                # Unhandled exception


@dataclass
class RouteFormat:
    """
    Per-API-format hooks used by the shared orchestrator.

    Attributes:
        endpoint_label: Endpoint path used in access log lines, e.g. "/v1/messages".
        build_payload: (request_data, conversation_id, profile_arn) -> Kiro payload.
            May raise ValueError for invalid requests.
        prepare_tokenizer_data: (request_data) -> kwargs forwarded to the stream
            runner and collector for fallback token counting.
        run_stream: Returns an async generator of SSE strings. Called with
            keyword args: make_request, http_client, model, model_cache,
            auth_manager, initial_response, tokenizer_data.
        make_error: (kind, status_code, message) -> Response. May raise instead
            of returning. See the ERROR_* constants for the kinds used.
        make_stream_error_frame: (exception) -> SSE string sent to the client
            when streaming fails after headers were already flushed. None to
            send nothing.
        reraise_stream_error: Whether to re-raise a streaming exception after
            emitting the error frame. Controls whether the client sees a broken
            connection or a clean end of stream.
        collect_response: Non-streaming collector, called with keyword args:
            http_client, response, model, model_cache, auth_manager,
            tokenizer_data. None for streaming-only formats.
        streaming_headers: Extra headers for the StreamingResponse.
    """
    endpoint_label: str
    build_payload: Callable[[Any, str, str], Dict[str, Any]]
    prepare_tokenizer_data: Callable[[Any], Dict[str, Any]]
    run_stream: Callable[..., AsyncGenerator[str, None]]
    make_error: Callable[[str, int, str], Response]
    make_stream_error_frame: Optional[Callable[[Exception], str]] = None
    reraise_stream_error: bool = True
    collect_response: Optional[Callable[..., Awaitable[Dict[str, Any]]]] = None
    streaming_headers: Optional[Dict[str, str]] = None


def _log_kiro_payload(kiro_payload: Dict[str, Any]) -> None:
    """Write the outgoing Kiro payload to the debug log, ignoring failures."""
    try:
        kiro_request_body = json.dumps(kiro_payload, ensure_ascii=False, indent=2).encode('utf-8')
        if debug_logger:
            debug_logger.log_kiro_request_body(kiro_request_body)
    except Exception as e:
        logger.warning(f"Failed to log Kiro request: {e}")


def _make_http_client(request: Request, auth_manager: Any, streaming: bool) -> KiroHttpClient:
    """
    Create a Kiro HTTP client.

    Streaming uses a per-request client so a network interface change (VPN
    disconnect/reconnect) cannot leave orphaned CLOSE_WAIT sockets (issue #54).
    Non-streaming reuses the shared pooled client.
    """
    if streaming:
        return KiroHttpClient(auth_manager, shared_client=None)
    return KiroHttpClient(auth_manager, shared_client=request.app.state.http_client)


async def _read_upstream_error(response: Any) -> str:
    """Read an error body from a Kiro response, tolerating read failures."""
    try:
        error_content = await response.aread()
    except Exception:
        error_content = b"Unknown error"
    return error_content.decode('utf-8', errors='replace')


def _enhance_error(error_text: str) -> Tuple[str, Optional[str]]:
    """
    Turn a raw Kiro error body into a user-facing message and a reason code.

    Args:
        error_text: Raw response body from Kiro.

    Returns:
        (user_message, reason). reason is None when the body is not Kiro JSON.
    """
    try:
        error_json = json.loads(error_text)
        from kiro.kiro_errors import enhance_kiro_error
        error_info = enhance_kiro_error(error_json)
        logger.debug(
            f"Original Kiro error: {error_info.original_message} (reason: {error_info.reason})"
        )
        return error_info.user_message, error_info.reason
    except (json.JSONDecodeError, KeyError):
        return error_text, None


def _build_stream_wrapper(
    fmt: RouteFormat,
    http_client: KiroHttpClient,
    url: str,
    kiro_payload: Dict[str, Any],
    model: str,
    model_cache: Any,
    auth_manager: Any,
    initial_response: Any,
    tokenizer_data: Dict[str, Any],
) -> Callable[[], AsyncGenerator[str, None]]:
    """
    Build the streaming generator, including client-disconnect and error handling.

    The generator owns the HTTP client's lifetime: it is closed in the finally
    block, after the last chunk has been yielded.
    """
    async def stream_wrapper() -> AsyncGenerator[str, None]:
        streaming_error: Optional[Exception] = None
        client_disconnected = False
        try:
            async def make_retry_request():
                return await http_client.request_with_retry(
                    "POST", url, kiro_payload, stream=True
                )

            async for chunk in fmt.run_stream(
                make_request=make_retry_request,
                http_client=http_client,
                model=model,
                model_cache=model_cache,
                auth_manager=auth_manager,
                initial_response=initial_response,
                tokenizer_data=tokenizer_data,
            ):
                yield chunk
        except GeneratorExit:
            # Client went away mid-stream. Normal, not an error.
            client_disconnected = True
            logger.debug("Client disconnected during streaming (GeneratorExit in routes)")
        except Exception as e:
            streaming_error = e
            # Headers are already flushed, so the only way to tell the client
            # what happened is an in-band error frame.
            if fmt.make_stream_error_frame is not None:
                try:
                    yield fmt.make_stream_error_frame(e)
                except Exception:
                    pass  # Client already gone
            if fmt.reraise_stream_error:
                raise
        finally:
            await http_client.close()
            label = fmt.endpoint_label
            if streaming_error:
                error_type = type(streaming_error).__name__
                error_msg = str(streaming_error) if str(streaming_error) else "(empty message)"
                logger.error(
                    f"HTTP 500 - POST {label} (streaming) - [{error_type}] {error_msg[:100]}"
                )
            elif client_disconnected:
                logger.info(f"HTTP 200 - POST {label} (streaming) - client disconnected")
            else:
                logger.info(f"HTTP 200 - POST {label} (streaming) - completed")
            if debug_logger:
                if streaming_error:
                    debug_logger.flush_on_error(500, str(streaming_error))
                else:
                    debug_logger.discard_buffers()

    return stream_wrapper


class _ConversionError(Exception):
    """
    The format converter rejected the client request.

    Distinct from a bare ValueError so that a ValueError raised anywhere else in
    the request path still surfaces as an internal error rather than a 400.
    """


class _UpstreamError(Exception):
    """
    Kiro returned a non-200 status.

    Carries the details the failover loop needs to classify the failure and, if
    it turns out to be fatal, to build a client-facing error.
    """

    def __init__(self, status_code: int, message: str, reason: Optional[str]):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.reason = reason


async def _execute_kiro_request(
    request: Request,
    request_data: Any,
    fmt: RouteFormat,
    auth_manager: Any,
    model_cache: Any,
    account_label: Optional[str] = None,
) -> Response:
    """
    Run one Kiro request against an already-selected account.

    Args:
        request: FastAPI request, for app.state access.
        request_data: Validated client request model.
        fmt: Format-specific hooks.
        auth_manager: Auth manager of the selected account.
        model_cache: Model info cache of the selected account.
        account_label: Account id, for log lines. None in legacy mode.

    Returns:
        A StreamingResponse or JSONResponse on success.

    Raises:
        _UpstreamError: Kiro returned a non-200 status.
        ValueError: The converter rejected the request.
        HTTPException: Network-level failure from the HTTP client, or an
            application error raised further down.
    """
    conversation_id = generate_conversation_id()
    # profileArn is required by runtime.kiro.dev for all auth types
    profile_arn_for_payload = auth_manager.profile_arn or PROFILE_ARN or ""

    try:
        kiro_payload = fmt.build_payload(request_data, conversation_id, profile_arn_for_payload)
    except ValueError as e:
        raise _ConversionError(str(e)) from e
    _log_kiro_payload(kiro_payload)

    url = f"{auth_manager.api_host}/generateAssistantResponse"
    if account_label:
        logger.debug(f"Kiro API URL: {url} (account: {account_label})")
    else:
        logger.debug(f"Kiro API URL: {url}")

    streaming = bool(getattr(request_data, "stream", False))
    http_client = _make_http_client(request, auth_manager, streaming)
    tokenizer_data = fmt.prepare_tokenizer_data(request_data)

    try:
        # Wait for Kiro's response headers before committing to a response type,
        # so a failure can still be surfaced as a proper HTTP status code.
        response = await http_client.request_with_retry("POST", url, kiro_payload, stream=True)

        if response.status_code != 200:
            error_text = await _read_upstream_error(response)
            await http_client.close()
            message, reason = _enhance_error(error_text)
            raise _UpstreamError(response.status_code, message, reason)

        if streaming:
            stream_wrapper = _build_stream_wrapper(
                fmt, http_client, url, kiro_payload, request_data.model,
                model_cache, auth_manager, response, tokenizer_data,
            )
            return StreamingResponse(
                stream_wrapper(),
                media_type="text/event-stream",
                headers=fmt.streaming_headers,
            )

        if fmt.collect_response is None:
            raise RuntimeError(
                f"{fmt.endpoint_label} is streaming-only but received stream=False"
            )

        collected = await fmt.collect_response(
            http_client=http_client,
            response=response,
            model=request_data.model,
            model_cache=model_cache,
            auth_manager=auth_manager,
            tokenizer_data=tokenizer_data,
        )
        await http_client.close()
        logger.info(f"HTTP 200 - POST {fmt.endpoint_label} (non-streaming) - completed")
        if debug_logger:
            debug_logger.discard_buffers()
        return JSONResponse(content=collected)

    except _UpstreamError:
        # Already closed at the raise site; closing twice would be wrong.
        raise
    except Exception:
        # Failures close the client here; success paths either hand ownership to
        # the stream wrapper or close it above.
        await http_client.close()
        raise


async def run_with_account_failover(request: Request, request_data: Any, fmt: RouteFormat) -> Response:
    """
    Execute a Kiro request, rotating through accounts on recoverable failures.

    Tries each account at most once per call. A FATAL upstream error returns to
    the client immediately; a RECOVERABLE one moves to the next account. With a
    single configured account there is nothing to fail over to, so the loop
    stops early and that account's original error is surfaced verbatim.

    Args:
        request: FastAPI request, for app.state access.
        request_data: Validated client request model.
        fmt: Format-specific hooks.

    Returns:
        A FastAPI Response.
    """
    from kiro.account_errors import classify_error, ErrorType

    account_manager = request.app.state.account_manager
    all_accounts = list(account_manager._accounts.keys())
    MAX_ATTEMPTS = len(all_accounts) * 2  # Full circle with margin
    single_account = len(all_accounts) == 1

    last_error_message: Optional[str] = None
    last_error_status: Optional[int] = None
    tried_accounts = set()

    for _attempt in range(MAX_ATTEMPTS):
        account = await account_manager.get_next_account(
            request_data.model,
            exclude_accounts=tried_accounts
        )

        if account is None:
            if single_account:
                # Surface the account's own error, not a generic one.
                return fmt.make_error(
                    ERROR_NO_ACCOUNTS,
                    last_error_status or 503,
                    last_error_message or "Account unavailable",
                )
            detail = "No available accounts for this model."
            if last_error_message:
                detail += f" Error from last account: {last_error_message}"
            return fmt.make_error(ERROR_NO_ACCOUNTS, 503, detail)

        tried_accounts.add(account.id)

        try:
            response = await _execute_kiro_request(
                request, request_data, fmt,
                account.auth_manager, account.model_cache, account_label=account.id,
            )
            await account_manager.report_success(account.id, request_data.model)
            return response

        except _ConversionError as e:
            logger.error(f"Conversion error: {e}")
            return fmt.make_error(ERROR_INVALID_REQUEST, 400, str(e))

        except _UpstreamError as e:
            last_error_message = e.message
            last_error_status = e.status_code
            error_type = classify_error(e.status_code, e.reason)

            await account_manager.report_failure(
                account.id, request_data.model, error_type, e.status_code, e.reason
            )

            if error_type == ErrorType.FATAL:
                logger.warning(
                    f"HTTP {e.status_code} - POST {fmt.endpoint_label} - {e.message[:100]}"
                )
                if debug_logger:
                    debug_logger.flush_on_error(e.status_code, e.message)
                return fmt.make_error(ERROR_UPSTREAM, e.status_code, e.message)

            # RECOVERABLE: nothing to fail over to with one account.
            if single_account:
                break
            continue

        except HTTPException as e:
            # request_with_retry raises 502/504 for network-level failures only;
            # HTTP-level errors come back as response objects.
            if e.status_code in (502, 504):
                await account_manager.report_failure(
                    account.id, request_data.model, ErrorType.RECOVERABLE, e.status_code, None
                )
                last_error_message = str(e.detail)
                last_error_status = e.status_code
                if single_account:
                    break
                logger.warning(f"Network error on account {account.id}, trying next account")
                continue

            # Anything else is an application error - surface it as-is.
            logger.error(f"HTTP {e.status_code} - POST {fmt.endpoint_label} - {e.detail}")
            if debug_logger:
                debug_logger.flush_on_error(e.status_code, str(e.detail))
            raise

        except Exception as e:
            logger.error(f"Internal error: {e}", exc_info=True)
            logger.error(f"HTTP 500 - POST {fmt.endpoint_label} - {str(e)[:100]}")
            if debug_logger:
                debug_logger.flush_on_error(500, str(e))
            return fmt.make_error(ERROR_INTERNAL, 500, f"Internal Server Error: {str(e)}")

    # Every attempt exhausted.
    if single_account:
        # last_error_status/message are guaranteed set: the loop only reaches
        # here after at least one failure.
        return fmt.make_error(ERROR_ALL_FAILED, last_error_status, last_error_message)

    detail = "All accounts failed after full circle."
    if last_error_message:
        detail += f" Error from last account: {last_error_message}"
    return fmt.make_error(ERROR_ALL_FAILED, 503, detail)


async def run_legacy_single_account(request: Request, request_data: Any, fmt: RouteFormat) -> Response:
    """
    Execute a Kiro request against the first account, without failover.

    Used when the account system is disabled.

    Args:
        request: FastAPI request, for app.state access.
        request_data: Validated client request model.
        fmt: Format-specific hooks.

    Returns:
        A FastAPI Response.
    """
    account = request.app.state.account_manager.get_first_account()
    if not account.auth_manager:
        logger.error("No initialized accounts available (legacy mode)")
        return fmt.make_error(ERROR_NO_AUTH, 503, "No initialized accounts available")

    try:
        return await _execute_kiro_request(
            request, request_data, fmt, account.auth_manager, account.model_cache,
        )

    except _ConversionError as e:
        logger.error(f"Conversion error: {e}")
        return fmt.make_error(ERROR_INVALID_REQUEST, 400, str(e))

    except _UpstreamError as e:
        logger.warning(f"HTTP {e.status_code} - POST {fmt.endpoint_label} - {e.message[:100]}")
        if debug_logger:
            debug_logger.flush_on_error(e.status_code, e.message)
        return fmt.make_error(ERROR_UPSTREAM, e.status_code, e.message)

    except HTTPException as e:
        if e.status_code in (502, 504):
            logger.warning("Network error (legacy mode, no failover available)")
        logger.error(f"HTTP {e.status_code} - POST {fmt.endpoint_label} - {e.detail}")
        if debug_logger:
            debug_logger.flush_on_error(e.status_code, str(e.detail))
        raise

    except Exception as e:
        logger.error(f"Internal error: {e}", exc_info=True)
        logger.error(f"HTTP 500 - POST {fmt.endpoint_label} - {str(e)[:100]}")
        if debug_logger:
            debug_logger.flush_on_error(500, str(e))
        return fmt.make_error(ERROR_INTERNAL, 500, f"Internal Server Error: {str(e)}")
