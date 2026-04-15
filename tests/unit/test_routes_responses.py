# -*- coding: utf-8 -*-

"""
Unit tests for Responses API endpoints (routes_responses.py).

Tests the POST /v1/responses endpoint including:
- Authentication (valid/invalid/missing API keys)
- Request validation
- Streaming and non-streaming paths
- Error handling
- Truncation recovery injection
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock
from fastapi import HTTPException
from kiro.routes_responses import verify_api_key
from kiro.config import PROXY_API_KEY


# =============================================================================
# Authentication
# =============================================================================

class TestVerifyApiKey:
    """Tests for the verify_api_key function."""

    @pytest.mark.asyncio
    async def test_valid_bearer_token(self):
        result = await verify_api_key(f"Bearer {PROXY_API_KEY}")
        assert result is True

    @pytest.mark.asyncio
    async def test_invalid_key_raises_401(self):
        with pytest.raises(HTTPException) as exc:
            await verify_api_key("Bearer wrong_key")
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_key_raises_401(self):
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(None)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_empty_string_raises_401(self):
        with pytest.raises(HTTPException) as exc:
            await verify_api_key("")
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_no_bearer_prefix_raises_401(self):
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(PROXY_API_KEY)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_scheme_raises_401(self):
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(f"Basic {PROXY_API_KEY}")
        assert exc.value.status_code == 401


# =============================================================================
# Request Validation
# =============================================================================

class TestRequestValidation:
    """Tests for request model validation."""

    def test_empty_list_input_rejected(self):
        from pydantic import ValidationError
        from kiro.models_responses import ResponsesApiRequest
        with pytest.raises(ValidationError, match="input must not be an empty list"):
            ResponsesApiRequest(model="claude-sonnet-4", input=[])

    def test_none_input_accepted(self):
        from kiro.models_responses import ResponsesApiRequest
        req = ResponsesApiRequest(model="claude-sonnet-4", input=None)
        assert req.input is None

    def test_string_input_accepted(self):
        from kiro.models_responses import ResponsesApiRequest
        req = ResponsesApiRequest(model="claude-sonnet-4", input="Hello")
        assert req.input == "Hello"

    def test_valid_list_input_accepted(self):
        from kiro.models_responses import ResponsesApiRequest
        req = ResponsesApiRequest(
            model="claude-sonnet-4",
            input=[{"type": "message", "role": "user", "content": "Hi"}],
        )
        assert len(req.input) == 1

    def test_missing_model_rejected(self):
        from pydantic import ValidationError
        from kiro.models_responses import ResponsesApiRequest
        with pytest.raises(ValidationError):
            ResponsesApiRequest(input="Hello")

    def test_stream_defaults_true(self):
        from kiro.models_responses import ResponsesApiRequest
        req = ResponsesApiRequest(model="test")
        assert req.stream is True


# =============================================================================
# Streaming Path
# =============================================================================

class TestStreamingPath:
    """Tests for the streaming response path."""

    @pytest.mark.asyncio
    async def test_streaming_returns_event_stream(self):
        """Streaming mode returns StreamingResponse with correct headers."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from kiro.routes_responses import router

        app = FastAPI()
        app.include_router(router)

        # Mock app state
        mock_auth = MagicMock()
        mock_auth.api_host = "https://test.api.com"
        mock_cache = MagicMock()
        mock_resolver = MagicMock()
        app.state.auth_manager = mock_auth
        app.state.model_cache = mock_cache
        app.state.model_resolver = mock_resolver

        async def mock_stream(*args, **kwargs):
            yield 'event: response.created\ndata: {"type":"response.created"}\n\n'
            yield 'event: response.completed\ndata: {"type":"response.completed"}\n\n'

        with patch("kiro.routes_responses.stream_with_first_token_retry", side_effect=mock_stream), \
             patch("kiro.routes_responses.build_kiro_payload_responses", return_value={"test": True}), \
             patch("kiro.routes_responses.KiroHttpClient") as mock_http:
            mock_client_instance = MagicMock()
            mock_client_instance.client = MagicMock()
            mock_client_instance.close = AsyncMock()
            mock_http.return_value = mock_client_instance

            client = TestClient(app)
            response = client.post(
                "/v1/responses",
                json={"model": "claude-sonnet-4", "input": "Hello", "stream": True},
                headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
            )

            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            assert response.headers.get("cache-control") == "no-cache"
            assert response.headers.get("connection") == "keep-alive"


# =============================================================================
# Non-Streaming Path
# =============================================================================

class TestNonStreamingPath:
    """Tests for the non-streaming response path."""

    @pytest.mark.asyncio
    async def test_non_streaming_returns_json(self):
        """Non-streaming mode returns JSONResponse."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from kiro.routes_responses import router

        app = FastAPI()
        app.include_router(router)

        mock_auth = MagicMock()
        mock_auth.api_host = "https://test.api.com"
        mock_cache = MagicMock()
        app.state.auth_manager = mock_auth
        app.state.model_cache = mock_cache
        app.state.model_resolver = MagicMock()

        mock_kiro_response = MagicMock()
        mock_kiro_response.status_code = 200

        with patch("kiro.routes_responses.build_kiro_payload_responses", return_value={"test": True}), \
             patch("kiro.routes_responses.KiroHttpClient") as mock_http, \
             patch("kiro.routes_responses.collect_responses_response", new_callable=AsyncMock) as mock_collect:

            mock_client_instance = MagicMock()
            mock_client_instance.client = MagicMock()
            mock_client_instance.close = AsyncMock()
            mock_client_instance.request_with_retry = AsyncMock(return_value=mock_kiro_response)
            mock_http.return_value = mock_client_instance

            mock_collect.return_value = {
                "id": "resp_test",
                "object": "response",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "Hi"}]}],
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }

            client = TestClient(app)
            response = client.post(
                "/v1/responses",
                json={"model": "claude-sonnet-4", "input": "Hello", "stream": False},
                headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "resp_test"
            assert data["status"] == "completed"


# =============================================================================
# Error Handling
# =============================================================================

class TestErrorHandling:
    """Tests for error response handling."""

    @pytest.mark.asyncio
    async def test_non_streaming_kiro_error_returns_json(self):
        """Non-200 from Kiro returns JSONResponse with error details."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from kiro.routes_responses import router

        app = FastAPI()
        app.include_router(router)

        mock_auth = MagicMock()
        mock_auth.api_host = "https://test.api.com"
        app.state.auth_manager = mock_auth
        app.state.model_cache = MagicMock()
        app.state.model_resolver = MagicMock()

        mock_kiro_response = MagicMock()
        mock_kiro_response.status_code = 400
        mock_kiro_response.aread = AsyncMock(
            return_value=b'{"message":"Improperly formed request.","reason":null}'
        )

        with patch("kiro.routes_responses.build_kiro_payload_responses", return_value={"test": True}), \
             patch("kiro.routes_responses.KiroHttpClient") as mock_http:

            mock_client_instance = MagicMock()
            mock_client_instance.client = MagicMock()
            mock_client_instance.close = AsyncMock()
            mock_client_instance.request_with_retry = AsyncMock(return_value=mock_kiro_response)
            mock_http.return_value = mock_client_instance

            client = TestClient(app)
            response = client.post(
                "/v1/responses",
                json={"model": "claude-sonnet-4", "input": "Hello", "stream": False},
                headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
            )

            assert response.status_code == 400
            data = response.json()
            assert "error" in data
            assert data["error"]["type"] == "kiro_api_error"

    def test_payload_build_error_returns_400(self):
        """ValueError from payload builder returns 400."""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI
        from kiro.routes_responses import router

        app = FastAPI()
        app.include_router(router)

        mock_auth = MagicMock()
        mock_auth.api_host = "https://test.api.com"
        app.state.auth_manager = mock_auth
        app.state.model_cache = MagicMock()
        app.state.model_resolver = MagicMock()

        with patch("kiro.routes_responses.build_kiro_payload_responses", side_effect=ValueError("No messages")):
            client = TestClient(app)
            response = client.post(
                "/v1/responses",
                json={"model": "claude-sonnet-4", "input": "Hello", "stream": True},
                headers={"Authorization": f"Bearer {PROXY_API_KEY}"},
            )
            assert response.status_code == 400


# =============================================================================
# Truncation Recovery
# =============================================================================

class TestTruncationRecovery:
    """Tests for truncation recovery injection on inbound requests."""

    def test_function_call_output_modified_when_truncated(self):
        """function_call_output gets truncation notice prepended when state exists."""
        from kiro.truncation_state import save_tool_truncation, _tool_truncation_cache, _cache_lock

        # Setup: save truncation state
        save_tool_truncation("call_trunc_1", "exec_command", {"reason": "test truncation", "size_bytes": 5000})

        input_items = [
            {"type": "message", "role": "user", "content": "Hello"},
            {"type": "function_call", "call_id": "call_trunc_1", "name": "exec_command", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_trunc_1", "output": "original result"},
            {"type": "message", "role": "user", "content": "Continue"},
        ]

        from kiro.truncation_state import get_tool_truncation
        from kiro.truncation_recovery import generate_truncation_tool_result

        # Simulate what routes_responses.py does
        modified_input = []
        for item in input_items:
            if not isinstance(item, dict):
                modified_input.append(item)
                continue
            if item.get("type") == "function_call_output":
                call_id = item.get("call_id", "")
                info = get_tool_truncation(call_id)
                if info:
                    synthetic = generate_truncation_tool_result(
                        tool_name=info.tool_name,
                        tool_use_id=call_id,
                        truncation_info=info.truncation_info,
                    )
                    modified_item = {**item, "output": f"{synthetic['content']}\n\n---\n\nOriginal tool result:\n{item['output']}"}
                    modified_input.append(modified_item)
                    continue
            modified_input.append(item)

        # The function_call_output should have been modified
        fco = [i for i in modified_input if i.get("type") == "function_call_output"][0]
        assert "[API Limitation]" in fco["output"]
        assert "original result" in fco["output"]

        # Cleanup
        with _cache_lock:
            _tool_truncation_cache.clear()

    def test_function_call_output_unchanged_without_truncation(self):
        """function_call_output passes through when no truncation state."""
        from kiro.truncation_state import get_tool_truncation

        input_items = [
            {"type": "function_call_output", "call_id": "call_clean", "output": "clean result"},
        ]

        modified_input = []
        for item in input_items:
            if item.get("type") == "function_call_output":
                call_id = item.get("call_id", "")
                info = get_tool_truncation(call_id)
                if info:
                    continue  # would modify
            modified_input.append(item)

        assert len(modified_input) == 1
        assert modified_input[0]["output"] == "clean result"
