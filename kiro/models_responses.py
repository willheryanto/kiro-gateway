# -*- coding: utf-8 -*-

"""
Pydantic models for OpenAI Responses API (/v1/responses).

Defines data schemas for the Responses API format used by Codex CLI.
This is a different wire protocol from Chat Completions - it uses
ResponseItem-based input/output instead of chat messages.
"""

import time
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, model_validator


# ==================================================================================================
# Input Item Models
# ==================================================================================================

class ResponseInputTextContent(BaseModel):
    """Text content block within a message input item."""
    type: str = "input_text"
    text: str = ""


class ResponseInputImageContent(BaseModel):
    """Image content block within a message input item."""
    type: str = "input_image"
    image_url: Optional[str] = None
    detail: Optional[str] = None


class ResponseInputMessage(BaseModel):
    """
    A message input item in the Responses API format.

    Can represent user, assistant, system, or developer messages.
    Content can be a string or a list of content blocks.
    """
    type: str = "message"
    id: Optional[str] = None
    role: str = "user"
    content: Optional[Union[str, List[Any]]] = None
    status: Optional[str] = None

    model_config = {"extra": "allow"}


class ResponseInputFunctionCall(BaseModel):
    """A function_call input item (assistant's tool invocation)."""
    type: str = "function_call"
    id: Optional[str] = None
    call_id: Optional[str] = None
    name: str = ""
    arguments: str = "{}"
    status: Optional[str] = None

    model_config = {"extra": "allow"}


class ResponseInputFunctionCallOutput(BaseModel):
    """A function_call_output input item (tool result)."""
    type: str = "function_call_output"
    id: Optional[str] = None
    call_id: Optional[str] = None
    output: str = ""
    status: Optional[str] = None

    model_config = {"extra": "allow"}


class ResponseInputItemParam(BaseModel):
    """
    Union-like model for any input item in the Responses API.

    Codex sends a heterogeneous list of items with a 'type' discriminator.
    We accept everything loosely and dispatch on 'type' in the converter.
    """
    type: str
    id: Optional[str] = None
    role: Optional[str] = None
    content: Optional[Union[str, List[Any]]] = None
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None
    output: Optional[str] = None
    status: Optional[str] = None
    summary: Optional[List[Any]] = None

    model_config = {"extra": "allow"}


# ==================================================================================================
# Tool Models
# ==================================================================================================

class ResponseToolFunction(BaseModel):
    """Function tool definition in Responses API format."""
    type: str = "function"
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = None

    model_config = {"extra": "allow"}


# ==================================================================================================
# Reasoning Models
# ==================================================================================================

class ResponseReasoning(BaseModel):
    """Reasoning configuration for the Responses API."""
    effort: Optional[str] = None
    summary: Optional[str] = None

    model_config = {"extra": "allow"}


# ==================================================================================================
# Main Request Model
# ==================================================================================================

class ResponsesApiRequest(BaseModel):
    """
    Request body for POST /v1/responses.

    This is the Responses API format used by Codex CLI.
    It differs from Chat Completions in that conversation history
    is represented as a flat list of typed items rather than messages.

    Attributes:
        model: Model ID for generation
        instructions: System-level instructions (equivalent to system message)
        input: List of input items (messages, function calls, function outputs)
        tools: List of tool definitions
        tool_choice: Tool selection strategy
        parallel_tool_calls: Whether to allow parallel tool calls
        reasoning: Reasoning/thinking configuration
        store: Whether to store the response server-side
        stream: Whether to stream the response
        include: List of additional fields to include in response
        service_tier: Service tier preference
        temperature: Generation temperature
        top_p: Top-p sampling
        max_output_tokens: Maximum output tokens
        previous_response_id: ID of previous response for continuation
    """
    model: str
    instructions: Optional[str] = None
    input: Optional[Union[str, List[Any]]] = None
    tools: Optional[List[Any]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    reasoning: Optional[ResponseReasoning] = None
    store: Optional[bool] = None
    stream: bool = True
    include: Optional[List[str]] = None
    service_tier: Optional[str] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_output_tokens: Optional[int] = None
    previous_response_id: Optional[str] = None
    text: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}

    @model_validator(mode="after")
    def validate_input_not_empty(self) -> "ResponsesApiRequest":
        """
        Validate that input is not an empty list.

        A string input or None is acceptable (None means no history),
        but an empty list is always a client error.

        Raises:
            ValueError: If input is an empty list
        """
        if isinstance(self.input, list) and len(self.input) == 0:
            raise ValueError("input must not be an empty list")
        return self


# ==================================================================================================
# Response Models (for type safety in streaming and route output)
# ==================================================================================================

class ResponsesApiUsage(BaseModel):
    """Usage statistics for a Responses API response."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    input_tokens_details: Optional[Dict[str, Any]] = None
    output_tokens_details: Optional[Dict[str, Any]] = None

    model_config = {"extra": "allow"}


class ResponsesApiError(BaseModel):
    """Error information in a failed response."""
    type: str = "server_error"
    message: str = ""

    model_config = {"extra": "allow"}


class ResponsesApiOutputItem(BaseModel):
    """
    An output item in the Responses API response.

    Can be a message (with content) or a function_call.
    """
    id: Optional[str] = None
    type: str = "message"
    role: Optional[str] = None
    status: str = "completed"
    content: Optional[List[Any]] = None
    # function_call fields
    call_id: Optional[str] = None
    name: Optional[str] = None
    arguments: Optional[str] = None

    model_config = {"extra": "allow"}


class ResponsesApiResponse(BaseModel):
    """
    Complete response object for the Responses API.

    Used for non-streaming responses and as the payload
    in response.completed streaming events.
    """
    id: str
    object: str = "response"
    created_at: int = 0
    model: str = ""
    status: str = "completed"
    output: List[ResponsesApiOutputItem] = Field(default_factory=list)
    usage: Optional[ResponsesApiUsage] = None
    error: Optional[ResponsesApiError] = None

    model_config = {"extra": "allow"}
