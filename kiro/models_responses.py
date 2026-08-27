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
Pydantic models for the OpenAI Responses API (/v1/responses).

The Responses API replaces Chat Completions in newer OpenAI clients - notably
the Codex CLI, which speaks only this protocol. Requests carry a flat ``input``
array of typed items instead of a nested message list.

``input`` items are kept as plain dicts rather than a discriminated union: the
protocol defines a dozen item types (message, function_call, reasoning,
local_shell_call, compaction variants...) and clients add more over time.
Converting only the types Kiro can use, and ignoring the rest, keeps this
forward-compatible. See kiro.converters_responses for that mapping.
"""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


class ResponsesRequest(BaseModel):
    """
    Request body for POST /v1/responses.

    Only ``model`` and ``input`` are required. Several fields are accepted but
    deliberately unused because Kiro has no equivalent: ``store`` (no response
    persistence), ``include`` (no encrypted reasoning to return),
    ``prompt_cache_key``, ``service_tier`` and ``client_metadata``.

    Attributes:
        model: Client-supplied model name, normalized during conversion.
        input: Flat array of typed input items, or a bare string prompt.
        instructions: System prompt for this turn.
        tools: Tool definitions in Responses format.
        tool_choice: Tool selection strategy. Codex always sends "auto".
        parallel_tool_calls: Whether the model may emit several calls at once.
        reasoning: Reasoning controls, e.g. {"effort": "high"}.
        stream: Must be true - this endpoint is streaming-only.
        max_output_tokens: Output token ceiling, used for the thinking budget.
        text: Output format controls (verbosity, JSON schema).
        store: Accepted and ignored - responses are never persisted.
        include: Accepted and ignored.
        previous_response_id: Accepted and ignored - no server-side chaining.
    """

    model: str
    input: Union[str, List[Dict[str, Any]]] = Field(default_factory=list)
    instructions: Optional[str] = None
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    parallel_tool_calls: Optional[bool] = None
    reasoning: Optional[Dict[str, Any]] = None
    stream: bool = True
    max_output_tokens: Optional[int] = None
    text: Optional[Dict[str, Any]] = None
    store: Optional[bool] = None
    include: Optional[List[str]] = None
    previous_response_id: Optional[str] = None

    # Forward compatibility: tolerate unknown fields from newer clients.
    model_config = {"extra": "allow"}
