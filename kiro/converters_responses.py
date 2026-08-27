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
Converter from OpenAI Responses API format to Kiro API format.

The Responses API sends a flat ``input`` array of typed items rather than a
nested message list. Tool calls and their outputs are peer items alongside
messages, linked by ``call_id``:

    [ {type: message, role: user, ...},
      {type: function_call, call_id: c1, name: shell, arguments: "{...}"},
      {type: function_call_output, call_id: c1, output: "..."},
      {type: message, role: user, ...} ]

This module folds that sequence back into the gateway's canonical
:class:`~kiro.converters_core.UnifiedMessage` list and delegates payload
assembly to the shared core builder, exactly as the OpenAI and Anthropic
converters do.

Item types Kiro cannot represent (``reasoning``, ``local_shell_call``,
``tool_search_call``, compaction variants...) are logged and skipped rather than
rejected, so a newer client cannot break the endpoint outright.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from kiro.config import HIDDEN_MODELS
from kiro.converters_core import (
    ThinkingConfig,
    UnifiedMessage,
    UnifiedTool,
    build_kiro_payload as core_build_kiro_payload,
    extract_text_content,
)
from kiro.converters_openai import reasoning_effort_to_budget
from kiro.model_resolver import get_model_id_for_kiro
from kiro.models_responses import ResponsesRequest

# Item types that carry no information Kiro can use. Skipped silently at debug
# level rather than warned about, since Codex sends `reasoning` on every turn
# after the first.
_IGNORED_ITEM_TYPES = frozenset({
    "reasoning",
    "web_search_call",
    "image_generation_call",
    "compaction",
    "compaction_summary",
    "context_compaction",
    # Tool definitions, not conversation content. Harvested separately by
    # extract_additional_tools() before message conversion.
    "additional_tools",
})

# Namespace that needs no qualification when a tool call is reported back.
_DEFAULT_TOOL_NAMESPACE = "functions"

# Synthetic property used to give freeform ("custom") tools a JSON schema, since
# Kiro requires one but such tools actually take raw text. The streaming layer
# unwraps this field when reporting the call back to the client.
FREEFORM_TOOL_INPUT_KEY = "input"

# Roles that carry instructions rather than conversation turns.
_INSTRUCTION_ROLES = frozenset({"system", "developer"})

# Default output ceiling when the client does not send max_output_tokens.
# Matches the OpenAI converter's fallback for consistency.
_DEFAULT_MAX_OUTPUT_TOKENS = 4096


def extract_output_text(output: Any) -> str:
    """
    Extract text from a function_call_output ``output`` field.

    The Responses protocol serializes this field as either a plain string or an
    array of content items, with no type tag to distinguish them.

    Args:
        output: Raw ``output`` value from a function_call_output item.

    Returns:
        Extracted text, or an empty string when there is none.

    Examples:
        >>> extract_output_text("done")
        'done'
        >>> extract_output_text([{"type": "output_text", "text": "done"}])
        'done'
    """
    if output is None:
        return ""
    if isinstance(output, str):
        return output
    # Lists and dicts both reduce to text via the shared extractor, which
    # already understands {"text": ...} content items.
    return extract_text_content(output)


def _build_tool_call(item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a tool-call item to the gateway's unified tool_call shape.

    ``function_call`` items carry ``arguments`` as a JSON string, which is what
    the core layer expects, so those pass through untouched.

    ``custom_tool_call`` items instead carry raw text in ``input``. The core
    layer parses this field with ``json.loads``, so raw text is re-wrapped under
    the same synthetic key used to give freeform tools a schema on the way out
    (see FREEFORM_TOOL_INPUT_KEY). Without that wrapping the whole turn fails
    with an opaque "Expecting value" conversion error.

    Args:
        item: A function_call or custom_tool_call input item.

    Returns:
        Tool call dict in unified format.
    """
    name = item.get("name", "")
    raw = item.get("arguments")
    if raw is None:
        raw = item.get("input")

    if item.get("type") == "custom_tool_call":
        # Raw text by definition; wrap it under the synthetic key.
        text = raw if isinstance(raw, str) else ("" if raw is None else str(raw))
        arguments = json.dumps({FREEFORM_TOOL_INPUT_KEY: text}, ensure_ascii=False)
    elif raw is None or raw == "":
        arguments = "{}"
    elif not isinstance(raw, str):
        # Already structured; the core layer accepts dicts directly.
        arguments = json.dumps(raw, ensure_ascii=False)
    else:
        arguments = raw
        # Defensive: a non-JSON arguments string would abort the whole turn, so
        # preserve the text instead of failing the conversion.
        try:
            json.loads(arguments)
        except (json.JSONDecodeError, TypeError):
            logger.warning(
                f"Tool call {name!r} sent non-JSON arguments; wrapping as "
                f"{FREEFORM_TOOL_INPUT_KEY!r}"
            )
            arguments = json.dumps({FREEFORM_TOOL_INPUT_KEY: arguments}, ensure_ascii=False)

    return {
        "id": item.get("call_id") or item.get("id") or "",
        "type": "function",
        "function": {
            "name": name,
            "arguments": arguments,
        },
    }


def convert_responses_input_to_unified(
    input_items: Any,
    instructions: Optional[str] = None,
) -> Tuple[str, List[UnifiedMessage]]:
    """
    Convert a Responses ``input`` array to unified messages.

    Tool calls are attached to the assistant message that precedes them, and
    tool outputs are flushed into a synthetic user message when the next
    conversational item arrives - mirroring how the OpenAI converter handles
    ``role="tool"`` messages.

    Args:
        input_items: The request's ``input`` field (string or list of items).
        instructions: Top-level ``instructions``, used as the system prompt base.

    Returns:
        Tuple of (system_prompt, unified_messages).
    """
    system_parts: List[str] = []
    if instructions:
        system_parts.append(instructions)

    # A bare string prompt is shorthand for a single user message.
    if isinstance(input_items, str):
        text = input_items.strip()
        messages = [UnifiedMessage(role="user", content=input_items)] if text else []
        return "\n".join(system_parts).strip(), messages

    if not isinstance(input_items, list):
        return "\n".join(system_parts).strip(), []

    processed: List[UnifiedMessage] = []
    pending_tool_results: List[Dict[str, Any]] = []
    total_tool_calls = 0
    total_tool_results = 0
    skipped_types: Dict[str, int] = {}

    def flush_tool_results() -> None:
        """Emit accumulated tool outputs as one synthetic user message."""
        if pending_tool_results:
            processed.append(UnifiedMessage(
                role="user",
                content="",
                tool_results=pending_tool_results.copy(),
            ))
            pending_tool_results.clear()

    for item in input_items:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")

        # Items with no Kiro equivalent. Dropping `reasoning` is safe: it
        # carries no call_id, so it cannot break function_call/output pairing.
        if item_type in _IGNORED_ITEM_TYPES:
            skipped_types[item_type] = skipped_types.get(item_type, 0) + 1
            continue

        if item_type in ("function_call_output", "custom_tool_call_output"):
            call_id = item.get("call_id") or item.get("id") or ""
            content = extract_output_text(item.get("output"))
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": call_id,
                "content": content or "(empty result)",
            })
            total_tool_results += 1
            continue

        if item_type in ("function_call", "custom_tool_call", "tool_search_call"):
            flush_tool_results()
            tool_call = _build_tool_call(item)
            # Attach to the preceding assistant turn when there is one, so the
            # history keeps its natural assistant-then-result shape.
            if processed and processed[-1].role == "assistant" and not processed[-1].tool_results:
                if processed[-1].tool_calls is None:
                    processed[-1].tool_calls = []
                processed[-1].tool_calls.append(tool_call)
            else:
                processed.append(UnifiedMessage(
                    role="assistant",
                    content="",
                    tool_calls=[tool_call],
                ))
            total_tool_calls += 1
            continue

        if item_type in ("message", "agent_message", None):
            role = item.get("role") or ("assistant" if item_type == "agent_message" else "user")
            text = extract_text_content(item.get("content"))

            # Instruction-bearing roles belong in the system prompt, not the
            # conversation history.
            if role in _INSTRUCTION_ROLES:
                if text:
                    system_parts.append(text)
                continue

            flush_tool_results()
            processed.append(UnifiedMessage(role=role, content=text))
            continue

        # Unrecognized type: skip it rather than failing the whole request.
        skipped_types[item_type] = skipped_types.get(item_type, 0) + 1

    flush_tool_results()

    if skipped_types:
        summary = ", ".join(f"{name}={count}" for name, count in sorted(skipped_types.items()))
        logger.debug(f"Skipped unsupported Responses input items: {summary}")

    if total_tool_calls or total_tool_results:
        logger.debug(
            f"Converted Responses input: {len(processed)} messages, "
            f"{total_tool_calls} tool_calls, {total_tool_results} tool_results"
        )

    return "\n".join(part for part in system_parts if part).strip(), processed


def extract_additional_tools(input_items: Any) -> List[Dict[str, Any]]:
    """
    Harvest tool definitions carried inside ``additional_tools`` input items.

    Newer models (e.g. ``gpt-5.6-sol``) use the "responses-lite" wire variant:
    the top-level ``tools`` field is empty and definitions travel as an
    ``additional_tools`` input item instead, grouped into ``namespace``
    containers::

        {"type": "additional_tools", "role": "developer", "tools": [
            {"type": "namespace", "name": "functions", "tools": [
                {"type": "custom", "name": "exec", ...},
                {"type": "function", "name": "wait", "parameters": {...}}]}]}

    Only the default ``functions`` namespace is harvested. Tools in other
    namespaces would have to be reported back with a ``namespace`` field on the
    call item, which the streaming layer cannot reconstruct from a tool name
    alone, so they are skipped and logged rather than offered and then misrouted.

    Args:
        input_items: The request's ``input`` field.

    Returns:
        Flat list of tool definition dicts, empty when there are none.
    """
    if not isinstance(input_items, list):
        return []

    harvested: List[Dict[str, Any]] = []
    skipped_namespaces: Dict[str, int] = {}

    for item in input_items:
        if not isinstance(item, dict) or item.get("type") != "additional_tools":
            continue

        for entry in item.get("tools") or []:
            if not isinstance(entry, dict):
                continue

            if entry.get("type") == "namespace":
                namespace = entry.get("name") or ""
                nested = [t for t in (entry.get("tools") or []) if isinstance(t, dict)]
                if namespace == _DEFAULT_TOOL_NAMESPACE:
                    harvested.extend(nested)
                else:
                    skipped_namespaces[namespace] = len(nested)
                continue

            # Ungrouped tool definition.
            harvested.append(entry)

    if skipped_namespaces:
        detail = ", ".join(f"{ns}({n} tools)" for ns, n in sorted(skipped_namespaces.items()))
        logger.info(f"Skipped non-default tool namespaces from additional_tools: {detail}")

    if harvested:
        logger.debug(f"Harvested {len(harvested)} tools from additional_tools input items")

    return harvested


def collect_request_tools(request_data: "ResponsesRequest") -> List[Dict[str, Any]]:
    """
    Collect every tool definition a request offers, from both wire variants.

    Standard requests put definitions in the top-level ``tools`` field;
    responses-lite requests carry them in an ``additional_tools`` input item.
    A request may in principle use both, so the two sources are merged with
    top-level definitions taking precedence on a name collision.

    Args:
        request_data: Validated Responses request.

    Returns:
        Flat list of tool definition dicts.
    """
    top_level = [t for t in (request_data.tools or []) if isinstance(t, dict)]
    seen = {t.get("name") for t in top_level if t.get("name")}

    merged = list(top_level)
    for tool in extract_additional_tools(request_data.input):
        name = tool.get("name")
        if name and name in seen:
            continue
        if name:
            seen.add(name)
        merged.append(tool)

    return merged


def convert_responses_tools_to_unified(
    tools: Optional[List[Dict[str, Any]]]
) -> Optional[List[UnifiedTool]]:
    """
    Convert Responses tool definitions to unified format.

    Responses flattens function tools to ``{type, name, description,
    parameters}``. The nested Chat Completions shape is also accepted, since
    some clients reuse it.

    Args:
        tools: Tool definitions from the request.

    Returns:
        List of UnifiedTool objects, or None when there are none usable.
    """
    if not tools:
        return None

    unified_tools: List[UnifiedTool] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue

        tool_type = tool.get("type", "function")
        if tool_type not in ("function", "custom"):
            # Server-side tool types (web_search, tool_search...) have no Kiro
            # equivalent and are executed by the provider, not the model.
            logger.debug(f"Skipping unsupported Responses tool type: {tool_type}")
            continue

        # Nested Chat Completions shape.
        function = tool.get("function")
        if isinstance(function, dict):
            name = function.get("name")
            if not name:
                continue
            unified_tools.append(UnifiedTool(
                name=name,
                description=function.get("description"),
                input_schema=function.get("parameters"),
            ))
            continue

        name = tool.get("name")
        if not name:
            logger.warning("Skipping Responses tool with no name")
            continue

        input_schema = tool.get("parameters") or tool.get("input_schema")
        if input_schema is None and tool_type == "custom":
            # Freeform tools take raw text and carry no JSON schema, but Kiro
            # requires one. Expose a single `input` string so the model can still
            # call the tool; the streaming layer unwraps that field when
            # reporting the call back. See FREEFORM_TOOL_INPUT_KEY.
            input_schema = {
                "type": "object",
                "properties": {
                    FREEFORM_TOOL_INPUT_KEY: {
                        "type": "string",
                        "description": "Raw input text for this tool.",
                    }
                },
                "required": [FREEFORM_TOOL_INPUT_KEY],
            }

        unified_tools.append(UnifiedTool(
            name=name,
            description=tool.get("description"),
            input_schema=input_schema,
        ))

    return unified_tools if unified_tools else None


def collect_freeform_tool_names(tools: Optional[List[Dict[str, Any]]]) -> List[str]:
    """
    List the names of freeform ("custom") tools.

    Calls to these tools must be reported back as ``custom_tool_call`` items
    carrying raw ``input``, not as ``function_call`` items with JSON
    ``arguments``, so the streaming layer needs to know which names they are.

    Args:
        tools: Tool definitions, already flattened.

    Returns:
        Names of tools declared with ``type: "custom"``.
    """
    if not tools:
        return []
    return [
        tool["name"] for tool in tools
        if isinstance(tool, dict) and tool.get("type") == "custom" and tool.get("name")
    ]


def extract_thinking_config_from_responses(request: ResponsesRequest) -> ThinkingConfig:
    """
    Build the thinking configuration from a Responses request.

    Reads ``reasoning.effort``:
    - absent          -> enabled with the default budget
    - "none"          -> disabled
    - known effort    -> enabled with a budget scaled to max_output_tokens
    - unknown effort  -> enabled with the default budget

    Args:
        request: Validated Responses request.

    Returns:
        ThinkingConfig for the core layer.
    """
    reasoning = request.reasoning or {}
    effort = reasoning.get("effort") if isinstance(reasoning, dict) else None

    if not effort:
        return ThinkingConfig(enabled=True, budget_tokens=None)

    if effort == "none":
        return ThinkingConfig(enabled=False, budget_tokens=None)

    max_tokens = request.max_output_tokens or _DEFAULT_MAX_OUTPUT_TOKENS
    try:
        budget = reasoning_effort_to_budget(max_tokens, effort)
    except KeyError:
        logger.warning(f"Unknown reasoning effort '{effort}', using default thinking budget")
        return ThinkingConfig(enabled=True, budget_tokens=None)

    logger.debug(
        f"Extracted thinking config from Responses: effort='{effort}', "
        f"max_output_tokens={max_tokens}, budget={budget}"
    )
    return ThinkingConfig(enabled=True, budget_tokens=budget)


def responses_to_kiro(
    request_data: ResponsesRequest,
    conversation_id: str,
    profile_arn: str
) -> dict:
    """
    Build a complete Kiro API payload from a Responses request.

    Main entry point for Responses -> Kiro conversion.

    Args:
        request_data: Request in Responses format.
        conversation_id: Unique conversation ID.
        profile_arn: AWS CodeWhisperer profile ARN.

    Returns:
        Payload dictionary for the Kiro API.

    Raises:
        ValueError: If there are no messages to send.
    """
    system_prompt, unified_messages = convert_responses_input_to_unified(
        request_data.input,
        request_data.instructions,
    )

    # Tools arrive either in the top-level field or, for responses-lite models,
    # inside an additional_tools input item.
    unified_tools = convert_responses_tools_to_unified(
        collect_request_tools(request_data)
    )

    # Pass-through principle: normalize and let Kiro reject invalid models.
    model_id = get_model_id_for_kiro(request_data.model, HIDDEN_MODELS)
    thinking_config = extract_thinking_config_from_responses(request_data)

    logger.debug(
        f"Converting Responses request: model={request_data.model} -> {model_id}, "
        f"messages={len(unified_messages)}, tools={len(unified_tools) if unified_tools else 0}, "
        f"system_prompt_length={len(system_prompt)}, "
        f"thinking_enabled={thinking_config.enabled}, thinking_budget={thinking_config.budget_tokens}"
    )

    result = core_build_kiro_payload(
        messages=unified_messages,
        system_prompt=system_prompt,
        model_id=model_id,
        tools=unified_tools,
        conversation_id=conversation_id,
        profile_arn=profile_arn,
        thinking_config=thinking_config,
    )

    return result.payload
