# -*- coding: utf-8 -*-

"""
Converters for transforming Responses API format to Kiro format.

This module converts the flat ResponseItem-based input from the Responses API
(used by Codex CLI) into the unified format used by converters_core.py.

The Responses API represents conversation history as a flat list of typed items:
- message (role=user/assistant/system/developer)
- function_call (assistant's tool invocation)
- function_call_output (tool result)
- reasoning (thinking/reasoning blocks)

These are converted to UnifiedMessage objects that converters_core.py understands.
"""

import json
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from kiro.config import HIDDEN_MODELS
from kiro.model_resolver import get_model_id_for_kiro
from kiro.converters_core import (
    extract_text_content,
    extract_images_from_content,
    UnifiedMessage,
    UnifiedTool,
    build_kiro_payload as core_build_kiro_payload,
)
from kiro.models_responses import ResponsesApiRequest


# ==================================================================================================
# Input Item Processing
# ==================================================================================================

def _extract_text_from_response_content(content: Any) -> str:
    """
    Extract text from Responses API content field.

    Content can be:
    - A plain string
    - A list of content blocks with type discriminator
      (input_text, output_text, text, etc.)

    Args:
        content: Content field from a Responses API input item

    Returns:
        Extracted text string
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, dict):
                block_type = block.get("type", "")
                if block_type in ("input_text", "output_text", "text"):
                    text_parts.append(block.get("text", ""))
                elif "text" in block:
                    text_parts.append(block["text"])
            elif isinstance(block, str):
                text_parts.append(block)
        return "".join(text_parts)
    return str(content)


def _extract_images_from_response_content(content: Any) -> List[Dict[str, Any]]:
    """
    Extract images from Responses API content blocks.

    Supports input_image blocks with image_url or data URL.

    Args:
        content: Content field from a Responses API input item

    Returns:
        List of images in unified format
    """
    if not isinstance(content, list):
        return []

    images = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type", "")
        if block_type == "input_image":
            image_url = block.get("image_url", "")
            if isinstance(image_url, dict):
                image_url = image_url.get("url", "")
            if image_url and image_url.startswith("data:"):
                # Parse data URL: data:image/jpeg;base64,/9j/...
                try:
                    header, data = image_url.split(",", 1)
                    media_type = header.split(":")[1].split(";")[0]
                    images.append({"media_type": media_type, "data": data})
                except (ValueError, IndexError):
                    logger.warning("Failed to parse image data URL")
    return images


def convert_responses_input_to_unified(
    input_items: Any,
    instructions: Optional[str] = None,
) -> Tuple[str, List[UnifiedMessage]]:
    """
    Convert Responses API input items to unified message format.

    Processes the flat list of typed items and groups them into
    UnifiedMessage objects that converters_core.py can handle.

    Function calls and their outputs are paired and converted to
    assistant tool_calls + user tool_results.

    Args:
        input_items: List of input items or a plain string
        instructions: System instructions (becomes system prompt)

    Returns:
        Tuple of (system_prompt, list of UnifiedMessage)
    """
    system_prompt = instructions or ""

    # Handle string input (simple prompt)
    if isinstance(input_items, str):
        return system_prompt, [UnifiedMessage(role="user", content=input_items)]

    if not input_items:
        return system_prompt, []

    unified_messages: List[UnifiedMessage] = []
    pending_tool_calls: List[Dict[str, Any]] = []
    pending_tool_results: List[Dict[str, Any]] = []

    for item in input_items:
        if isinstance(item, str):
            unified_messages.append(UnifiedMessage(role="user", content=item))
            continue

        if not isinstance(item, dict):
            continue

        item_type = item.get("type", "")

        if item_type == "message":
            # Flush any pending tool calls/results before processing message
            _flush_tool_pairs(unified_messages, pending_tool_calls, pending_tool_results)

            role = item.get("role", "user")
            content = item.get("content", "")
            text = _extract_text_from_response_content(content)
            images = _extract_images_from_response_content(content) or None

            # Map developer role to system prompt
            if role in ("system", "developer"):
                if text:
                    system_prompt = f"{system_prompt}\n{text}".strip() if system_prompt else text
                continue

            unified_messages.append(UnifiedMessage(
                role=role,
                content=text,
                images=images,
            ))

        elif item_type == "function_call":
            # Flush any pending tool results first
            if pending_tool_results:
                _flush_tool_pairs(unified_messages, pending_tool_calls, pending_tool_results)

            pending_tool_calls.append({
                "id": item.get("call_id", item.get("id", "")),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                },
            })

        elif item_type == "function_call_output":
            pending_tool_results.append({
                "type": "tool_result",
                "tool_use_id": item.get("call_id", item.get("id", "")),
                "content": item.get("output", "") or "(empty result)",
            })

        elif item_type == "reasoning":
            # Reasoning items are informational context from previous turns.
            # We skip them - the gateway handles its own thinking injection.
            continue

        else:
            logger.debug(f"Skipping unknown input item type: {item_type}")

    # Flush remaining tool pairs
    _flush_tool_pairs(unified_messages, pending_tool_calls, pending_tool_results)

    return system_prompt, unified_messages


def _flush_tool_pairs(
    messages: List[UnifiedMessage],
    tool_calls: List[Dict[str, Any]],
    tool_results: List[Dict[str, Any]],
) -> None:
    """
    Flush accumulated tool calls and results into unified messages.

    Creates an assistant message with tool_calls followed by a user
    message with tool_results, matching the pattern expected by
    converters_core.py.

    Args:
        messages: List to append messages to (mutated in place)
        tool_calls: Accumulated tool calls (cleared after flush)
        tool_results: Accumulated tool results (cleared after flush)
    """
    if tool_calls:
        messages.append(UnifiedMessage(
            role="assistant",
            content="",
            tool_calls=tool_calls.copy(),
        ))
        tool_calls.clear()

    if tool_results:
        messages.append(UnifiedMessage(
            role="user",
            content="",
            tool_results=tool_results.copy(),
        ))
        tool_results.clear()


# ==================================================================================================
# Tool Conversion
# ==================================================================================================

# Client-side tool types that have no meaningful schema to convert.
# These are implicit capabilities or UI-only features — skip silently.
_SKIP_TOOL_TYPES = frozenset({
    "local_shell",        # bare shell tool — already covered by exec_command
    "web_search",         # client-side web search
    "image_generation",   # client-side image generation
    "tool_search",        # client-side tool discovery
})


def _convert_custom_tool_to_function(tool: Dict[str, Any]) -> Optional[UnifiedTool]:
    """
    Convert a custom/freeform tool (e.g. apply_patch) to a standard function tool.

    Codex CLI sends tools like apply_patch as type="custom" with a freeform
    grammar format. Kiro API doesn\'t support custom tool types, but the model
    can call them as standard function tools with a simple {input: string} schema.

    The response flows back as a function_call event, and Codex CLI intercepts
    it by tool name — so the model just needs to know the tool exists and how
    to format its input.

    Args:
        tool: Tool definition dict with type="custom"

    Returns:
        UnifiedTool if conversion succeeded, None otherwise
    """
    name = tool.get("name", "")
    if not name:
        logger.warning("Custom tool has no name, skipping")
        return None

    description = tool.get("description", "")

    # Build a simple {input: string} schema — matches Codex\'s own
    # create_apply_patch_json_tool() fallback for non-freeform models
    input_schema = {
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": f"The complete input for the {name} tool",
            }
        },
        "required": ["input"],
        "additionalProperties": False,
    }

    logger.info(
        f"Converted custom tool \'{name}\' to function tool "
        f"(description={len(description)} chars)"
    )

    return UnifiedTool(
        name=name,
        description=description,
        input_schema=input_schema,
    )


def convert_responses_tools_to_unified(tools: Optional[List[Any]]) -> Optional[List[UnifiedTool]]:
    """
    Convert Responses API tool definitions to unified format.

    Converts function tools directly and custom/freeform tools (like apply_patch)
    to function tools with a simple {input: string} schema.

    Client-side tools without meaningful schemas (local_shell, web_search, etc.)
    are skipped.

    Args:
        tools: List of tool definitions from the request

    Returns:
        List of UnifiedTool objects, or None if no tools
    """
    if not tools:
        return None

    logger.debug(f"Received {len(tools)} tools from client")

    unified_tools = []
    converted_custom = 0
    skipped = 0
    dropped_unknown = 0

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        tool_type = tool.get("type", "function")
        tool_name = tool.get("name", "unnamed")

        logger.debug(f"  Tool: {tool_name} (type={tool_type})")

        # Standard function tools — convert to Kiro format
        if tool_type == "function":
            name = tool.get("name", "")
            if not name:
                # Try nested function format (OpenAI Chat Completions style)
                func = tool.get("function", {})
                if isinstance(func, dict):
                    name = func.get("name", "")
                    unified_tools.append(UnifiedTool(
                        name=name,
                        description=func.get("description"),
                        input_schema=func.get("parameters"),
                    ))
                    continue

            unified_tools.append(UnifiedTool(
                name=name,
                description=tool.get("description"),
                input_schema=tool.get("parameters"),
            ))
            continue

        # Custom/freeform tools — convert to function tools
        if tool_type == "custom":
            converted = _convert_custom_tool_to_function(tool)
            if converted:
                unified_tools.append(converted)
                converted_custom += 1
            continue

        # Client-side tools without schemas — skip silently
        if tool_type in _SKIP_TOOL_TYPES:
            skipped += 1
            logger.debug(f"  Skipping client-side tool '{tool_name}' (type={tool_type})")
            continue

        # Unknown tool type — log warning
        dropped_unknown += 1
        logger.warning(
            f"Dropping tool '{tool_name}' with unrecognized type '{tool_type}' "
            f"— not converted to Kiro format"
        )

    logger.debug(
        f"Converted {len(unified_tools)} tools to unified format"
        + (f", {converted_custom} from custom" if converted_custom > 0 else "")
        + (f", skipped {skipped} client-side" if skipped > 0 else "")
        + (f", dropped {dropped_unknown} unknown" if dropped_unknown > 0 else "")
    )

    return unified_tools if unified_tools else None


def _extract_text_format_config(text_config):
    """
    Extract text format configuration from the request.

    Supports json_schema format for structured output.

    Args:
        text_config: The 'text' field from the request

    Returns:
        Format configuration dict if applicable, None otherwise
    """
    if not text_config or not isinstance(text_config, dict):
        return None

    fmt = text_config.get("format")
    if not fmt or not isinstance(fmt, dict):
        return None

    fmt_type = fmt.get("type")
    if fmt_type == "json_schema":
        return fmt

    return None

def build_kiro_payload_responses(
    request_data: ResponsesApiRequest,
    conversation_id: str,
    profile_arn: str,
) -> dict:
    """
    Build complete Kiro API payload from a Responses API request.

    This is the main entry point for Responses API -> Kiro conversion.
    Converts the flat item-based input to unified messages, then uses
    the shared core builder.

    Args:
        request_data: Request in Responses API format
        conversation_id: Unique conversation ID
        profile_arn: AWS CodeWhisperer profile ARN

    Returns:
        Payload dictionary for POST request to Kiro API

    Raises:
        ValueError: If there are no messages to send
    """
    # Convert input items to unified format
    system_prompt, unified_messages = convert_responses_input_to_unified(
        request_data.input,
        request_data.instructions,
    )

    # Convert tools to unified format
    unified_tools = convert_responses_tools_to_unified(request_data.tools)


    # Get model ID for Kiro API
    model_id = get_model_id_for_kiro(request_data.model, HIDDEN_MODELS)

    logger.debug(
        f"Converting Responses API request: model={request_data.model} -> {model_id}, "
        f"messages={len(unified_messages)}, tools={len(unified_tools) if unified_tools else 0}, "
        f"system_prompt_length={len(system_prompt)}"
    )

    # Use core function to build payload
    result = core_build_kiro_payload(
        messages=unified_messages,
        system_prompt=system_prompt,
        model_id=model_id,
        tools=unified_tools,
        conversation_id=conversation_id,
        profile_arn=profile_arn,
        inject_thinking=True,
    )

    return result.payload
