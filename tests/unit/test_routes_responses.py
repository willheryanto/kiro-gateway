# -*- coding: utf-8 -*-

"""
Unit tests for the OpenAI Responses API endpoint (/v1/responses).

Covers the wire-contract invariants a Responses client depends on:
- response.created arrives first, response.completed last and exactly once
- tool calls are delivered whole in response.output_item.done, never as
  incremental argument fragments (clients ignore those events entirely)
- endpoint-level behaviour: bearer auth, streaming-only rejection
"""

import json

import pytest
from unittest.mock import MagicMock, patch

from kiro.converters_responses import (
    collect_freeform_tool_names,
    collect_request_tools,
    convert_responses_input_to_unified,
    convert_responses_tools_to_unified,
    extract_additional_tools,
    extract_output_text,
)
from kiro.models_responses import ResponsesRequest
from kiro.streaming_core import KiroEvent
from kiro.streaming_responses import stream_kiro_to_responses_internal


# ==================================================================================================
# Fixtures and helpers
# ==================================================================================================

@pytest.fixture
def mock_model_cache():
    """Mock for ModelInfoCache."""
    cache = MagicMock()
    cache.get_max_input_tokens.return_value = 200000
    return cache


@pytest.fixture
def mock_auth_manager():
    """Mock for KiroAuthManager."""
    return MagicMock()


@pytest.fixture
def mock_response():
    """Mock for httpx.Response."""
    response = MagicMock()
    response.status_code = 200
    return response


def parse_sse_events(frames):
    """
    Parse SSE frames into event dicts.

    Args:
        frames: List of raw SSE frame strings.

    Returns:
        List of decoded event payloads, in arrival order.
    """
    events = []
    for frame in frames:
        for line in frame.split("\n"):
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
    return events


async def collect_stream(events_to_yield, model_cache, auth_manager, response,
                         freeform_tool_names=None):
    """
    Run the emitter over a faked Kiro stream and return the decoded events.

    Args:
        events_to_yield: KiroEvent objects the faked upstream stream produces.
        model_cache: Model cache mock.
        auth_manager: Auth manager mock.
        response: Response mock.
        freeform_tool_names: Names the client declared as freeform tools.

    Returns:
        List of decoded SSE event payloads.
    """
    async def fake_parse_kiro_stream(*args, **kwargs):
        for event in events_to_yield:
            yield event

    frames = []
    with patch('kiro.streaming_responses.parse_kiro_stream', fake_parse_kiro_stream):
        with patch('kiro.streaming_responses.parse_bracket_tool_calls', return_value=[]):
            async for frame in stream_kiro_to_responses_internal(
                response, "claude-sonnet-4-5", model_cache, auth_manager,
                response_id="resp_test",
                freeform_tool_names=freeform_tool_names,
            ):
                frames.append(frame)
    return parse_sse_events(frames)


# Tool payload as captured from a real gpt-5.6-sol request: definitions travel
# inside an additional_tools input item, grouped by namespace.
LITE_ADDITIONAL_TOOLS = {
    "type": "additional_tools",
    "role": "developer",
    "tools": [
        {"type": "namespace", "name": "functions", "description": "Default tools.", "tools": [
            {"type": "custom", "name": "exec", "description": "Run a command."},
            {"type": "function", "name": "wait", "description": "Wait.",
             "parameters": {"type": "object", "properties": {"ms": {"type": "number"}}}},
        ]},
        {"type": "namespace", "name": "collaboration", "description": "Agents.", "tools": [
            {"type": "function", "name": "spawn_agent",
             "parameters": {"type": "object", "properties": {}}},
        ]},
    ],
}


# ==================================================================================================
# Tests for the SSE event sequence
# ==================================================================================================

class TestResponsesEventSequence:
    """Tests for the emitted Responses SSE event sequence."""

    @pytest.mark.asyncio
    async def test_text_only_stream_order(self, mock_model_cache, mock_auth_manager, mock_response):
        """
        What it does: Verifies the minimal valid stream for a text-only turn.
        Goal: created -> output_text.delta -> output_item.done(message) -> completed.
        """
        print("Setup: Kiro stream with two content events...")
        events = await collect_stream(
            [KiroEvent(type="content", content="Hello"),
             KiroEvent(type="content", content=" world")],
            mock_model_cache, mock_auth_manager, mock_response,
        )
        types = [e["type"] for e in events]
        print(f"Event types: {types}")

        assert types[0] == "response.created"
        assert types[-1] == "response.completed"

        deltas = [e["delta"] for e in events if e["type"] == "response.output_text.delta"]
        assert deltas == ["Hello", " world"]

        items = [e["item"] for e in events if e["type"] == "response.output_item.done"]
        assert len(items) == 1
        assert items[0]["type"] == "message"
        assert items[0]["role"] == "assistant"
        assert items[0]["content"][0]["text"] == "Hello world"
        print("✓ Text-only stream has the expected shape and order")

    @pytest.mark.asyncio
    async def test_message_item_opened_before_deltas(self, mock_model_cache, mock_auth_manager, mock_response):
        """
        What it does: output_item.added precedes any text delta, sharing its id.
        Goal: Clients track an "active item" and reject deltas that arrive
              without one ("OutputTextDelta without active item"). The added
              item must also start empty, or its text is counted twice.
        """
        print("Setup: stream with two content events...")
        events = await collect_stream(
            [KiroEvent(type="content", content="Hel"),
             KiroEvent(type="content", content="lo")],
            mock_model_cache, mock_auth_manager, mock_response,
        )
        types = [e["type"] for e in events]
        print(f"Event types: {types}")

        added_index = types.index("response.output_item.added")
        first_delta_index = types.index("response.output_text.delta")
        assert added_index < first_delta_index, "item must be opened before deltas"

        added_item = events[added_index]["item"]
        assert added_item["type"] == "message"
        assert added_item["content"] == [], "opened item must start empty"

        # added and done must describe the same item.
        done_item = [
            e["item"] for e in events
            if e["type"] == "response.output_item.done" and e["item"]["type"] == "message"
        ][0]
        assert added_item["id"] == done_item["id"]

        # Every delta must reference that item id.
        delta_ids = {
            e["item_id"] for e in events if e["type"] == "response.output_text.delta"
        }
        assert delta_ids == {added_item["id"]}
        print("✓ Item opened before deltas, ids consistent, opened empty")

    @pytest.mark.asyncio
    async def test_no_item_added_without_content(self, mock_model_cache, mock_auth_manager, mock_response):
        """
        What it does: No message item is opened when the turn has no text.
        Goal: Avoid opening an item that never receives content.
        """
        print("Setup: tool-call-only stream...")
        events = await collect_stream(
            [KiroEvent(type="tool_use", tool_use={
                "id": "c1", "type": "function",
                "function": {"name": "ls", "arguments": "{}"},
            })],
            mock_model_cache, mock_auth_manager, mock_response,
        )
        types = [e["type"] for e in events]
        print(f"Event types: {types}")

        assert "response.output_item.added" not in types
        print("✓ No item opened for a text-free turn")

    @pytest.mark.asyncio
    async def test_completed_is_terminal_and_unique(self, mock_model_cache, mock_auth_manager, mock_response):
        """
        What it does: response.completed appears exactly once, at the very end.
        Goal: A missing or duplicated terminal event fails the turn client-side.
        """
        print("Setup: Kiro stream with content and usage...")
        events = await collect_stream(
            [KiroEvent(type="content", content="hi"),
             KiroEvent(type="context_usage", context_usage_percentage=10.0),
             KiroEvent(type="usage", usage={"credits": 1})],
            mock_model_cache, mock_auth_manager, mock_response,
        )
        types = [e["type"] for e in events]

        assert types.count("response.completed") == 1, "completed must appear exactly once"
        assert types[-1] == "response.completed", "completed must be the final event"

        usage = events[-1]["response"]["usage"]
        print(f"Usage: {usage}")
        assert usage["output_tokens"] > 0
        # 10% of the mocked 200k context window.
        assert usage["total_tokens"] == 20000
        assert usage["input_tokens"] == usage["total_tokens"] - usage["output_tokens"]
        print("✓ Terminal event is unique and carries usage")

    @pytest.mark.asyncio
    async def test_tool_call_emitted_whole(self, mock_model_cache, mock_auth_manager, mock_response):
        """
        What it does: A tool call arrives complete in one output_item.done.
        Goal: Clients ignore function_call_arguments deltas, so a fragmented
              call would never be seen. This is the critical invariant.
        """
        print("Setup: Kiro stream with a complete tool_use event...")
        tool_use = {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "shell", "arguments": '{"cmd":"ls"}'},
        }
        events = await collect_stream(
            [KiroEvent(type="tool_use", tool_use=tool_use)],
            mock_model_cache, mock_auth_manager, mock_response,
        )
        types = [e["type"] for e in events]
        print(f"Event types: {types}")

        calls = [
            e["item"] for e in events
            if e["type"] == "response.output_item.done" and e["item"]["type"] == "function_call"
        ]
        assert len(calls) == 1, "expected exactly one function_call item"
        call = calls[0]
        assert call["call_id"] == "call_abc"
        assert call["name"] == "shell"
        # Arguments must be a valid JSON *string* - fragmenting would break this.
        assert json.loads(call["arguments"]) == {"cmd": "ls"}
        assert call["id"].startswith("fc_")

        # Negative assertion: emitting these would signal a wrong design.
        assert "response.function_call_arguments.delta" not in types
        assert "response.function_call_arguments.done" not in types
        print("✓ Tool call delivered whole, with no argument-fragment events")

    @pytest.mark.asyncio
    async def test_pure_tool_call_turn_has_no_message_item(self, mock_model_cache, mock_auth_manager, mock_response):
        """
        What it does: No empty assistant message when a turn is only a tool call.
        Goal: Avoid emitting a message item with no text.
        """
        print("Setup: tool_use with no content events...")
        events = await collect_stream(
            [KiroEvent(type="tool_use", tool_use={
                "id": "c1", "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            })],
            mock_model_cache, mock_auth_manager, mock_response,
        )
        items = [e["item"] for e in events if e["type"] == "response.output_item.done"]
        item_types = [i["type"] for i in items]
        print(f"Item types: {item_types}")

        assert "message" not in item_types
        assert item_types == ["function_call"]
        # A pending tool call means the turn is not finished.
        assert events[-1]["response"]["end_turn"] is False
        print("✓ Pure tool-call turn emits no message item")

    @pytest.mark.asyncio
    async def test_thinking_is_not_emitted_as_reasoning(self, mock_model_cache, mock_auth_manager, mock_response):
        """
        What it does: Thinking content produces no reasoning events.
        Goal: Kiro's synthetic thinking has no valid encrypted_content to
              round-trip, so no reasoning items are emitted at all.
        """
        print("Setup: stream with thinking and content events...")
        events = await collect_stream(
            [KiroEvent(type="thinking", thinking_content="pondering"),
             KiroEvent(type="content", content="answer")],
            mock_model_cache, mock_auth_manager, mock_response,
        )
        types = [e["type"] for e in events]
        print(f"Event types: {types}")

        assert not any(t.startswith("response.reasoning") for t in types)
        # Thinking text must not leak into the visible output either.
        deltas = "".join(e["delta"] for e in events if e["type"] == "response.output_text.delta")
        assert deltas == "answer"
        print("✓ No reasoning events, and thinking stays out of visible text")

    @pytest.mark.asyncio
    async def test_dict_arguments_are_serialized(self, mock_model_cache, mock_auth_manager, mock_response):
        """
        What it does: Dict arguments are encoded as a JSON string.
        Goal: Clients parse `arguments` as a string; a raw object breaks them.
        """
        print("Setup: tool_use whose arguments are a dict...")
        events = await collect_stream(
            [KiroEvent(type="tool_use", tool_use={
                "id": "c1", "type": "function",
                "function": {"name": "grep", "arguments": {"pattern": "x"}},
            })],
            mock_model_cache, mock_auth_manager, mock_response,
        )
        call = [
            e["item"] for e in events
            if e["type"] == "response.output_item.done" and e["item"]["type"] == "function_call"
        ][0]
        print(f"Arguments: {call['arguments']!r}")

        assert isinstance(call["arguments"], str)
        assert json.loads(call["arguments"]) == {"pattern": "x"}
        print("✓ Dict arguments serialized to a JSON string")


# ==================================================================================================
# Tests for input conversion
# ==================================================================================================

class TestResponsesInputConversion:
    """Tests for converting Responses input items to unified messages."""

    def test_tool_exchange_round_trip(self):
        """
        What it does: Converts a two-turn tool exchange to unified messages.
        Goal: Verify call_id pairing and JSON-string argument pass-through.
        """
        print("Setup: Codex-shaped input with a tool call and its output...")
        system, messages = convert_responses_input_to_unified([
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "list files"}]},
            {"type": "function_call", "call_id": "c1", "name": "shell",
             "arguments": '{"cmd":"ls"}'},
            {"type": "function_call_output", "call_id": "c1", "output": "a.txt"},
        ], instructions="You are Codex.")

        print(f"System: {system!r}, messages: {len(messages)}")
        assert system == "You are Codex."
        assert [m.role for m in messages] == ["user", "assistant", "user"]

        call = messages[1].tool_calls[0]
        assert call["id"] == "c1"
        assert call["function"]["arguments"] == '{"cmd":"ls"}'

        result = messages[2].tool_results[0]
        assert result["tool_use_id"] == "c1"
        assert result["content"] == "a.txt"
        print("✓ Tool exchange converted with correct pairing")

    def test_reasoning_items_are_dropped(self):
        """
        What it does: reasoning items are skipped.
        Goal: Kiro cannot consume them; they carry no call_id so dropping them
              cannot break function_call/output pairing.
        """
        print("Setup: input containing a reasoning item...")
        _system, messages = convert_responses_input_to_unified([
            {"type": "reasoning", "id": "rs_1", "encrypted_content": "opaque",
             "summary": [{"type": "summary_text", "text": "thinking"}]},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "hi"}]},
        ])
        print(f"Messages: {[(m.role, m.content) for m in messages]}")

        assert len(messages) == 1
        assert messages[0].content == "hi"
        print("✓ Reasoning item dropped, conversation intact")

    def test_unknown_item_types_do_not_raise(self):
        """
        What it does: Unrecognized item types are skipped, not rejected.
        Goal: A newer client must not break the endpoint outright.
        """
        print("Setup: input with an unknown item type...")
        _system, messages = convert_responses_input_to_unified([
            {"type": "local_shell_call", "call_id": "x", "action": {}},
            {"type": "some_future_type", "data": 1},
            {"type": "message", "role": "user", "content": "hello"},
        ])
        assert len(messages) == 1
        assert messages[0].content == "hello"
        print("✓ Unknown item types skipped gracefully")

    def test_instruction_roles_fold_into_system_prompt(self):
        """
        What it does: system/developer messages become part of the system prompt.
        Goal: Keep instructions out of the conversation history.
        """
        print("Setup: input with a developer message...")
        system, messages = convert_responses_input_to_unified([
            {"type": "message", "role": "developer", "content": "Be terse."},
            {"type": "message", "role": "user", "content": "hi"},
        ], instructions="Base.")
        print(f"System: {system!r}")

        assert "Base." in system and "Be terse." in system
        assert [m.role for m in messages] == ["user"]
        print("✓ Instruction roles folded into the system prompt")

    def test_output_as_array_or_string(self):
        """
        What it does: function_call_output.output accepts both wire shapes.
        Goal: The field is untagged - either a string or content items.
        """
        assert extract_output_text("plain") == "plain"
        assert extract_output_text([{"type": "output_text", "text": "boxed"}]) == "boxed"
        assert extract_output_text(None) == ""
        print("✓ Both output shapes handled")

    def test_server_side_tools_are_skipped(self):
        """
        What it does: Provider-executed tool types are not sent to Kiro.
        Goal: Only function tools have a Kiro equivalent.
        """
        tools = convert_responses_tools_to_unified([
            {"type": "function", "name": "shell", "description": "run",
             "parameters": {"type": "object"}},
            {"type": "web_search"},
        ])
        assert len(tools) == 1
        assert tools[0].name == "shell"
        print("✓ Server-side tool types skipped")


# ==================================================================================================
# Tests for the responses-lite wire variant (gpt-5.6-sol and similar)
# ==================================================================================================

class TestResponsesLiteTools:
    """
    Tests for tool definitions carried inside additional_tools input items.

    Some models leave the top-level `tools` field empty and pack definitions
    into an `additional_tools` input item instead. Dropping that item means the
    model is offered no tools at all and correctly reports it cannot act.
    """

    def test_harvests_default_namespace(self):
        """
        What it does: Pulls tools out of the default "functions" namespace.
        Goal: Without this the model receives zero tools.
        """
        print("Setup: additional_tools item with two namespaces...")
        tools = extract_additional_tools([LITE_ADDITIONAL_TOOLS])
        names = [t["name"] for t in tools]
        print(f"Harvested: {names}")

        assert names == ["exec", "wait"]
        # Non-default namespaces need a `namespace` field on the call item that
        # cannot be reconstructed from a tool name, so they are left out.
        assert "spawn_agent" not in names

    def test_top_level_tools_still_work(self):
        """
        What it does: The standard wire format is unaffected.
        Goal: Claude models send top-level tools; that path must not regress.
        """
        req = ResponsesRequest(
            model="claude-sonnet-4.5",
            input=[{"type": "message", "role": "user", "content": "hi"}],
            tools=[{"type": "function", "name": "shell",
                    "parameters": {"type": "object", "properties": {}}}],
        )
        names = [t["name"] for t in collect_request_tools(req)]
        print(f"Collected: {names}")
        assert names == ["shell"]

    def test_merges_both_sources_without_duplicates(self):
        """
        What it does: Merges top-level and additional_tools definitions.
        Goal: A request may use both; top-level wins on a name collision.
        """
        req = ResponsesRequest(
            model="gpt-5.6-sol",
            input=[LITE_ADDITIONAL_TOOLS],
            tools=[{"type": "function", "name": "exec",
                    "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}],
        )
        collected = collect_request_tools(req)
        names = [t["name"] for t in collected]
        print(f"Collected: {names}")

        assert names.count("exec") == 1, "no duplicate definitions"
        assert set(names) == {"exec", "wait"}
        # The top-level definition takes precedence, so exec keeps its schema.
        exec_tool = [t for t in collected if t["name"] == "exec"][0]
        assert exec_tool.get("parameters") is not None

    def test_additional_tools_never_becomes_a_message(self):
        """
        What it does: The additional_tools item is not conversation content.
        Goal: Its role is "developer"; treating it as a message would inject
              tool JSON into the prompt.
        """
        print("Setup: input with additional_tools plus one real message...")
        _system, messages = convert_responses_input_to_unified(
            [LITE_ADDITIONAL_TOOLS,
             {"type": "message", "role": "user", "content": "list files"}]
        )
        print(f"Messages: {[(m.role, m.content) for m in messages]}")

        assert len(messages) == 1
        assert messages[0].content == "list files"

    def test_freeform_tool_gets_a_schema(self):
        """
        What it does: Freeform ("custom") tools receive a synthetic schema.
        Goal: Kiro requires a JSON schema, but freeform tools declare none.
        """
        tools = extract_additional_tools([LITE_ADDITIONAL_TOOLS])
        unified = convert_responses_tools_to_unified(tools)
        by_name = {u.name: u for u in unified}
        print(f"exec schema: {by_name['exec'].input_schema}")

        assert collect_freeform_tool_names(tools) == ["exec"]
        schema = by_name["exec"].input_schema
        assert schema["type"] == "object"
        assert "input" in schema["properties"]
        # A declared schema is passed through untouched.
        assert by_name["wait"].input_schema["properties"]["ms"]["type"] == "number"

    def test_plain_function_without_params_gets_no_fake_schema(self):
        """
        What it does: Only custom tools get the synthetic input schema.
        Goal: A schemaless plain function must not gain a bogus `input` string.
        """
        unified = convert_responses_tools_to_unified(
            [{"type": "function", "name": "ping", "description": "no params"}]
        )
        print(f"ping schema: {unified[0].input_schema}")
        assert unified[0].input_schema is None

    @pytest.mark.asyncio
    async def test_freeform_call_returns_custom_tool_call(
        self, mock_model_cache, mock_auth_manager, mock_response
    ):
        """
        What it does: A freeform tool call comes back as custom_tool_call.
        Goal: The client declared `exec` as custom, so it expects raw `input`,
              not a JSON `arguments` string. The synthetic wrapper key is
              unwrapped on the way out.
        """
        print("Setup: Kiro returns exec with the synthetic input key...")
        events = await collect_stream(
            [KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "exec", "arguments": '{"input":"ls -la"}'},
            })],
            mock_model_cache, mock_auth_manager, mock_response,
            freeform_tool_names={"exec"},
        )
        items = [e["item"] for e in events if e["type"] == "response.output_item.done"]
        print(f"Items: {items}")

        assert len(items) == 1
        item = items[0]
        assert item["type"] == "custom_tool_call"
        assert item["name"] == "exec"
        assert item["call_id"] == "call_1"
        # Raw text, not a JSON envelope.
        assert item["input"] == "ls -la"
        assert "arguments" not in item

    @pytest.mark.asyncio
    async def test_non_freeform_call_still_function_call(
        self, mock_model_cache, mock_auth_manager, mock_response
    ):
        """
        What it does: Tools not declared freeform keep the function_call shape.
        Goal: Only freeform tools change representation.
        """
        events = await collect_stream(
            [KiroEvent(type="tool_use", tool_use={
                "id": "call_2", "type": "function",
                "function": {"name": "wait", "arguments": '{"ms":100}'},
            })],
            mock_model_cache, mock_auth_manager, mock_response,
            freeform_tool_names={"exec"},
        )
        item = [e["item"] for e in events if e["type"] == "response.output_item.done"][0]
        print(f"Item: {item}")

        assert item["type"] == "function_call"
        assert json.loads(item["arguments"]) == {"ms": 100}

    def test_custom_tool_call_input_becomes_valid_json(self):
        """
        What it does: A returned custom_tool_call's raw input is re-wrapped as JSON.
        Goal: The core layer runs json.loads on tool arguments
              (converters_core.py:803). Raw shell text there aborts the whole
              turn with "Expecting value: line 1 column 1".
        """
        raw = "ls\nfind . -maxdepth 1 -name '*.txt'"
        print(f"Setup: custom_tool_call carrying raw text {raw[:20]!r}...")
        _system, messages = convert_responses_input_to_unified([
            {"type": "custom_tool_call", "call_id": "c1", "name": "exec", "input": raw},
        ])
        args = messages[0].tool_calls[0]["function"]["arguments"]
        print(f"arguments: {args[:60]}")

        parsed = json.loads(args)  # must not raise
        assert parsed["input"] == raw, "raw text must survive the round trip"

    def test_function_call_json_arguments_unchanged(self):
        """
        What it does: Plain function_call arguments pass through untouched.
        Goal: Guard against the re-wrapping leaking into the standard path.
        """
        _system, messages = convert_responses_input_to_unified([
            {"type": "function_call", "call_id": "c2", "name": "wait",
             "arguments": '{"ms":50}'},
        ])
        args = messages[0].tool_calls[0]["function"]["arguments"]
        print(f"arguments: {args}")
        assert json.loads(args) == {"ms": 50}
        assert "input" not in json.loads(args)

    def test_non_json_arguments_are_wrapped_not_fatal(self):
        """
        What it does: Non-JSON arguments on a plain function_call are wrapped.
        Goal: One malformed tool call must not fail the entire conversion.
        """
        _system, messages = convert_responses_input_to_unified([
            {"type": "function_call", "call_id": "c3", "name": "odd",
             "arguments": "not json at all"},
        ])
        args = messages[0].tool_calls[0]["function"]["arguments"]
        print(f"wrapped: {args}")
        assert json.loads(args) == {"input": "not json at all"}

    def test_empty_custom_input_still_valid_json(self):
        """
        What it does: An empty freeform input still produces parseable JSON.
        Goal: Avoid emitting "" where the core layer expects JSON.
        """
        _system, messages = convert_responses_input_to_unified([
            {"type": "custom_tool_call", "call_id": "c4", "name": "exec", "input": ""},
        ])
        args = messages[0].tool_calls[0]["function"]["arguments"]
        print(f"arguments: {args}")
        assert json.loads(args) == {"input": ""}

    @pytest.mark.asyncio
    async def test_freeform_call_tolerates_missing_wrapper_key(
        self, mock_model_cache, mock_auth_manager, mock_response
    ):
        """
        What it does: Handles a model that ignores the synthetic key.
        Goal: A single string value is still recovered as raw input.
        """
        events = await collect_stream(
            [KiroEvent(type="tool_use", tool_use={
                "id": "c", "type": "function",
                "function": {"name": "exec", "arguments": '{"command":"pwd"}'},
            })],
            mock_model_cache, mock_auth_manager, mock_response,
            freeform_tool_names={"exec"},
        )
        item = [e["item"] for e in events if e["type"] == "response.output_item.done"][0]
        print(f"Recovered input: {item['input']!r}")
        assert item["input"] == "pwd"


# ==================================================================================================
# Tests for the endpoint
# ==================================================================================================

class TestResponsesEndpoint:
    """Tests for POST /v1/responses behaviour."""

    def test_requires_authentication(self, test_client):
        """
        What it does: Rejects requests with no bearer token.
        Goal: The endpoint must not be reachable unauthenticated.
        """
        print("Action: POST /v1/responses without auth...")
        response = test_client.post("/v1/responses", json={
            "model": "claude-sonnet-4-5",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
        })
        print(f"Status: {response.status_code}")
        assert response.status_code == 401

    def test_rejects_invalid_key(self, test_client):
        """
        What it does: Rejects an incorrect bearer token.
        Goal: Verify the key is actually checked.
        """
        print("Action: POST /v1/responses with a wrong key...")
        response = test_client.post(
            "/v1/responses",
            headers={"Authorization": "Bearer wrong_key_123"},
            json={"model": "claude-sonnet-4-5", "input": "hi"},
        )
        print(f"Status: {response.status_code}")
        assert response.status_code == 401

    def test_rejects_non_streaming(self, test_client, valid_proxy_api_key):
        """
        What it does: Rejects stream=false with a clear error.
        Goal: The endpoint is streaming-only by design.
        """
        print("Action: POST /v1/responses with stream=false...")
        response = test_client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
            json={"model": "claude-sonnet-4-5", "input": "hi", "stream": False},
        )
        print(f"Status: {response.status_code}, body: {response.text[:120]}")

        assert response.status_code == 400
        assert response.json()["error"]["type"] == "invalid_request_error"
        print("✓ Non-streaming requests rejected")
