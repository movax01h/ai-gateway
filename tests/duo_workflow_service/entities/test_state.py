from typing import Dict, List, Optional

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from contract import contract_pb2
from duo_workflow_service.entities.state import (
    TOOL_RESPONSE_MAX_DISPLAY_MSG,
    ApprovalSource,
    MessageTypeEnum,
    ToolInfo,
    UiChatLog,
    _conversation_history_reducer,
    _ui_chat_log_reducer,
    build_tool_info,
    policy_ref_to_log_dict,
    render_for_display,
)
from tests.duo_workflow_service.ui_chat_log_contract import (
    assert_client_valid_tool_info,
)


class TestApprovalSourceFromProto:
    """Documents the mapping between the proto enum and the StrEnum."""

    def test_every_proto_value_maps_to_a_loggable_string(self):
        for name, value in contract_pb2.Approval.ApprovalSource.items():
            mapped = ApprovalSource.from_proto(value)
            short = name.removeprefix("APPROVAL_SOURCE_")
            if short == "UNSPECIFIED":
                assert mapped == "unspecified"
            else:
                assert mapped == ApprovalSource[short].value

    def test_every_proto_value_is_prefixed(self):
        for name in contract_pb2.Approval.ApprovalSource.keys():
            assert name.startswith("APPROVAL_SOURCE_"), (
                f"{name} lacks the APPROVAL_SOURCE_ prefix; proto enum values "
                "share the enclosing message's namespace, so keep them prefixed."
            )

    def test_every_member_has_a_proto_counterpart(self):
        proto_names = {
            name.removeprefix("APPROVAL_SOURCE_")
            for name in contract_pb2.Approval.ApprovalSource.keys()
        }
        for member in ApprovalSource:
            assert member.name in proto_names, (
                f"{member.name} has no APPROVAL_SOURCE_{member.name} counterpart in the proto enum; "
                "add it to contract.proto and regenerate the stubs."
            )

    def test_unknown_value_does_not_raise(self):
        assert ApprovalSource.from_proto(99) == "unknown(99)"


class TestApprovalSourceFromApproval:
    """Documents ApprovalSource.from_approval."""

    def test_uses_client_reported_source_when_present(self):
        approved = contract_pb2.Approval.Approved(
            approval_source=contract_pb2.Approval.ApprovalSource.APPROVAL_SOURCE_AUTO_MODE
        )
        assert ApprovalSource.from_approval(approved) == ApprovalSource.AUTO_MODE.value

    def test_defaults_to_none_when_source_unset(self):
        # An Approved message with no approval_source field (never sent) resolves
        # to None, matching the call site in abstract_workflow.py.
        assert ApprovalSource.from_approval(contract_pb2.Approval.Approved()) is None

    def test_none_defaults_to_none(self):
        assert ApprovalSource.from_approval(None) is None

    def test_explicit_unspecified_is_distinct_from_never_sent(self):
        # Proto3 optional presence: explicitly stamping APPROVAL_SOURCE_UNSPECIFIED
        # (value 0) sets the field, so from_approval resolves it to "unspecified"
        # rather than the None returned for the never-sent case above.
        approved = contract_pb2.Approval.Approved(
            approval_source=contract_pb2.Approval.ApprovalSource.APPROVAL_SOURCE_UNSPECIFIED
        )
        assert ApprovalSource.from_approval(approved) == "unspecified"

    def test_client_stamped_session_approval_passes_through_unverified(self):
        # SESSION_APPROVAL is documented as server-produced only, but there is no
        # server-side verification today: a client that stamps
        # APPROVAL_SOURCE_SESSION_APPROVAL has it passed through as-is. This test
        # pins that current (unverified) behavior so a future clamp is visible.
        approved = contract_pb2.Approval.Approved(
            approval_source=contract_pb2.Approval.ApprovalSource.APPROVAL_SOURCE_SESSION_APPROVAL
        )
        assert (
            ApprovalSource.from_approval(approved)
            == ApprovalSource.SESSION_APPROVAL.value
        )


class TestPolicyRefToLogDict:
    """Documents the PolicyRef -> log dict conversion.

    Like ApprovalSource, PolicyRef values are client-provided and informational only; the helper is presence-aware so
    logs distinguish "client set this field" from proto3 empty-string defaults.
    """

    def test_all_fields_set(self):
        policy_ref = contract_pb2.Approval.PolicyRef(
            origin="gitlab_default",
            file=".gitlab/duo/pretooluse.rego",
            hash="sha256:deadbeef",
            version="1.2.3",
        )
        assert policy_ref_to_log_dict(policy_ref) == {
            "origin": "gitlab_default",
            "file": ".gitlab/duo/pretooluse.rego",
            "hash": "sha256:deadbeef",
            "version": "1.2.3",
        }

    def test_partial_fields_only_includes_set_fields(self):
        policy_ref = contract_pb2.Approval.PolicyRef(
            origin="customer_policy",
            file="policies/custom.rego",
        )
        assert policy_ref_to_log_dict(policy_ref) == {
            "origin": "customer_policy",
            "file": "policies/custom.rego",
        }

    def test_empty_message_maps_to_empty_dict(self):
        # A present-but-empty PolicyRef maps to {}, keeping "client sent an
        # empty policy_ref" distinguishable from "client never sent one"
        # (callers log None for the absent-submessage case).
        assert policy_ref_to_log_dict(contract_pb2.Approval.PolicyRef()) == {}

    def test_explicitly_set_empty_string_is_kept_but_unset_fields_are_omitted(self):
        # Proto3 optional tracks presence: an explicitly-set empty string is
        # emitted, while unset fields never surface as "".
        policy_ref = contract_pb2.Approval.PolicyRef(origin="")
        result = policy_ref_to_log_dict(policy_ref)
        assert result == {"origin": ""}
        for unset in ("file", "hash", "version"):
            assert unset not in result


class TestConversationHistoryReducer:
    def test_appends_new_messages_to_existing(self):
        current = {"agent1": [HumanMessage(content="hello")]}
        new = {"agent1": [AIMessage(content="hi there")]}

        result = _conversation_history_reducer(current, new)

        assert len(result["agent1"]) == 2
        assert result["agent1"][0].content == "hello"
        assert result["agent1"][1].content == "hi there"

    def test_returns_copy_when_new_is_none(self):
        current = {"agent1": [HumanMessage(content="hello")]}

        result = _conversation_history_reducer(current, None)

        assert result == current
        assert result is not current

    def test_adds_new_agent_key(self):
        current = {"agent1": [HumanMessage(content="hello")]}
        new = {"agent2": [HumanMessage(content="world")]}

        result = _conversation_history_reducer(current, new)

        assert "agent1" in result
        assert "agent2" in result
        assert len(result["agent2"]) == 1

    def test_skips_empty_new_messages(self):
        current = {"agent1": [HumanMessage(content="hello")]}
        new = {"agent1": []}

        result = _conversation_history_reducer(current, new)

        assert len(result["agent1"]) == 1

    def test_handles_empty_current(self):
        current: Dict[str, List[BaseMessage]] = {}
        new = {"agent1": [HumanMessage(content="hello")]}

        result = _conversation_history_reducer(current, new)

        assert len(result["agent1"]) == 1
        assert result["agent1"][0].content == "hello"

    def test_does_not_mutate_current(self):
        original_messages = [HumanMessage(content="hello")]
        current = {"agent1": original_messages}
        new = {"agent1": [AIMessage(content="hi")]}

        result = _conversation_history_reducer(current, new)

        # Original current dict should not be mutated
        assert len(current["agent1"]) == 1
        assert len(result["agent1"]) == 2

    def test_handles_multiple_agents(self):
        current = {
            "planner": [HumanMessage(content="plan this")],
            "executor": [HumanMessage(content="execute this")],
        }
        new = {
            "planner": [AIMessage(content="here's the plan")],
            "executor": [AIMessage(content="done executing")],
        }

        result = _conversation_history_reducer(current, new)

        assert len(result["planner"]) == 2
        assert len(result["executor"]) == 2

    def test_does_not_trim_messages(self):
        """Verify the reducer appends without trimming.

        Token-based trimming is deferred to agent run time via the history-optimizer pipeline, so the reducer should
        never discard messages.
        """
        # Build a large history that would have triggered the old trim logic
        large_history = [HumanMessage(content=f"message {i}" * 500) for i in range(100)]
        current = {"agent1": large_history}
        new = {"agent1": [AIMessage(content="new response")]}

        result = _conversation_history_reducer(current, new)

        assert len(result["agent1"]) == 101

    def test_idempotency_returns_new_object_each_call(self):
        current = {"agent1": [HumanMessage(content="hello")]}
        new = {"agent1": [AIMessage(content="hi")]}

        result1 = _conversation_history_reducer(current, new)
        result2 = _conversation_history_reducer(current, new)

        assert result1 is not result2
        assert result1 == result2


def test_ui_chat_log_reducer():
    current: List[UiChatLog] = [
        {
            "message_type": MessageTypeEnum.AGENT,
            "message_sub_type": None,
            "content": "first message",
            "timestamp": "2024-01-01T10:00:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        },
        {
            "message_type": MessageTypeEnum.TOOL,
            "message_sub_type": None,
            "content": "second message",
            "timestamp": "2024-01-01T10:01:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": {"name": "read_file", "args": {"file_path": "a/b/c.py"}},
            "additional_context": None,
            "message_id": None,
        },
    ]

    new: Optional[List[UiChatLog]] = [
        {
            "message_type": MessageTypeEnum.USER,
            "message_sub_type": None,
            "content": "third message",
            "timestamp": "2024-01-01T10:02:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        }
    ]

    result = _ui_chat_log_reducer(current, new)

    assert result == [
        {
            "message_type": MessageTypeEnum.AGENT,
            "message_sub_type": None,
            "content": "first message",
            "timestamp": "2024-01-01T10:00:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        },
        {
            "message_type": MessageTypeEnum.TOOL,
            "message_sub_type": None,
            "content": "second message",
            "timestamp": "2024-01-01T10:01:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": {"name": "read_file", "args": {"file_path": "a/b/c.py"}},
            "additional_context": None,
            "message_id": None,
        },
        {
            "message_type": MessageTypeEnum.USER,
            "message_sub_type": None,
            "content": "third message",
            "timestamp": "2024-01-01T10:02:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        },
    ]


def test_ui_chat_log_reducer_idempotency():
    current: List[UiChatLog] = [
        {
            "message_type": MessageTypeEnum.AGENT,
            "message_sub_type": None,
            "content": "first message",
            "timestamp": "2024-01-01T10:00:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        }
    ]

    new: Optional[List[UiChatLog]] = [
        {
            "message_type": MessageTypeEnum.USER,
            "message_sub_type": None,
            "content": "second message",
            "timestamp": "2024-01-01T10:01:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        }
    ]

    _ui_chat_log_reducer(current, new)
    result = _ui_chat_log_reducer(current, new)

    assert result == [
        {
            "message_type": MessageTypeEnum.AGENT,
            "message_sub_type": None,
            "content": "first message",
            "timestamp": "2024-01-01T10:00:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        },
        {
            "message_type": MessageTypeEnum.USER,
            "message_sub_type": None,
            "content": "second message",
            "timestamp": "2024-01-01T10:01:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        },
    ]


def test_ui_chat_log_reducer_with_none():
    current: List[UiChatLog] = [
        {
            "message_type": MessageTypeEnum.AGENT,
            "message_sub_type": None,
            "content": "first message",
            "timestamp": "2024-01-01T10:00:00Z",
            "status": None,
            "correlation_id": None,
            "tool_info": None,
            "additional_context": None,
            "message_id": None,
        }
    ]

    new: Optional[List[UiChatLog]] = None

    result = _ui_chat_log_reducer(current, new)

    assert result == current
    assert result is not current


def test_ui_chat_log_reducer_with_empty_lists():
    current: List[UiChatLog] = []
    new: Optional[List[UiChatLog]] = []

    result = _ui_chat_log_reducer(current, new)

    assert not result
    assert result is not current


def test_build_tool_info_without_response():
    result = build_tool_info("my_tool", {"arg1": "val1"})

    assert result == ToolInfo(name="my_tool", args={"arg1": "val1"})
    assert "tool_response" not in result
    assert_client_valid_tool_info(result)


def test_build_tool_info_with_short_response():
    result = build_tool_info("my_tool", {"arg1": "val1"}, tool_response="short output")

    assert result["tool_response"] == "short output"
    assert_client_valid_tool_info(result)


def test_build_tool_info_truncates_long_string_response():
    long_response = "x" * (TOOL_RESPONSE_MAX_DISPLAY_MSG + 100)

    result = build_tool_info("my_tool", {}, tool_response=long_response)

    assert result["tool_response"].startswith("x" * TOOL_RESPONSE_MAX_DISPLAY_MSG)
    assert result["tool_response"].endswith(
        f"[... display truncated: {TOOL_RESPONSE_MAX_DISPLAY_MSG:,} of "
        f"{TOOL_RESPONSE_MAX_DISPLAY_MSG + 100:,} characters shown ...]"
    )
    assert_client_valid_tool_info(result)


class TestRenderForDisplay:
    """Render, redact, cap, in that order.

    Every card builder calls this once, so this is where the order and the cap are pinned.
    """

    def test_short_text_is_not_capped(self):
        text = "x" * TOOL_RESPONSE_MAX_DISPLAY_MSG

        assert render_for_display(text, tool_name="t") == text

    def test_long_text_keeps_the_head_and_says_so(self):
        text = "a" * TOOL_RESPONSE_MAX_DISPLAY_MSG + "b" * 500

        result = render_for_display(text, tool_name="t")

        assert result.startswith("a" * TOOL_RESPONSE_MAX_DISPLAY_MSG)
        assert "b" not in result.split("\n[... display truncated")[0]
        assert result.endswith(
            f"[... display truncated: {TOOL_RESPONSE_MAX_DISPLAY_MSG:,} of "
            f"{TOOL_RESPONSE_MAX_DISPLAY_MSG + 500:,} characters shown ...]"
        )

    def test_a_block_list_comes_out_as_one_redacted_capped_string(self):
        token = "glpat-AAAAABBBBCCCCDDDDEEEE"
        blocks = [
            {"type": "text", "text": f"token {token}"},
            {"type": "image", "base64": "A" * 4096, "mime_type": "image/png"},
            {"type": "text", "text": "y" * TOOL_RESPONSE_MAX_DISPLAY_MSG},
        ]

        result = render_for_display(blocks, tool_name="read_file")

        assert result.startswith(
            "token [REDACTED]\n[image/png omitted from history]\nyyyy"
        )
        assert token not in result
        assert "A" * 64 not in result
        assert result.endswith("characters shown ...]")

    def test_the_cap_measures_the_redacted_text(self):
        """Redaction runs before the cap, so the marker counts what the client sees, not what the tool returned."""
        token = "glpat-AAAAABBBBCCCCDDDDEEEE"
        text = f"{token} " + "x" * TOOL_RESPONSE_MAX_DISPLAY_MSG

        result = render_for_display(text, tool_name="read_file")

        redacted_len = len("[REDACTED] ") + TOOL_RESPONSE_MAX_DISPLAY_MSG
        assert result.startswith("[REDACTED] xxxx")
        assert result.endswith(
            f"[... display truncated: {TOOL_RESPONSE_MAX_DISPLAY_MSG:,} of "
            f"{redacted_len:,} characters shown ...]"
        )


def test_contract_mirror_rejects_a_non_array_suggested_patterns():
    """The mirror must not be looser than the client, or it gives false safety.

    `all(isinstance(p, str) for p in value)` passes for a dict (its keys are strings) and for a bare string (its
    characters are), where the client's schema requires an array of strings.
    """
    info = ToolInfo(name="t", args={}, suggested_patterns={"git *": 1})

    with pytest.raises(AssertionError, match="must be an array"):
        assert_client_valid_tool_info(info)


class TestBuildToolInfoStructuredContent:
    """Every structured tool_response is rendered as one string.

    The CLI and the IDE validate `tool_response` as a string (or a message whose `content` is one) and discard the
    whole checkpoint's chat log on the first entry that is neither. That is true whether or not an image is involved,
    so the collapse is not gated on one.
    """

    def test_image_content_becomes_string_without_payload(self):
        payload = "B" * 8192
        tool_response = [
            {"type": "text", "text": "Read image file: ./shot.png (image/png, 6 KB)."},
            {"type": "image", "base64": payload, "mime_type": "image/png"},
        ]

        info = build_tool_info("read_file", {"file_path": "./shot.png"}, tool_response)

        assert info["tool_response"] == (
            "Read image file: ./shot.png (image/png, 6 KB).\n"
            "[image/png omitted from history]"
        )
        assert payload not in info["tool_response"]
        assert_client_valid_tool_info(info)

    def test_text_only_list_becomes_string(self):
        # No image, ~30 bytes, and it still drops the chat log on the client
        # if it arrives as a list.
        info = build_tool_info("read_file", {}, [{"type": "text", "text": "x"}])

        assert info["tool_response"] == "x"
        assert_client_valid_tool_info(info)

    def test_dict_response_becomes_string(self):
        # Previously pinned as passing through untouched; a dict is not in the
        # client's union either.
        info = build_tool_info("my_tool", {}, {"key": "value", "data": [1, 2, 3]})

        assert isinstance(info["tool_response"], str)
        assert_client_valid_tool_info(info)

    def test_flattened_list_is_still_capped(self):
        # The cap used to be string-only, so list content was never capped.
        blocks = [{"type": "text", "text": "y" * TOOL_RESPONSE_MAX_DISPLAY_MSG}] * 2

        info = build_tool_info("my_tool", {}, blocks)

        assert info["tool_response"].startswith("y" * TOOL_RESPONSE_MAX_DISPLAY_MSG)
        assert "[... display truncated:" in info["tool_response"]
        assert_client_valid_tool_info(info)

    def test_flattened_list_is_still_redacted(self):
        token = "glpat-AAAAABBBBCCCCDDDDEEEE"

        info = build_tool_info(
            "my_tool", {}, [{"type": "text", "text": f"token {token}"}]
        )

        assert token not in info["tool_response"]
        assert "[REDACTED]" in info["tool_response"]

    def test_string_responses_unchanged(self):
        info = build_tool_info("read_file", {}, "ordinary text output")

        assert info["tool_response"] == "ordinary text output"
        assert_client_valid_tool_info(info)
