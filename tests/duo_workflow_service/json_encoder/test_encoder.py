import base64
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Send

from duo_workflow_service.entities.state import (
    AdditionalContext,
    ApprovalStateRejection,
)
from duo_workflow_service.json_encoder.encoder import CustomEncoder

PNG_B64 = base64.b64encode(b"\x89PNG\r\n\x1a\npixels").decode()


def test_default_with_system_message():
    encoder = CustomEncoder()

    system_message = SystemMessage(content="This is a system message")
    encoded_system = encoder.default(system_message)
    assert encoded_system == {
        "type": "SystemMessage",
        "content": "This is a system message",
        "additional_kwargs": {},
        "response_metadata": {},
        "name": None,
        "id": None,
    }


def test_default_with_human_message():
    encoder = CustomEncoder()

    human_message = HumanMessage(content="This is a human message")
    encoded_human = encoder.default(human_message)
    assert encoded_human == {
        "type": "HumanMessage",
        "content": "This is a human message",
        "additional_kwargs": {},
        "response_metadata": {},
        "name": None,
        "id": None,
    }


def test_default_with_ai_message():
    encoder = CustomEncoder()

    ai_message = AIMessage(content="This is an AI message")
    encoded_ai = encoder.default(ai_message)
    assert encoded_ai == {
        "type": "AIMessage",
        "content": "This is an AI message",
        "additional_kwargs": {},
        "response_metadata": {},
        "invalid_tool_calls": [],
        "usage_metadata": None,
        "tool_calls": [],
        "name": None,
        "id": None,
    }


def test_default_with_tool_message():
    encoder = CustomEncoder()

    tool_message = ToolMessage(content="This is a tool message", tool_call_id="call id")
    encoded_tool = encoder.default(tool_message)
    assert encoded_tool == {
        "type": "ToolMessage",
        "content": "This is a tool message",
        "additional_kwargs": {},
        "response_metadata": {},
        "artifact": None,
        "status": "success",
        "tool_call_id": "call id",
        "name": None,
        "id": None,
    }


def test_default_with_approval_state_rejection():
    encoder = CustomEncoder()
    o = ApprovalStateRejection(message="Cancel this tool")

    encoded_approval_state = encoder.default(o)
    assert encoded_approval_state == {
        "message": "Cancel this tool",
        "type": "ApprovalStateRejection",
    }


def test_default_with_send():
    encoder = CustomEncoder()
    send = Send("developer", {"goal": "Fix the bug"})

    encoded_send = encoder.default(send)
    assert encoded_send == {
        "type": "Send",
        "node": "developer",
        "arg": {"goal": "Fix the bug"},
    }


def test_default_with_additional_context():
    encoder = CustomEncoder()
    additional_context = AdditionalContext(
        category="merge_request",
        id="12345",
        content="This is the merge request content",
        metadata={
            "url": "https://gitlab.com/repo/merge_requests/12345",
            "state": "open",
        },
    )
    encoded_context = encoder.default(additional_context)
    assert encoded_context == {
        "type": "AdditionalContext",
        "category": "merge_request",
        "id": "12345",
        "content": "This is the merge request content",
        "metadata": {
            "url": "https://gitlab.com/repo/merge_requests/12345",
            "state": "open",
        },
    }


def image_content(text: str) -> list:
    return [
        {"type": "text", "text": text},
        {"type": "image", "base64": PNG_B64, "mime_type": "image/png"},
    ]


@pytest.mark.parametrize(
    "message",
    [
        HumanMessage(content=image_content("what is this?")),
        AIMessage(content=image_content("here is the diagram")),
        ToolMessage(content=image_content("Contents of a.png:"), tool_call_id="1"),
        SystemMessage(content=image_content("reference:")),
    ],
    ids=["human", "ai", "tool", "system"],
)
def test_image_blocks_are_stripped_whatever_the_message_role(message):
    """An image can arrive as a user attachment or as a tool result, so the stripping keys on content shape rather than
    on the role."""
    encoded = CustomEncoder().default(message)

    assert encoded["content"][0]["type"] == "text"
    assert encoded["content"][1] == {
        "type": "text",
        "text": "[image/png omitted from history]",
    }
    assert PNG_B64 not in json.dumps(encoded)


def test_the_placeholder_does_not_name_a_producer():
    """A tool-carried image and an attached image get the same placeholder."""
    encoded = CustomEncoder().default(
        ToolMessage(content=image_content("Contents of a.png:"), tool_call_id="1")
    )

    assert "attachment" not in json.dumps(encoded)


def test_checkpoint_serialization_excludes_image_payloads_end_to_end():
    """The whole point of the stripping: base64 must never reach a checkpoint."""
    state = {
        "conversation_history": {
            "agent": [
                HumanMessage(content=image_content("look at this")),
                ToolMessage(content=image_content("Contents:"), tool_call_id="1"),
            ]
        },
    }

    serialized = json.dumps(state, cls=CustomEncoder)

    assert PNG_B64 not in serialized


def test_string_content_messages_are_unaffected():
    encoder = CustomEncoder()

    encoded = encoder.default(HumanMessage(content="plain text"))

    assert encoded["content"] == "plain text"


def test_an_interrupt_mid_turn_loses_the_image_but_keeps_its_label():
    """Pins a known v1 limitation so that changing it is a deliberate act.

    Stripping happens on every checkpoint write, and a resume rebuilds history from the
    written checkpoint, so an image does not survive its own turn: one tool-call approval
    is enough to lose it. The filename block is what keeps the turn intelligible -- it is
    not an image block, so it survives, and the model can say which file it can no longer
    see instead of guessing.

    If this test starts failing because the payload survives, the durable fix has landed
    and the docstrings in `image_blocks` and `encoder` need updating with it.
    """
    message = HumanMessage(
        content=[
            {"type": "text", "text": "[attached file: screenshot.png]"},
            {"type": "image", "base64": PNG_B64, "mime_type": "image/png"},
            {"type": "text", "text": "fix the bug it shows"},
        ]
    )

    # The write half of an interrupt: the graph checkpoints before waiting for the
    # approval.
    persisted = json.dumps(
        {"conversation_history": {"agent": [message]}}, cls=CustomEncoder
    )

    # The read half: a resume reloads history from exactly those bytes.
    reloaded = json.loads(persisted)
    content = reloaded["conversation_history"]["agent"][0]["content"]

    assert PNG_B64 not in persisted
    assert not [block for block in content if block.get("base64")]
    assert {"type": "text", "text": "[attached file: screenshot.png]"} in content
    assert {"type": "text", "text": "fix the bug it shows"} in content
    assert any("omitted from history" in block.get("text", "") for block in content)
