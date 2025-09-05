from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from duo_workflow_service.entities.state import (
    AdditionalContext,
    ApprovalStateRejection,
)
from duo_workflow_service.json_encoder.encoder import CustomEncoder


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
        "example": False,
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
        "example": False,
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
