from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from duo_workflow_service.entities.state import ReplacementRule, SearchAndReplaceConfig
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


def test_default_with_search_and_replace_config():
    encoder = CustomEncoder()
    o = SearchAndReplaceConfig(
        file_types=["*.vue"],
        domain_speciality="accessibility expert",
        assignment_description="accessibility issues",
        replacement_rules=[
            ReplacementRule(element="gl-icon", rules="Add aria-label"),
            ReplacementRule(element="gl-avatar", rules="Add alt text"),
        ],
    )

    encoded_config = encoder.default(o)
    assert encoded_config == {
        "assignment_description": "accessibility issues",
        "domain_speciality": "accessibility expert",
        "file_types": [
            "*.vue",
        ],
        "replacement_rules": [
            {
                "element": "gl-icon",
                "rules": "Add aria-label",
            },
            {
                "element": "gl-avatar",
                "rules": "Add alt text",
            },
        ],
        "type": "SearchAndReplaceConfig",
    }
