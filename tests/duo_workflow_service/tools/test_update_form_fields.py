import json

import pytest
from pydantic import ValidationError

from duo_workflow_service.tools.update_form_fields import (
    UpdateFormFields,
    UpdateFormFieldsInput,
)


@pytest.mark.asyncio
async def test_execute_with_select():
    tool = UpdateFormFields(metadata={})

    response = await tool.arun(
        {
            "form_id": "ask-duo-pat",
            "select": ["read_project", "write_project"],
            "clear": [],
        }
    )

    parsed = json.loads(response)
    assert parsed["form_id"] == "ask-duo-pat"
    assert parsed["select"] == ["read_project", "write_project"]
    assert parsed["clear"] == []


@pytest.mark.asyncio
async def test_execute_with_clear():
    tool = UpdateFormFields(metadata={})

    response = await tool.arun(
        {"form_id": "ask-duo-pat", "select": [], "clear": ["read_project"]}
    )

    parsed = json.loads(response)
    assert parsed["form_id"] == "ask-duo-pat"
    assert parsed["select"] == []
    assert parsed["clear"] == ["read_project"]


@pytest.mark.asyncio
async def test_execute_with_none_defaults():
    tool = UpdateFormFields(metadata={})

    response = await tool.arun({"form_id": "ci-variable-editor"})

    parsed = json.loads(response)
    assert parsed["form_id"] == "ci-variable-editor"
    assert parsed["select"] == []
    assert parsed["clear"] == []


def test_format_display_message_add():
    tool = UpdateFormFields(metadata={})
    args = UpdateFormFieldsInput(
        form_id="ask-duo-pat",
        select=["read_project", "write_project"],
    )

    message = tool.format_display_message(args)

    assert message == "Update form fields — Select: read_project, write_project"


def test_format_display_message_remove():
    tool = UpdateFormFields(metadata={})
    args = UpdateFormFieldsInput(form_id="ask-duo-pat", clear=["read_project"])

    message = tool.format_display_message(args)

    assert message == "Update form fields — Clear: read_project"


def test_format_display_message_add_and_remove():
    tool = UpdateFormFields(metadata={})
    args = UpdateFormFieldsInput(
        form_id="ask-duo-pat",
        select=["write_project"],
        clear=["read_project"],
    )

    message = tool.format_display_message(args)

    assert message == "Update form fields — Select: write_project; Clear: read_project"


def test_format_display_message_empty():
    tool = UpdateFormFields(metadata={})
    args = UpdateFormFieldsInput(form_id="ask-duo-pat")

    message = tool.format_display_message(args)

    assert message == "No form field changes"


def test_tool_properties():
    tool = UpdateFormFields(metadata={})

    assert tool.name == "update_form_fields"
    assert "GitLab UI" in tool.description
    assert tool.args_schema == UpdateFormFieldsInput


def test_form_id_required_in_llm_schema():
    """form_id is a required LLM-supplied argument — the system prompt tells the model what value to use."""
    schema = UpdateFormFields(metadata={}).get_input_schema().model_json_schema()
    assert "form_id" in schema.get("required", [])


def test_input_schema_rejects_empty_form_id():
    """An empty form_id is rejected at the schema boundary.

    The flow YAML marks form_context as optional, so a missing envelope would
    leave the system prompt rendering ``Always set form_id to ""``. Guard
    against the LLM faithfully echoing that back.
    """
    with pytest.raises(ValidationError):
        UpdateFormFieldsInput(form_id="", select=["read_project"], clear=[])
