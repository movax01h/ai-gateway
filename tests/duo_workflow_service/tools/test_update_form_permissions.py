import json

import pytest

from duo_workflow_service.tools.update_form_permissions import (
    UpdateFormPermissions,
    UpdateFormPermissionsInput,
)


@pytest.mark.asyncio
async def test_execute_with_select():
    tool = UpdateFormPermissions(metadata={})

    response = await tool.arun(
        {"select": ["read_project", "write_project"], "clear": []}
    )

    parsed = json.loads(response)
    assert parsed["select"] == ["read_project", "write_project"]
    assert parsed["clear"] == []


@pytest.mark.asyncio
async def test_execute_with_clear():
    tool = UpdateFormPermissions(metadata={})

    response = await tool.arun({"select": [], "clear": ["read_project"]})

    parsed = json.loads(response)
    assert parsed["select"] == []
    assert parsed["clear"] == ["read_project"]


@pytest.mark.asyncio
async def test_execute_with_none_defaults():
    tool = UpdateFormPermissions(metadata={})

    response = await tool.arun({})

    parsed = json.loads(response)
    assert parsed["select"] == []
    assert parsed["clear"] == []


def test_format_display_message_add():
    tool = UpdateFormPermissions(metadata={})
    args = UpdateFormPermissionsInput(select=["read_project", "write_project"])

    message = tool.format_display_message(args)

    assert (
        message
        == "Update access token permissions — Select: read_project, write_project"
    )


def test_format_display_message_remove():
    tool = UpdateFormPermissions(metadata={})
    args = UpdateFormPermissionsInput(clear=["read_project"])

    message = tool.format_display_message(args)

    assert message == "Update access token permissions — Clear: read_project"


def test_format_display_message_add_and_remove():
    tool = UpdateFormPermissions(metadata={})
    args = UpdateFormPermissionsInput(
        select=["write_project"],
        clear=["read_project"],
    )

    message = tool.format_display_message(args)

    assert (
        message
        == "Update access token permissions — Select: write_project; Clear: read_project"
    )


def test_format_display_message_empty():
    tool = UpdateFormPermissions(metadata={})
    args = UpdateFormPermissionsInput()

    message = tool.format_display_message(args)

    assert message == "No permission changes"


def test_tool_properties():
    tool = UpdateFormPermissions(metadata={})

    assert tool.name == "update_form_permissions"
    assert "access token" in tool.description
    assert tool.args_schema == UpdateFormPermissionsInput
