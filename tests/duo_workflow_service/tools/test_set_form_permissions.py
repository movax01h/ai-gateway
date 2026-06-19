import json

import pytest

from duo_workflow_service.tools.set_form_permissions import (
    SetFormPermissions,
    SetFormPermissionsInput,
)


@pytest.mark.asyncio
async def test_execute_with_select_buckets():
    tool = SetFormPermissions(metadata={})

    response = await tool.arun(
        {"select": {"namespace": ["read_project"], "user": ["read_snippet"]}}
    )

    parsed = json.loads(response)
    assert parsed["select"] == {"namespace": ["read_project"], "user": ["read_snippet"]}
    assert parsed["clear"] == {}


@pytest.mark.asyncio
async def test_execute_with_clear_buckets():
    tool = SetFormPermissions(metadata={})

    response = await tool.arun({"clear": {"namespace": ["read_project"]}})

    parsed = json.loads(response)
    assert parsed["select"] == {}
    assert parsed["clear"] == {"namespace": ["read_project"]}


@pytest.mark.asyncio
async def test_execute_with_none_defaults():
    tool = SetFormPermissions(metadata={})

    response = await tool.arun({})

    parsed = json.loads(response)
    assert parsed["select"] == {}
    assert parsed["clear"] == {}


@pytest.mark.asyncio
async def test_execute_with_permission_in_multiple_boundaries():
    tool = SetFormPermissions(metadata={})

    response = await tool.arun(
        {"select": {"user": ["read_snippet"], "namespace": ["read_snippet"]}}
    )

    parsed = json.loads(response)
    assert parsed["select"] == {
        "user": ["read_snippet"],
        "namespace": ["read_snippet"],
    }


@pytest.mark.asyncio
async def test_execute_accepts_global_level():
    tool = SetFormPermissions(metadata={})

    response = await tool.arun({"select": {"global": ["read_instance_metadata"]}})

    parsed = json.loads(response)
    assert parsed["select"] == {"global": ["read_instance_metadata"]}


@pytest.mark.asyncio
async def test_execute_drops_unknown_scope_keys():
    tool = SetFormPermissions(metadata={})

    response = await tool.arun(
        {
            "select": {
                "namespace": ["read_project"],
                "GROUP": ["write_project"],
                "project": ["read_snippet"],
                "instance": ["read_instance_metadata"],
            }
        }
    )

    parsed = json.loads(response)
    assert parsed["select"] == {"namespace": ["read_project"]}


@pytest.mark.asyncio
async def test_execute_dedupes_names_within_a_level():
    tool = SetFormPermissions(metadata={})

    response = await tool.arun(
        {"select": {"namespace": ["read_project", "read_project", "write_project"]}}
    )

    parsed = json.loads(response)
    assert parsed["select"] == {"namespace": ["read_project", "write_project"]}


@pytest.mark.asyncio
async def test_execute_drops_non_dict_buckets():
    tool = SetFormPermissions(metadata={})

    response = await tool.arun({"select": ["read_project"], "clear": "read_snippet"})

    parsed = json.loads(response)
    assert parsed["select"] == {}
    assert parsed["clear"] == {}


@pytest.mark.asyncio
async def test_execute_drops_non_list_scope_value():
    tool = SetFormPermissions(metadata={})

    response = await tool.arun({"select": {"namespace": "read_project"}})

    parsed = json.loads(response)
    assert parsed["select"] == {}


@pytest.mark.asyncio
async def test_execute_normalizes_scope_key_casing_and_whitespace():
    tool = SetFormPermissions(metadata={})

    response = await tool.arun(
        {
            "select": {
                "Global": ["read_instance_metadata"],
                " namespace ": ["read_project"],
            }
        }
    )

    parsed = json.loads(response)
    assert parsed["select"] == {
        "namespace": ["read_project"],
        "global": ["read_instance_metadata"],
    }


@pytest.mark.asyncio
async def test_execute_trims_and_drops_blank_names():
    tool = SetFormPermissions(metadata={})

    response = await tool.arun(
        {"select": {"namespace": [" read_project ", "", "   ", "read_project"]}}
    )

    parsed = json.loads(response)
    assert parsed["select"] == {"namespace": ["read_project"]}


def test_format_display_message_groups_by_permission():
    tool = SetFormPermissions(metadata={})
    args = SetFormPermissionsInput(
        select={"user": ["read_snippet"], "namespace": ["read_snippet", "read_project"]}
    )

    message = tool.format_display_message(args)

    assert message == (
        "Update access token permissions — "
        "Select: read_snippet (namespace, user), read_project (namespace)"
    )


def test_format_display_message_select_and_clear():
    tool = SetFormPermissions(metadata={})
    args = SetFormPermissionsInput(
        select={"namespace": ["write_project"]},
        clear={"user": ["read_snippet"]},
    )

    message = tool.format_display_message(args)

    assert message == (
        "Update access token permissions — "
        "Select: write_project (namespace); Clear: read_snippet (user)"
    )


def test_format_display_message_empty():
    tool = SetFormPermissions(metadata={})

    assert (
        tool.format_display_message(SetFormPermissionsInput())
        == "No permission changes"
    )


def test_tool_properties():
    tool = SetFormPermissions(metadata={})

    assert tool.name == "set_form_permissions"
    assert "access level" in tool.description
    assert tool.args_schema == SetFormPermissionsInput
