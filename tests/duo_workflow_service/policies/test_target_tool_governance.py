import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from structlog.testing import capture_logs

from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.policies.target_tool_governance import (
    TargetToolGovernanceUnavailable,
    ToolCallDeniedByTargetRules,
    governs,
)
from duo_workflow_service.tools import (
    CreateMergeRequestNote,
    CreateWorkItem,
    CreateWorkItemNote,
    DismissVulnerability,
    GetWorkItem,
    UpdateMergeRequest,
    UpdateWorkItem,
    WorkplanPublishTracks,
)
from lib.feature_flags.context import FeatureFlag, current_feature_flag_context


@pytest.fixture(name="flag_pushed")
def flag_pushed_fixture(request):
    pushed = getattr(request, "param", True)
    flags = {FeatureFlag.DUO_WORKFLOW_TARGET_TOOL_GOVERNANCE.value} if pushed else set()
    token = current_feature_flag_context.set(flags)
    yield pushed
    current_feature_flag_context.reset(token)


@pytest.fixture(name="gitlab_client")
def gitlab_client_fixture():
    return MagicMock(spec=GitlabHttpClient, graphql=AsyncMock())


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client):
    return {
        "gitlab_client": gitlab_client,
        "gitlab_host": "gitlab.example.com",
        "workflow_id": "42",
    }


def verdict(permission, message=None):
    return {
        "duoWorkflowWorkflows": {
            "nodes": [
                {
                    "toolCallGovernance": {
                        "permission": permission,
                        "message": message,
                    }
                }
            ]
        }
    }


async def run(tool_class, metadata, **kwargs):
    with patch.object(tool_class, "_execute", AsyncMock(return_value="ran")) as execute:
        try:
            return await tool_class(metadata=metadata)._arun(**kwargs), execute
        except (ToolCallDeniedByTargetRules, TargetToolGovernanceUnavailable) as error:
            return error, execute


@pytest.mark.asyncio
async def test_a_denied_write_fails_with_the_reason_and_does_not_run(
    flag_pushed, gitlab_client, metadata
):
    gitlab_client.graphql.return_value = verdict(
        "DENY",
        "update_merge_request is denied by the AI tool rules of group-b/project.",
    )

    result, execute = await run(
        UpdateMergeRequest, metadata, project_id="group-b/project", merge_request_iid=1
    )

    assert isinstance(result, ToolCallDeniedByTargetRules)
    assert str(result) == (
        "update_merge_request is denied by the AI tool rules of group-b/project."
    )
    execute.assert_not_called()


@pytest.mark.asyncio
async def test_the_check_sends_the_tool_name_the_session_and_only_the_target_arguments(
    flag_pushed, gitlab_client, metadata
):
    gitlab_client.graphql.return_value = verdict("ALLOW")

    await run(
        CreateWorkItem,
        metadata,
        url="https://gitlab.example.com/group-b/project",
        group_id=None,
        title="Secret title",
        type_name="Issue",
    )

    _query, variables = gitlab_client.graphql.call_args.args
    assert gitlab_client.graphql.call_args.kwargs == {
        "timeout": 5.0,
        "retry_transient_errors": False,
    }
    assert variables["workflowId"] == "gid://gitlab/Ai::DuoWorkflows::Workflow/42"
    assert variables["toolName"] == "create_work_item"
    assert json.loads(variables["toolCallArgs"]) == {
        "url": "https://gitlab.example.com/group-b/project"
    }


@pytest.mark.asyncio
async def test_the_check_reads_the_target_a_flow_enforces(
    flag_pushed, gitlab_client, metadata
):
    gitlab_client.graphql.return_value = verdict("ALLOW")
    tool = UpdateMergeRequest(metadata=metadata)
    setattr(
        tool,
        "_tool_options",
        {"update_merge_request": {"project_id": "flow/project"}},
    )

    with patch.object(UpdateMergeRequest, "_execute", AsyncMock(return_value="ran")):
        await tool._arun(project_id="model/project", merge_request_iid=1)

    _query, variables = gitlab_client.graphql.call_args.args
    assert json.loads(variables["toolCallArgs"]) == {"project_id": "flow/project"}


@pytest.mark.asyncio
@pytest.mark.parametrize("permission", ["ALLOW", "ASK"])
async def test_a_write_that_is_not_denied_runs(
    flag_pushed, gitlab_client, metadata, permission
):
    gitlab_client.graphql.return_value = verdict(permission)

    result, execute = await run(
        UpdateMergeRequest, metadata, project_id=7, merge_request_iid=1
    )

    assert result == "ran"
    execute.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        Exception("GraphQL request timed out after 10.0 seconds"),
        Exception("GraphQL request failed with HTTP 500: error"),
    ],
)
async def test_a_failed_check_refuses_the_write(
    flag_pushed, gitlab_client, metadata, failure
):
    gitlab_client.graphql.side_effect = failure

    result, execute = await run(
        UpdateMergeRequest, metadata, project_id=7, merge_request_iid=1
    )

    assert isinstance(result, TargetToolGovernanceUnavailable)
    assert "could not check the AI tool rules" in str(result)
    execute.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        {"duoWorkflowWorkflows": {"nodes": []}},
        {"duoWorkflowWorkflows": {"nodes": [{"toolCallGovernance": None}]}},
        {
            "duoWorkflowWorkflows": {
                "nodes": [{"toolCallGovernance": {"permission": None}}]
            }
        },
        {
            "duoWorkflowWorkflows": {
                "nodes": [{"toolCallGovernance": {"permission": "deny"}}]
            }
        },
        {},
    ],
)
async def test_an_unparsable_answer_refuses_the_write(
    flag_pushed, gitlab_client, metadata, response
):
    gitlab_client.graphql.return_value = response

    result, execute = await run(
        UpdateMergeRequest, metadata, project_id=7, merge_request_iid=1
    )

    assert isinstance(result, TargetToolGovernanceUnavailable)
    execute.assert_not_called()


@pytest.mark.asyncio
async def test_a_denial_without_a_message_still_names_the_tool(
    flag_pushed, gitlab_client, metadata
):
    gitlab_client.graphql.return_value = verdict("DENY")

    result, _execute = await run(
        UpdateMergeRequest, metadata, project_id=7, merge_request_iid=1
    )

    assert str(result) == "update_merge_request is denied by AI tool rules."


@pytest.mark.asyncio
@pytest.mark.parametrize("flag_pushed", [False], indirect=True)
async def test_without_the_flag_gitlab_is_never_asked(
    flag_pushed, gitlab_client, metadata
):
    result, execute = await run(
        UpdateMergeRequest, metadata, project_id=7, merge_request_iid=1
    )

    assert result == "ran"
    execute.assert_called_once()
    gitlab_client.graphql.assert_not_called()


@pytest.mark.asyncio
async def test_a_read_tool_is_not_checked(flag_pushed, gitlab_client, metadata):
    result, _execute = await run(GetWorkItem, metadata, project_id=7, work_item_iid=1)

    assert result == "ran"
    gitlab_client.graphql.assert_not_called()


@pytest.mark.asyncio
async def test_a_write_without_target_arguments_is_not_checked(
    flag_pushed, gitlab_client, metadata
):
    result, _execute = await run(
        DismissVulnerability,
        metadata,
        vulnerability_id="gid://gitlab/Vulnerability/1",
        comment="c",
        dismissal_reason="FALSE_POSITIVE",
    )

    assert result == "ran"
    gitlab_client.graphql.assert_not_called()


@pytest.mark.asyncio
async def test_a_write_that_names_no_target_is_not_checked(
    flag_pushed, gitlab_client, metadata
):
    result, _execute = await run(WorkplanPublishTracks, metadata, url=None, tracks=None)

    assert result == "ran"
    gitlab_client.graphql.assert_not_called()


@pytest.mark.parametrize(
    "tool_class",
    [
        CreateWorkItem,
        CreateWorkItemNote,
        CreateMergeRequestNote,
        UpdateWorkItem,
        UpdateMergeRequest,
    ],
)
def test_the_slack_assistant_write_tools_are_governed(tool_class):
    assert governs(tool_class(metadata={}))


@pytest.mark.asyncio
async def test_a_denial_is_logged_with_its_reason_and_target(
    flag_pushed, gitlab_client, metadata
):
    gitlab_client.graphql.return_value = verdict("DENY", "denied")

    with capture_logs() as logs:
        await run(UpdateMergeRequest, metadata, project_id=7, merge_request_iid=1)

    assert {
        "event": "Tool call denied by the AI tool rules of its target",
        "log_level": "info",
        "reason": "denied_by_target_rules",
        "tool_name": "update_merge_request",
        "workflow_id": "42",
        "target_args": {"project_id": 7},
    }.items() <= logs[-1].items()


@pytest.mark.asyncio
async def test_a_failed_check_is_logged_with_its_reason_and_target(
    flag_pushed, gitlab_client, metadata
):
    gitlab_client.graphql.side_effect = Exception("GraphQL request timed out")

    with capture_logs() as logs:
        await run(UpdateMergeRequest, metadata, project_id=7, merge_request_iid=1)

    assert {
        "event": "Target tool governance check failed, refusing the tool call",
        "log_level": "warning",
        "reason": "check_unavailable",
        "tool_name": "update_merge_request",
        "workflow_id": "42",
        "target_args": {"project_id": 7},
    }.items() <= logs[-1].items()
