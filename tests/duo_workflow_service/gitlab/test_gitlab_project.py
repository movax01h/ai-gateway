from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.gitlab.gitlab_project import (
    extract_project_id_from_workflow,
    fetch_workflow_and_project_data,
)


def test_extract_project_id_from_workflow():
    # Test with GraphQL format project ID
    workflow = {"projectId": "gid://gitlab/Project/123"}
    assert extract_project_id_from_workflow(workflow) == 123

    # Test with numeric string project ID
    workflow = {"projectId": "456"}
    assert extract_project_id_from_workflow(workflow) == 456

    # Test with missing project ID
    workflow = {}
    assert extract_project_id_from_workflow(workflow) == 0

    # Test with empty string project ID
    workflow = {"projectId": ""}
    assert extract_project_id_from_workflow(workflow) == 0

    # Test with None project ID
    workflow = {"projectId": None}
    assert extract_project_id_from_workflow(workflow) == 0

    # Test with non-numeric project ID that can't be converted
    workflow = {"projectId": "not-a-number"}
    with pytest.raises(ValueError):
        extract_project_id_from_workflow(workflow)


@pytest.mark.asyncio
async def test_fetch_workflow_and_project_data_success():
    gitlab_client = AsyncMock()
    # Mock GraphQL response
    gitlab_client.graphql.return_value = {
        "duoWorkflowWorkflows": {
            "nodes": [
                {
                    "statusName": "created",
                    "projectId": "gid://gitlab/Project/123",
                    "project": {
                        "id": "gid://gitlab/Project/123",
                        "name": "test-project",
                        "description": "Test Project",
                        "httpUrlToRepo": "http://example.com/test-project.git",
                        "webUrl": "http://example.com/test-project",
                    },
                    "agentPrivilegesNames": ["read_repository", "write_repository"],
                    "preApprovedAgentPrivilegesNames": ["read_repository"],
                    "mcpEnabled": True,
                    "allowAgentToRequestUser": True,
                    "firstCheckpoint": {"checkpoint": "{}"},
                }
            ]
        }
    }

    workflow_id = "111"
    project, workflow_config = await fetch_workflow_and_project_data(
        gitlab_client, workflow_id
    )

    # Verify GraphQL call
    gitlab_client.graphql.assert_called_once()
    call_args = gitlab_client.graphql.call_args[0]
    assert isinstance(call_args[0], str)
    assert call_args[1] == {
        "workflowId": f"gid://gitlab/Ai::DuoWorkflows::Workflow/{workflow_id}"
    }

    # Verify project data
    assert project["id"] == 123
    assert project["description"] == "Test Project"
    assert project["name"] == "test-project"
    assert project["http_url_to_repo"] == "http://example.com/test-project.git"
    assert project["web_url"] == "http://example.com/test-project"

    # Verify workflow config
    assert workflow_config["agent_privileges_names"] == [
        "read_repository",
        "write_repository",
    ]
    assert workflow_config["pre_approved_agent_privileges_names"] == ["read_repository"]
    assert workflow_config["workflow_status"] == "created"
    assert workflow_config["mcp_enabled"] is True
    assert workflow_config["allow_agent_to_request_user"] is True
    assert workflow_config["first_checkpoint"] == {"checkpoint": "{}"}


@pytest.mark.asyncio
async def test_fetch_workflow_and_project_data_no_workflow():
    gitlab_client = AsyncMock()
    # Mock empty response
    gitlab_client.graphql.return_value = {"duoWorkflowWorkflows": {"nodes": []}}

    workflow_id = "abc-123"
    with pytest.raises(
        Exception, match=f"No workflow found for workflow ID: {workflow_id}"
    ):
        await fetch_workflow_and_project_data(gitlab_client, workflow_id)

    # Verify GraphQL call
    gitlab_client.graphql.assert_called_once()
