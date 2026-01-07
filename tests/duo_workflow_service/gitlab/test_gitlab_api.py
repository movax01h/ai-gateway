from unittest.mock import AsyncMock, patch

import pytest

from duo_workflow_service.gitlab.gitlab_api import (
    GITLAB_18_2_QUERY,
    GITLAB_18_3_OR_ABOVE_QUERY,
    GITLAB_18_8_OR_ABOVE_QUERY,
    PromptInjectionProtectionLevel,
    extract_id_from_global_id,
    fetch_workflow_and_container_data,
    fetch_workflow_and_container_query,
)


@pytest.fixture(name="workflow_and_project_data")
def workflow_and_project_data_fixture():
    return {
        "duoWorkflowWorkflows": {
            "nodes": [
                {
                    "statusName": "created",
                    "projectId": "gid://gitlab/Project/789",
                    "project": {
                        "id": "gid://gitlab/Project/789",
                        "name": "no-languages-project",
                        "description": "Project without languages field",
                        "httpUrlToRepo": "http://example.com/no-lang-project.git",
                        "webUrl": "http://example.com/no-lang-project",
                    },
                    "namespaceId": None,
                    "namespace": None,
                    "agentPrivilegesNames": [],
                    "preApprovedAgentPrivilegesNames": [],
                    "mcpEnabled": False,
                    "allowAgentToRequestUser": False,
                    "firstCheckpoint": None,
                }
            ]
        }
    }


def test_extract_id_from_global_id():
    # Test with GraphQL format project ID
    assert extract_id_from_global_id("gid://gitlab/Project/123") == 123

    # Test with numeric string project ID
    assert extract_id_from_global_id("456") == 456

    # Test with missing project ID
    assert extract_id_from_global_id(None) == 0

    # Test with empty string project ID
    assert extract_id_from_global_id("") == 0

    # Test with non-numeric project ID that can't be converted
    with pytest.raises(ValueError):
        extract_id_from_global_id("not-a-number")


@pytest.mark.asyncio
async def test_fetch_workflow_and_container_data_success():
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
                        "languages": [
                            {"name": "Python", "share": 75.5},
                            {"name": "JavaScript", "share": 24.5},
                        ],
                        "statisticsDetailsPaths": {
                            "repository": "https://example.com/test-project/-/tree/main",
                        },
                    },
                    "namespaceId": None,
                    "namespace": None,
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
    project, namespace, workflow_config = await fetch_workflow_and_container_data(
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
    assert project["languages"] == [
        {"name": "Python", "share": 75.5},
        {"name": "JavaScript", "share": 24.5},
    ]
    assert project["default_branch"] == "main"
    assert namespace is None

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
async def test_fetch_workflow_and_container_data_no_workflow():
    gitlab_client = AsyncMock()

    # Mock empty response
    gitlab_client.graphql.return_value = {"duoWorkflowWorkflows": {"nodes": []}}

    workflow_id = "abc-123"
    with pytest.raises(
        Exception, match=f"No workflow found for workflow ID: {workflow_id}"
    ):
        await fetch_workflow_and_container_data(gitlab_client, workflow_id)

    # Verify GraphQL call
    gitlab_client.graphql.assert_called_once()


@pytest.mark.asyncio
async def test_fetch_workflow_and_container_data_with_empty_languages():
    gitlab_client = AsyncMock()
    # Mock GraphQL response with empty languages
    gitlab_client.graphql.return_value = {
        "duoWorkflowWorkflows": {
            "nodes": [
                {
                    "statusName": "created",
                    "projectId": "gid://gitlab/Project/456",
                    "project": {
                        "id": "gid://gitlab/Project/456",
                        "name": "empty-languages-project",
                        "description": "Project with no languages",
                        "httpUrlToRepo": "http://example.com/empty-project.git",
                        "webUrl": "http://example.com/empty-project",
                        "languages": [],
                    },
                    "namespaceId": None,
                    "namespace": None,
                    "agentPrivilegesNames": [],
                    "preApprovedAgentPrivilegesNames": [],
                    "mcpEnabled": False,
                    "allowAgentToRequestUser": False,
                    "firstCheckpoint": None,
                }
            ]
        }
    }

    workflow_id = "456"
    project, namespace, workflow_config = await fetch_workflow_and_container_data(
        gitlab_client, workflow_id
    )

    # Verify project data with empty languages
    assert project["id"] == 456
    assert project["languages"] == []
    assert namespace is None


@pytest.mark.asyncio
async def test_fetch_workflow_and_project_data_with_missing_languages(
    workflow_and_project_data,
):
    gitlab_client = AsyncMock()
    # Mock GraphQL response without languages field
    gitlab_client.graphql.return_value = workflow_and_project_data


@pytest.mark.asyncio
async def test_fetch_workflow_and_container_data_with_missing_languages():
    gitlab_client = AsyncMock()
    # Mock GraphQL response without languages field
    gitlab_client.graphql.return_value = {
        "duoWorkflowWorkflows": {
            "nodes": [
                {
                    "statusName": "created",
                    "projectId": "gid://gitlab/Project/789",
                    "project": {
                        "id": "gid://gitlab/Project/789",
                        "name": "no-languages-project",
                        "description": "Project without languages field",
                        "httpUrlToRepo": "http://example.com/no-lang-project.git",
                        "webUrl": "http://example.com/no-lang-project",
                    },
                    "namespaceId": None,
                    "namespace": None,
                    "agentPrivilegesNames": [],
                    "preApprovedAgentPrivilegesNames": [],
                    "mcpEnabled": False,
                    "allowAgentToRequestUser": False,
                    "firstCheckpoint": None,
                }
            ]
        }
    }

    workflow_id = "789"
    project, namespace, workflow_config = await fetch_workflow_and_container_data(
        gitlab_client, workflow_id
    )

    # Verify project data with missing languages defaults to empty list
    assert project["id"] == 789
    assert project["languages"] == []


@pytest.mark.asyncio
async def test_fetch_workflow_and_project_data_with_missing_repository(
    workflow_and_project_data,
):
    gitlab_client = AsyncMock()
    # Mock GraphQL response without repository field
    gitlab_client.graphql.return_value = workflow_and_project_data

    workflow_id = "1"
    project, namespace, workflow_config = await fetch_workflow_and_container_data(
        gitlab_client, workflow_id
    )

    assert project["id"] == 789
    assert project["default_branch"] is None
    assert namespace is None


@pytest.mark.asyncio
async def test_fetch_namespace_level_workflow():
    gitlab_client = AsyncMock()
    # Mock GraphQL response
    gitlab_client.graphql.return_value = {
        "duoWorkflowWorkflows": {
            "nodes": [
                {
                    "statusName": "created",
                    "projectId": None,
                    "project": None,
                    "namespaceId": "gid://gitlab/Type::Namespace/123",
                    "namespace": {
                        "id": "gid://gitlab/Group/123",
                        "name": "test-group",
                        "description": "Test Group",
                        "webUrl": "http://example.com/test-group",
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
    project, namespace, workflow_config = await fetch_workflow_and_container_data(
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
    assert namespace["id"] == 123
    assert namespace["description"] == "Test Group"
    assert namespace["name"] == "test-group"
    assert namespace["web_url"] == "http://example.com/test-group"
    assert project is None


@pytest.mark.asyncio
async def test_fetch_workflow_and_container_data_with_exclusion_rules():
    """Test that exclusion rules are included when available in the response."""
    gitlab_client = AsyncMock()

    # Mock GraphQL response with exclusion rules
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
                        "languages": [{"name": "Python", "share": 100.0}],
                        "duoContextExclusionSettings": {
                            "exclusionRules": ["*.secret", ".env"]
                        },
                    },
                    "namespaceId": None,
                    "namespace": None,
                    "agentPrivilegesNames": [],
                    "preApprovedAgentPrivilegesNames": [],
                    "mcpEnabled": False,
                    "allowAgentToRequestUser": False,
                    "firstCheckpoint": None,
                }
            ]
        }
    }

    workflow_id = "123"
    project, namespace, workflow_config = await fetch_workflow_and_container_data(
        gitlab_client, workflow_id
    )

    # Verify that exclusion rules are included
    assert project["exclusion_rules"] is not None
    assert len(project["exclusion_rules"]) == 2
    assert project["exclusion_rules"][0] == "*.secret"
    assert project["exclusion_rules"][1] == ".env"


@pytest.mark.asyncio
async def test_fetch_workflow_and_container_data_without_exclusion_rules():
    """Test that exclusion rules default to empty list when not present."""
    gitlab_client = AsyncMock()

    # Mock GraphQL response without exclusion rules
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
                        "languages": [{"name": "Python", "share": 100.0}],
                    },
                    "namespaceId": None,
                    "namespace": None,
                    "agentPrivilegesNames": [],
                    "preApprovedAgentPrivilegesNames": [],
                    "mcpEnabled": False,
                    "allowAgentToRequestUser": False,
                    "firstCheckpoint": None,
                }
            ]
        }
    }

    workflow_id = "123"
    project, namespace, workflow_config = await fetch_workflow_and_container_data(
        gitlab_client, workflow_id
    )

    # Verify that exclusion rules default to empty list
    assert project["exclusion_rules"] == []


@pytest.mark.asyncio
async def test_fetch_workflow_with_prompt_injection_protection_level():
    """Test that prompt injection protection level is parsed from project.rootGroup.aiSettings."""
    gitlab_client = AsyncMock()
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
                        "httpUrlToRepo": "http://example.com/test.git",
                        "webUrl": "http://example.com/test",
                        "rootGroup": {
                            "aiSettings": {
                                "promptInjectionProtectionLevel": "INTERRUPT"
                            }
                        },
                    },
                    "namespaceId": "gid://gitlab/Group/123",
                    "namespace": {
                        "id": "gid://gitlab/Group/123",
                        "name": "test-group",
                        "description": "Test Group",
                        "webUrl": "http://example.com/test-group",
                    },
                    "agentPrivilegesNames": [],
                    "preApprovedAgentPrivilegesNames": [],
                    "mcpEnabled": False,
                    "allowAgentToRequestUser": False,
                    "firstCheckpoint": None,
                }
            ]
        }
    }

    project, namespace, workflow_config = await fetch_workflow_and_container_data(
        gitlab_client, "123"
    )

    assert (
        workflow_config["prompt_injection_protection_level"]
        == PromptInjectionProtectionLevel.INTERRUPT
    )


@pytest.mark.asyncio
async def test_fetch_workflow_protection_level_defaults_to_log_only():
    """Test that protection level defaults to LOG_ONLY when not specified."""
    gitlab_client = AsyncMock()
    gitlab_client.graphql.return_value = {
        "duoWorkflowWorkflows": {
            "nodes": [
                {
                    "statusName": "created",
                    "projectId": "gid://gitlab/Project/123",
                    "project": {
                        "id": "gid://gitlab/Project/123",
                        "name": "test-project",
                        "description": "Test",
                        "httpUrlToRepo": "http://example.com/test.git",
                        "webUrl": "http://example.com/test",
                    },
                    "namespaceId": None,
                    "namespace": None,
                    "agentPrivilegesNames": [],
                    "preApprovedAgentPrivilegesNames": [],
                    "mcpEnabled": False,
                    "allowAgentToRequestUser": False,
                    "firstCheckpoint": None,
                }
            ]
        }
    }

    project, namespace, workflow_config = await fetch_workflow_and_container_data(
        gitlab_client, "123"
    )

    assert (
        workflow_config["prompt_injection_protection_level"]
        == PromptInjectionProtectionLevel.LOG_ONLY
    )


class TestQueryVersionSelection:
    """Test fetch_workflow_and_container_query returns correct query per GitLab version."""

    def test_query_selection_gitlab_below_18_8(self):
        """GitLab < 18.8 returns GITLAB_18_3_OR_ABOVE_QUERY (no aiSettings)."""
        with patch(
            "duo_workflow_service.gitlab.gitlab_api.gitlab_version"
        ) as mock_version:
            mock_version.get.return_value = "18.7.0"

            result = fetch_workflow_and_container_query()

            assert result == GITLAB_18_3_OR_ABOVE_QUERY
            assert "aiSettings" not in result

    def test_query_selection_gitlab_18_8_or_above(self):
        """GitLab >= 18.8 returns GITLAB_18_8_OR_ABOVE_QUERY (with aiSettings)."""
        with patch(
            "duo_workflow_service.gitlab.gitlab_api.gitlab_version"
        ) as mock_version:
            mock_version.get.return_value = "18.8.0"

            result = fetch_workflow_and_container_query()

            assert result == GITLAB_18_8_OR_ABOVE_QUERY
            assert "aiSettings" in result

    def test_query_selection_gitlab_below_18_3(self):
        """GitLab < 18.3 returns GITLAB_18_2_QUERY."""
        with patch(
            "duo_workflow_service.gitlab.gitlab_api.gitlab_version"
        ) as mock_version:
            mock_version.get.return_value = "18.2.0"

            result = fetch_workflow_and_container_query()

            assert result == GITLAB_18_2_QUERY


@pytest.mark.asyncio
async def test_protection_level_defaults_to_log_only_when_ai_settings_missing():
    """GitLab < 18.8: namespace exists but aiSettings missing, defaults to LOG_ONLY."""
    gitlab_client = AsyncMock()
    gitlab_client.graphql.return_value = {
        "duoWorkflowWorkflows": {
            "nodes": [
                {
                    "statusName": "created",
                    "projectId": None,
                    "project": None,
                    "namespaceId": "gid://gitlab/Group/123",
                    "namespace": {
                        "id": "gid://gitlab/Group/123",
                        "name": "test-group",
                        "description": "Test Group",
                        "webUrl": "http://example.com/test-group",
                        # No aiSettings field - simulates GitLab < 18.8
                    },
                    "agentPrivilegesNames": [],
                    "preApprovedAgentPrivilegesNames": [],
                    "mcpEnabled": False,
                    "allowAgentToRequestUser": False,
                    "firstCheckpoint": None,
                }
            ]
        }
    }

    project, namespace, workflow_config = await fetch_workflow_and_container_data(
        gitlab_client, "123"
    )

    assert (
        workflow_config["prompt_injection_protection_level"]
        == PromptInjectionProtectionLevel.LOG_ONLY
    )
