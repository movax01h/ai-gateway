from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.gitlab.gitlab_project import (
    fetch_project_data_with_workflow_id,
)


@pytest.mark.asyncio
async def test_fetch_project_data_with_workflow_id_success():
    gitlab_client = AsyncMock()
    # Mock response from workflow lookup
    gitlab_client.aget.side_effect = [
        # First call returns a dict with project_id
        {"project_id": 123},
        # Second call returns the actual project data
        {
            "id": 123,
            "description": "Test Project",
            "name": "test-project",
            "http_url_to_repo": "http://example.com/test-project.git",
        },
    ]

    workflow_id = "111"
    project = await fetch_project_data_with_workflow_id(gitlab_client, workflow_id)

    # Verify the first call: fetch workflow details
    gitlab_client.aget.assert_any_call(
        path=f"/api/v4/ai/duo_workflows/workflows/{workflow_id}", parse_json=True
    )
    # Verify the second call: fetch project details
    gitlab_client.aget.assert_any_call(path="/api/v4/projects/123", parse_json=True)

    assert project["id"] == 123
    assert project["description"] == "Test Project"
    assert project["name"] == "test-project"
    assert project["http_url_to_repo"] == "http://example.com/test-project.git"


@pytest.mark.asyncio
async def test_fetch_project_data_with_workflow_id_missing_project_id():
    gitlab_client = AsyncMock()
    # Mock response that omits project_id
    gitlab_client.aget.return_value = {"not_project_id": 999}

    workflow_id = "abc-123"
    with pytest.raises(Exception, match="Failed to retrieve project ID"):
        await fetch_project_data_with_workflow_id(gitlab_client, workflow_id)

    gitlab_client.aget.assert_called_once_with(
        path="/api/v4/ai/duo_workflows/workflows/abc-123", parse_json=True
    )
