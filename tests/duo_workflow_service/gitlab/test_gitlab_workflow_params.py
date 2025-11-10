from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.gitlab.gitlab_workflow_params import fetch_workflow_config
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse


@pytest.mark.asyncio
async def test_fetch_workflow_config_success():
    gitlab_client = AsyncMock()

    data = {
        "id": "44",
        "project_id": 123,
        "agent_privileges": [1, 2],
        "agent_privileges_names": ["read_write_files", "read_only_gitlab"],
        "workflow_definition": "software_development",
        "status": "finished",
        "allow_agent_to_request_user": True,
    }

    mock_response = GitLabHttpResponse(
        status_code=200,
        body=data,
    )
    gitlab_client.aget = AsyncMock(return_value=mock_response)

    workflow_id = "44"
    workflow = await fetch_workflow_config(gitlab_client, workflow_id)

    # Verify the first call: fetch workflow details
    gitlab_client.aget.assert_called_once_with(
        path="/api/v4/ai/duo_workflows/workflows/44",
        parse_json=True,
    )

    assert workflow["id"] == workflow_id
    assert workflow["project_id"] == 123
    assert workflow["agent_privileges"] == [1, 2]
    assert workflow["agent_privileges_names"] == [
        "read_write_files",
        "read_only_gitlab",
    ]
    assert workflow["workflow_definition"] == "software_development"
    assert workflow["status"] == "finished"
    assert workflow["allow_agent_to_request_user"] == True
