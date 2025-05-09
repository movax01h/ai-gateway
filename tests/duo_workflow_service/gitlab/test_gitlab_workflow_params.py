from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.gitlab.gitlab_workflow_params import fetch_workflow_config


@pytest.mark.asyncio
async def test_fetch_workflow_config_success():
    gitlab_client = AsyncMock()

    gitlab_client.aget.side_effect = [
        {
            "id": "44",
            "project_id": 123,
            "agent_privileges": [1, 2],
            "agent_privileges_names": ["read_write_files", "read_only_gitlab"],
            "workflow_definition": "software_development",
            "status": "finished",
            "allow_agent_to_request_user": True,
        }
    ]

    workflow_id = "44"
    workflow = await fetch_workflow_config(gitlab_client, workflow_id)

    # Verify the first call: fetch workflow details
    gitlab_client.aget.assert_any_call(
        path="/api/v4/ai/duo_workflows/workflows/44", parse_json=True
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
