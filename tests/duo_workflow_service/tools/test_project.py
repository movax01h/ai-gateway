from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.tools.project import GetProject, GetProjectInput


@pytest.mark.asyncio
async def test_get_project():

    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget.side_effect = ["{}"]
    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = GetProject(metadata=metadata)  # type: ignore

    response = await tool.arun({"project_id": "1"})

    assert response == "{}"

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/projects/1", parse_json=False
    )


def test_get_project_format_display_message():
    tool = GetProject(description="Get project description")

    input_data = GetProjectInput(project_id=123)

    message = tool.format_display_message(input_data)

    expected_message = "Get project information for project 123"
    assert message == expected_message
