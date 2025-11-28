import json
from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.user import GetCurrentUser, GetCurrentUserInput


@pytest.mark.asyncio
async def test_get_current_user_success():
    # Mock response from GitLab API
    mock_response_data = {
        "username": "john_doe",
        "job_title": "Software Engineer",
        "preferred_language": "en",
        "id": 123,
        "email": "john@example.com",  # This should be filtered out
    }

    mock_response = GitLabHttpResponse(
        status_code=200,
        body=mock_response_data,
    )
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = GetCurrentUser(metadata=metadata)

    response = await tool.arun({})

    parsed_response = json.loads(response)
    assert "user" in parsed_response

    user_data = parsed_response["user"]
    assert user_data["user_id"] == 123
    assert user_data["user_name"] == "john_doe"
    assert user_data["job_title"] == "Software Engineer"
    assert user_data["preferred_language"] == "en"
    assert len(user_data) == 4
    assert "email" not in user_data

    gitlab_client_mock.aget.assert_called_once_with(
        path="/api/v4/user", parse_json=True
    )


@pytest.mark.asyncio
async def test_get_current_user_with_missing_fields():
    mock_response_data = {
        "id": 123,
        "username": "jane_doe",
        # job_title and preferred_language are missing
    }

    mock_response = GitLabHttpResponse(
        status_code=200,
        body=mock_response_data,
    )
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = GetCurrentUser(metadata=metadata)

    response = await tool.arun({})

    parsed_response = json.loads(response)
    user_data = parsed_response["user"]
    assert user_data["user_id"] == 123
    assert user_data["user_name"] == "jane_doe"
    assert user_data["job_title"] is None
    assert user_data["preferred_language"] is None


@pytest.mark.asyncio
async def test_get_current_user_api_error():
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget.side_effect = Exception("API connection failed")

    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = GetCurrentUser(metadata=metadata)

    response = await tool.arun({})

    parsed_response = json.loads(response)
    assert "error" in parsed_response
    assert parsed_response["error"] == "API connection failed"
    assert "user" not in parsed_response


def test_get_current_user_format_display_message():
    tool = GetCurrentUser(metadata={})

    input_data = GetCurrentUserInput()

    message = tool.format_display_message(input_data)

    assert message == "Get current user information"


def test_get_current_user_tool_properties():
    tool = GetCurrentUser(metadata={})

    assert tool.name == "get_current_user"
    assert "Get the current user information from GitLab API" in tool.description
    assert tool.args_schema == GetCurrentUserInput
    assert "user id" in tool.description
    assert "user name" in tool.description
    assert "job title" in tool.description
    assert "preferred language" in tool.description


@pytest.mark.asyncio
async def test_get_current_user_json_serialization():
    mock_response_data = {
        "id": 123,
        "username": "test_user",
        "job_title": "Developer",
        "preferred_language": "es",
    }

    mock_response = GitLabHttpResponse(
        status_code=200,
        body=mock_response_data,
    )
    gitlab_client_mock = AsyncMock()
    gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

    metadata = {
        "gitlab_client": gitlab_client_mock,
    }

    tool = GetCurrentUser(metadata=metadata)

    response = await tool.arun({})

    # Verify response is valid JSON
    parsed_response = json.loads(response)
    assert isinstance(parsed_response, dict)

    # Verify it can be serialized back to JSON
    json.dumps(parsed_response)  # Should not raise an exception
