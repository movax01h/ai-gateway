from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from duo_workflow_service.gitlab.http_client import GitlabHttpClient, checkpoint_decoder


class MockGitLabHttpClient(GitlabHttpClient):
    """Mock implementation of GitlabHttpClient for testing interface methods."""

    def __init__(self):
        self.mock_call = AsyncMock()

    async def _call(
        self,
        path,
        method,
        parse_json=True,
        data=None,
        params=None,
        object_hook=None,
    ):
        return await self.mock_call(path, method, parse_json, data, params, object_hook)


@pytest.fixture
def client():
    return MockGitLabHttpClient()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, path, body, params, parse_json, mock_return_value, expected_result",
    [
        (
            "GET",
            "/api/v4/projects/1",
            None,
            None,
            True,
            {"key": "value"},
            {"key": "value"},
        ),
        (
            "GET",
            "/api/v4/projects/1/jobs/102/trace",
            None,
            None,
            False,
            "Non-JSON response",
            "Non-JSON response",
        ),
        (
            "GET",
            "/api/v4/projects",
            None,
            {"per_page": 100},
            True,
            {"projects": []},
            {"projects": []},
        ),
        (
            "POST",
            "/api/v4/test",
            '{ "test": 1 }',
            None,
            True,
            {"key": "value"},
            {"key": "value"},
        ),
        (
            "PUT",
            "/api/v4/test",
            '{ "test": 1 }',
            None,
            True,
            {"key": "value"},
            {"key": "value"},
        ),
        (
            "PATCH",
            "/api/v4/test",
            '{ "test": 1 }',
            None,
            True,
            {"key": "value"},
            {"key": "value"},
        ),
    ],
)
async def test_gitlab_http_client_interface_methods(
    client,
    method,
    path,
    body,
    params,
    parse_json,
    mock_return_value,
    expected_result,
):
    client.mock_call.return_value = mock_return_value

    if method == "GET":
        result = await client.aget(path, params=params, parse_json=parse_json)
        client.mock_call.assert_called_once_with(
            path, "GET", parse_json, None, params, None
        )
    elif method == "POST":
        result = await client.apost(path, body, parse_json=parse_json)
        client.mock_call.assert_called_once_with(
            path, "POST", parse_json, body, None, None
        )
    elif method == "PUT":
        result = await client.aput(path, body, parse_json=parse_json)
        client.mock_call.assert_called_once_with(
            path, "PUT", parse_json, body, None, None
        )
    elif method == "PATCH":
        result = await client.apatch(path, body, parse_json=parse_json)
        client.mock_call.assert_called_once_with(
            path, "PATCH", parse_json, body, None, None
        )
    else:
        pytest.fail(f"Unexpected HTTP method: {method}")
        result = None

    assert result == expected_result


@pytest.mark.asyncio
async def test_gitlab_http_client_with_object_hook(client):
    # Setup mock response with checkpoint data
    checkpoint_json = {
        "type": "SystemMessage",
        "content": "You are an AI planner.",
        "additional_kwargs": {},
        "response_metadata": {},
        "name": None,
        "id": None,
    }

    # Configure the mock to actually apply the object_hook to the returned data
    def side_effect(path, method, parse_json, data, params, object_hook):
        if object_hook:
            return object_hook(checkpoint_json)
        return checkpoint_json

    client.mock_call.side_effect = side_effect

    # Call aget with object_hook
    result = await client.aget(
        "/api/test", parse_json=True, object_hook=checkpoint_decoder
    )

    # Verify the object hook was passed correctly
    client.mock_call.assert_called_once_with(
        "/api/test", "GET", True, None, None, checkpoint_decoder
    )

    # Result should be a SystemMessage instance
    assert isinstance(result, SystemMessage)
    assert result.content == "You are an AI planner."


def test_checkpoint_decoder():
    # Test with SystemMessage
    system_json = {
        "type": "SystemMessage",
        "content": "System content",
        "additional_kwargs": {},
        "response_metadata": {},
    }
    result = checkpoint_decoder(system_json)
    assert isinstance(result, SystemMessage)
    assert result.content == "System content"

    # Test with HumanMessage
    human_json = {
        "type": "HumanMessage",
        "content": "Human content",
        "additional_kwargs": {},
        "response_metadata": {},
    }
    result = checkpoint_decoder(human_json)
    assert isinstance(result, HumanMessage)
    assert result.content == "Human content"

    # Test with AIMessage
    ai_json = {
        "type": "AIMessage",
        "content": "AI content",
        "additional_kwargs": {},
        "response_metadata": {},
        "tool_calls": [],
    }
    result = checkpoint_decoder(ai_json)
    assert isinstance(result, AIMessage)
    assert result.content == "AI content"

    # Test with ToolMessage
    tool_json = {
        "type": "ToolMessage",
        "content": "Tool content",
        "tool_call_id": "tool123",
        "additional_kwargs": {},
        "response_metadata": {},
    }
    result = checkpoint_decoder(tool_json)
    assert isinstance(result, ToolMessage)
    assert result.content == "Tool content"
    assert result.tool_call_id == "tool123"

    # Test with unknown type
    unknown_json = {"type": "Unknown", "content": "Unknown content"}
    result = checkpoint_decoder(unknown_json)
    assert not isinstance(result, (SystemMessage, HumanMessage, AIMessage, ToolMessage))
    assert result["type"] == "Unknown"
    assert result["content"] == "Unknown content"

    # Test with non-message JSON
    regular_json = {"key": "value"}
    result = checkpoint_decoder(regular_json)
    assert result == {"key": "value"}


def test_parse_response():
    """Test the _parse_response method of GitlabHttpClient."""
    client = MockGitLabHttpClient()

    # Test 1: Valid JSON string
    json_str = '{"key": "value", "number": 42}'
    result = client._parse_response(json_str)
    assert result == {"key": "value", "number": 42}

    # Test 2: Valid JSON string with object hook
    message_json = '{"type": "SystemMessage", "content": "Test message", "additional_kwargs": {}, "response_metadata": {}}'
    result = client._parse_response(message_json, object_hook=checkpoint_decoder)
    assert isinstance(result, SystemMessage)
    assert result.content == "Test message"

    # Test 3: Non-JSON string with parse_json=False
    non_json = "This is not JSON"
    result = client._parse_response(non_json, parse_json=False)
    assert (
        result == "This is not JSON"
    )  # Should return original string when parse_json=False

    # Test 4: Non-JSON string with parse_json=True (should handle error)
    result = client._parse_response(non_json)
    assert result == {}  # Returns empty dict on JSON decode error

    # Test 5: Already parsed JSON (dict)
    parsed_json = {"already": "parsed"}
    result = client._parse_response(parsed_json)
    assert result == parsed_json

    # Test 6: Already parsed JSON (list) - should return empty dict
    parsed_list = ["item1", "item2"]
    result = client._parse_response(parsed_list)
    assert result == {}

    # Test 7: Empty string with parse_json=True (should return empty dict)
    result = client._parse_response("")
    assert result == {}

    # Test 8: Empty string with parse_json=False (should return empty string)
    result = client._parse_response("", parse_json=False)
    assert result == ""

    # Test 9: None input with parse_json=True (should return empty dict)
    result = client._parse_response(None)
    assert result == {}

    # Test 10: None input with parse_json=False (should return None)
    result = client._parse_response(None, parse_json=False)
    assert result is None
