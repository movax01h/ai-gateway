import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from contract import contract_pb2
from duo_workflow_service.gitlab.executor_http_client import ExecutorGitLabHttpClient


@pytest.fixture
def mock_execute_action():
    return AsyncMock()


@pytest.fixture
def client():
    outbox = asyncio.Queue()
    inbox = asyncio.Queue()
    return ExecutorGitLabHttpClient(outbox, inbox)


@pytest.fixture
def monkeypatch_execute_action(monkeypatch, mock_execute_action):
    monkeypatch.setattr(
        "duo_workflow_service.gitlab.executor_http_client._execute_action",
        mock_execute_action,
    )
    return mock_execute_action


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
            '{"key": "value"}',
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
            '{"projects": []}',
            {"projects": []},
        ),
        (
            "POST",
            "/api/v4/test",
            '{ "test": 1 }',
            None,
            True,
            '{"key": "value"}',
            {"key": "value"},
        ),
        (
            "PUT",
            "/api/v4/test",
            '{ "test": 1 }',
            None,
            True,
            '{"key": "value"}',
            {"key": "value"},
        ),
        (
            "PATCH",
            "/api/v4/test",
            '{ "test": 1 }',
            None,
            True,
            '{"key": "value"}',
            {"key": "value"},
        ),
    ],
)
async def test_executor_gitlab_http_client(
    client,
    monkeypatch_execute_action,
    method,
    path,
    body,
    params,
    parse_json,
    mock_return_value,
    expected_result,
):
    monkeypatch_execute_action.return_value = mock_return_value

    expected_path = path
    if params:
        from urllib.parse import urlencode

        query_string = urlencode(params)
        expected_path = f"{path}?{query_string}"

    if method == "GET":
        result = await client.aget(path, params=params, parse_json=parse_json)
    elif method == "POST":
        result = await client.apost(path, body, parse_json=parse_json)
    elif method == "PUT":
        result = await client.aput(path, body, parse_json=parse_json)
    elif method == "PATCH":
        result = await client.apatch(path, body, parse_json=parse_json)
    else:
        pytest.fail(f"Unexpected HTTP method: {method}")
        result = None

    monkeypatch_execute_action.assert_called_once()
    call_args = monkeypatch_execute_action.call_args[0]

    assert "outbox" in call_args[0]
    assert "inbox" in call_args[0]
    assert call_args[0]["outbox"] == client.outbox
    assert call_args[0]["inbox"] == client.inbox

    assert isinstance(call_args[1], contract_pb2.Action)
    assert call_args[1].runHTTPRequest.path == expected_path
    assert call_args[1].runHTTPRequest.method == method

    actual_body = call_args[1].runHTTPRequest.body
    if body is None:
        # If body is expected to be None, accept either None or empty string
        assert actual_body in (
            None,
            "",
        ), f"Expected body to be None or empty string, got: {repr(actual_body)}"
    else:
        assert actual_body == body

    assert result == expected_result


@pytest.mark.asyncio
async def test_executor_gitlab_http_client_json_decode_error(
    client,
    monkeypatch_execute_action,
):
    # Setup non-JSON response
    invalid_json = "This is not valid JSON"
    monkeypatch_execute_action.return_value = invalid_json

    # Call with parse_json=True to trigger JSON decode error
    result = await client.aget("/api/v4/test", parse_json=True)

    # Should return the raw string when JSON parsing fails
    assert result == {}


@pytest.mark.asyncio
async def test_executor_gitlab_http_client_with_object_hook(
    client,
    monkeypatch_execute_action,
):
    # Setup a JSON string that will be decoded
    json_str = '{"key": "value", "nested": {"id": 1}}'
    monkeypatch_execute_action.return_value = json_str

    # Create a simple object hook
    def custom_hook(obj):
        if "id" in obj:
            obj["id"] = f"ID-{obj['id']}"
        return obj

    # Call with the object hook
    result = await client.aget("/api/v4/test", parse_json=True, object_hook=custom_hook)

    # Check that the hook was applied
    assert result["key"] == "value"
    assert result["nested"]["id"] == "ID-1"
