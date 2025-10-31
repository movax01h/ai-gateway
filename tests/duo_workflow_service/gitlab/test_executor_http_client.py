import asyncio
import json
from unittest.mock import AsyncMock

import pytest
from langchain_core.tools import ToolException

from contract import contract_pb2
from duo_workflow_service.executor.outbox import Outbox
from duo_workflow_service.gitlab.executor_http_client import ExecutorGitLabHttpClient
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse


@pytest.fixture(name="mock_execute_action")
def mock_execute_action_fixture():
    return AsyncMock()


@pytest.fixture(name="client")
def client_fixture():
    outbox = Outbox()
    return ExecutorGitLabHttpClient(outbox)


@pytest.fixture(name="monkeypatch_execute_action")
def monkeypatch_execute_action_fixture(monkeypatch, mock_execute_action):
    monkeypatch.setattr(
        "duo_workflow_service.gitlab.executor_http_client._execute_action",
        mock_execute_action,
    )
    return mock_execute_action


@pytest.fixture(name="mock_execute_http_response")
def mock_execute_http_response_fixture():
    return AsyncMock()


@pytest.fixture(name="monkeypatch_execute_http_response")
def monkeypatch_execute_http_response_fixture(monkeypatch, mock_execute_http_response):
    monkeypatch.setattr(
        "duo_workflow_service.gitlab.executor_http_client._execute_action_and_get_action_response",
        mock_execute_http_response,
    )
    return mock_execute_http_response


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method, path, body, params, parse_json, use_http_response, mock_return_value, expected_result",
    [
        (
            "GET",
            "/api/v4/projects/1",
            None,
            None,
            True,
            False,
            '{"key": "value"}',
            {"key": "value"},
        ),
        (
            "GET",
            "/api/v4/projects/1/jobs/102/trace",
            None,
            None,
            False,
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
            False,
            '{"projects": []}',
            {"projects": []},
        ),
        (
            "POST",
            "/api/v4/test",
            '{ "test": 1 }',
            None,
            True,
            False,
            '{"key": "value"}',
            {"key": "value"},
        ),
        (
            "PUT",
            "/api/v4/test",
            '{ "test": 1 }',
            None,
            True,
            True,
            '{"key": "value"}',
            GitLabHttpResponse(status_code=200, body={"key": "value"}),
        ),
        (
            "PATCH",
            "/api/v4/test",
            '{ "test": 1 }',
            None,
            True,
            True,
            '{"key": "value"}',
            GitLabHttpResponse(status_code=200, body={"key": "value"}),
        ),
    ],
)
async def test_executor_gitlab_http_client(
    client,
    monkeypatch_execute_action,
    monkeypatch_execute_http_response,
    method,
    path,
    body,
    params,
    parse_json,
    use_http_response,
    mock_return_value,
    expected_result,
):
    if use_http_response:
        # When use_http_response=True, we need to mock _execute_action_and_get_action_response
        # and return a proper ActionResponse object
        action_response = contract_pb2.ActionResponse()
        action_response.httpResponse.statusCode = 200
        action_response.httpResponse.body = mock_return_value
        monkeypatch_execute_http_response.return_value = action_response
    else:
        # When use_http_response=False, use the regular _execute_action mock
        monkeypatch_execute_action.return_value = mock_return_value

    expected_path = path
    if params:
        from urllib.parse import urlencode

        query_string = urlencode(params)
        expected_path = f"{path}?{query_string}"

    if method == "GET":
        result = await client.aget(
            path,
            params=params,
            parse_json=parse_json,
            use_http_response=use_http_response,
        )
    elif method == "POST":
        result = await client.apost(
            path, body, parse_json=parse_json, use_http_response=use_http_response
        )
    elif method == "PUT":
        result = await client.aput(
            path, body, parse_json=parse_json, use_http_response=use_http_response
        )
    elif method == "PATCH":
        result = await client.apatch(
            path, body, parse_json=parse_json, use_http_response=use_http_response
        )
    else:
        pytest.fail(f"Unexpected HTTP method: {method}")
        result = None

    if use_http_response:
        # When use_http_response=True, check the _execute_action_and_get_action_response mock
        monkeypatch_execute_http_response.assert_called_once()
        call_args = monkeypatch_execute_http_response.call_args[0]
    else:
        # When use_http_response=False, check the _execute_action mock
        monkeypatch_execute_action.assert_called_once()
        call_args = monkeypatch_execute_action.call_args[0]

    assert "outbox" in call_args[0]
    assert call_args[0]["outbox"] == client.outbox

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

    if use_http_response:
        # When use_http_response=True, we expect a GitLabHttpResponse object
        assert isinstance(result, GitLabHttpResponse)
        assert isinstance(expected_result, GitLabHttpResponse)
        assert result.status_code == expected_result.status_code
        assert result.body == expected_result.body
    else:
        # When use_http_response=False, we expect the raw data
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


@pytest.mark.asyncio
async def test_graphql_basic_query(client, monkeypatch_execute_action):
    mock_response = json.dumps(
        {
            "data": {
                "group": {
                    "name": "Test Group",
                    "projects": {
                        "nodes": [
                            {"id": "gid://gitlab/Project/1", "name": "Project 1"},
                            {"id": "gid://gitlab/Project/2", "name": "Project 2"},
                        ]
                    },
                }
            }
        }
    )
    monkeypatch_execute_action.return_value = mock_response

    query = """
    query GetGroupProjects($fullPath: ID!) {
        group(fullPath: $fullPath) {
            name
            projects {
                nodes {
                    id
                    name
                }
            }
        }
    }
    """
    variables = {"fullPath": "test-group"}

    result = await client.graphql(query, variables)

    assert result["group"]["name"] == "Test Group"
    assert len(result["group"]["projects"]["nodes"]) == 2
    assert result["group"]["projects"]["nodes"][0]["name"] == "Project 1"

    monkeypatch_execute_action.assert_called_once()
    call_args = monkeypatch_execute_action.call_args[0]

    assert call_args[0]["outbox"] == client.outbox

    http_request = call_args[1].runHTTPRequest
    assert http_request.path == "/api/graphql"
    assert http_request.method == "POST"

    payload = json.loads(http_request.body)
    assert payload["query"] == query
    assert payload["variables"] == variables


@pytest.mark.asyncio
async def test_graphql_without_variables(client, monkeypatch_execute_action):
    mock_response = json.dumps(
        {
            "data": {
                "currentUser": {"username": "test-user", "email": "test@example.com"}
            }
        }
    )
    monkeypatch_execute_action.return_value = mock_response

    query = """
    query {
        currentUser {
            username
            email
        }
    }
    """

    result = await client.graphql(query)

    assert result["currentUser"]["username"] == "test-user"
    assert result["currentUser"]["email"] == "test@example.com"

    monkeypatch_execute_action.assert_called_once()
    call_args = monkeypatch_execute_action.call_args[0]

    http_request = call_args[1].runHTTPRequest
    assert http_request.path == "/api/graphql"
    assert http_request.method == "POST"

    payload = json.loads(http_request.body)
    assert payload["query"] == query
    assert payload["variables"] == {}  # Empty variables object


@pytest.mark.asyncio
async def test_graphql_invalid_json_response(client, monkeypatch_execute_action):
    monkeypatch_execute_action.return_value = "This is not valid JSON"

    # Define query
    query = """
    query {
        currentUser {
            username
        }
    }
    """

    with pytest.raises(Exception) as excinfo:
        await client.graphql(query)

    assert "Invalid JSON response from GraphQL" in str(excinfo.value)


@pytest.mark.asyncio
async def test_graphql_with_errors(client, monkeypatch_execute_action):
    mock_response = json.dumps(
        {
            "errors": [
                {
                    "message": "Access denied",
                    "locations": [{"line": 2, "column": 3}],
                    "path": ["group"],
                }
            ],
            "data": None,
        }
    )
    monkeypatch_execute_action.return_value = mock_response

    query = """
    query {
        group(fullPath: "private-group") {
            name
        }
    }
    """

    with pytest.raises(Exception) as excinfo:
        await client.graphql(query)

    assert "GraphQL errors" in str(excinfo.value)
    assert "Access denied" in str(excinfo.value)


@pytest.mark.asyncio
async def test_executor_gitlab_http_client_with_use_http_response_success(
    client, monkeypatch_execute_http_response
):
    action_response = contract_pb2.ActionResponse()
    action_response.httpResponse.statusCode = 200
    action_response.httpResponse.body = '{"key": "value"}'

    monkeypatch_execute_http_response.return_value = action_response

    result = await client.aget("/api/v4/test", use_http_response=True, parse_json=True)

    expected_response = GitLabHttpResponse(
        status_code=200,
        body={"key": "value"},
    )

    assert isinstance(result, GitLabHttpResponse)
    assert result.status_code == expected_response.status_code
    assert result.body == expected_response.body

    monkeypatch_execute_http_response.assert_called_once()


@pytest.mark.asyncio
async def test_executor_gitlab_http_client_with_use_http_response_http_connection_error(
    client, monkeypatch_execute_http_response
):
    monkeypatch_execute_http_response.side_effect = ToolException("Connection refused")

    with pytest.raises(ToolException, match="Connection refused"):
        await client.aget("/api/v4/test", use_http_response=True)

    monkeypatch_execute_http_response.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "side_effects, expected_attempts, expected_sleep_calls, should_succeed, expected_result",
    [
        # Single retry - error then success
        (
            [
                Exception("Temporary error"),
                json.dumps({"data": {"currentUser": {"username": "test-user"}}}),
            ],
            2,
            1,
            True,
            {"currentUser": {"username": "test-user"}},
        ),
        # Two retries - two timeouts then success
        (
            [
                asyncio.TimeoutError("Request timed out"),
                asyncio.TimeoutError("Request timed out"),
                json.dumps({"data": {"project": {"name": "Test Project"}}}),
            ],
            3,
            2,
            True,
            {"project": {"name": "Test Project"}},
        ),
        # Invalid JSON then success
        (
            [
                "Not valid JSON",
                json.dumps({"data": {"currentUser": {"username": "test-user"}}}),
            ],
            2,
            1,
            True,
            {"currentUser": {"username": "test-user"}},
        ),
        # GraphQL error then success
        (
            [
                json.dumps(
                    {
                        "errors": [{"message": "Service temporarily unavailable"}],
                        "data": None,
                    }
                ),
                json.dumps({"data": {"currentUser": {"username": "test-user"}}}),
            ],
            2,
            1,
            True,
            {"currentUser": {"username": "test-user"}},
        ),
    ],
)
async def test_graphql_retry_succeeds(
    client,
    monkeypatch_execute_action,
    monkeypatch,
    side_effects,
    expected_attempts,
    expected_sleep_calls,
    should_succeed,
    expected_result,
):
    """Test that graphql retries on various errors and eventually succeeds."""
    # Mock asyncio.sleep to avoid actual delays in tests
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)

    monkeypatch_execute_action.side_effect = side_effects

    query = """
    query {
        currentUser {
            username
        }
    }
    """

    result = await client.graphql(query)

    assert result == expected_result
    assert monkeypatch_execute_action.call_count == expected_attempts
    assert sleep_mock.call_count == expected_sleep_calls
    if expected_sleep_calls > 0:
        sleep_mock.assert_called_with(client.RETRY_DELAY_SECONDS)


@pytest.mark.asyncio
async def test_graphql_retry_exhausted_raises_exception(
    client, monkeypatch_execute_action, monkeypatch
):
    """Test that graphql raises exception after exhausting all retries."""
    # Mock asyncio.sleep to avoid actual delays in tests
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)

    # All attempts fail
    monkeypatch_execute_action.side_effect = Exception("Persistent error")

    query = """
    query {
        currentUser {
            username
        }
    }
    """

    with pytest.raises(Exception, match="Persistent error"):
        await client.graphql(query)

    assert monkeypatch_execute_action.call_count == client.MAX_RETRIES
    assert sleep_mock.call_count == client.MAX_RETRIES - 1


@pytest.mark.asyncio
async def test_graphql_no_retry_on_immediate_success(
    client, monkeypatch_execute_action, monkeypatch
):
    """Test that graphql does not retry when the first attempt succeeds."""
    # Mock asyncio.sleep to verify it's not called
    sleep_mock = AsyncMock()
    monkeypatch.setattr("asyncio.sleep", sleep_mock)

    success_response = json.dumps({"data": {"currentUser": {"username": "test-user"}}})

    monkeypatch_execute_action.return_value = success_response

    query = """
    query {
        currentUser {
            username
        }
    }
    """

    result = await client.graphql(query)

    assert result["currentUser"]["username"] == "test-user"
    monkeypatch_execute_action.assert_called_once()
    sleep_mock.assert_not_called()
