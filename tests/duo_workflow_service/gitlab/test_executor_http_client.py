# pylint: disable=import-outside-toplevel
import asyncio
import json
from unittest.mock import AsyncMock, patch

import pytest
from langchain_core.tools import ToolException

from contract import contract_pb2
from duo_workflow_service.executor.outbox import Outbox
from duo_workflow_service.gitlab.executor_http_client import (
    ExecutorGitLabHttpClient,
    _is_retryable_error,
    _RetryableStatusError,
)
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse


@pytest.fixture(autouse=True, name="mock_tenacity_sleep")
def mock_tenacity_sleep_fixture():
    """Patch asyncio.sleep to return immediately for retry wait tests.

    Retry tests back off by 3s/9s/27s. Tenacity's async sleep path calls asyncio.sleep() via a lazy import inside
    _portable_async_sleep, so patching asyncio.sleep directly is the correct interception point. This drops each retry-
    triggering test from ~39s to <5ms without affecting retry logic correctness.

    Yields the mock so tests can assert on the backoff schedule.
    """
    sleep_mock = AsyncMock(return_value=None)
    with patch("asyncio.sleep", new=sleep_mock):
        yield sleep_mock


def http_action_response(body: str, status_code: int = 200):
    """Build the ActionResponse an executor returns for a runHTTPRequest action."""
    action_response = contract_pb2.ActionResponse()
    action_response.httpResponse.statusCode = status_code
    action_response.httpResponse.body = body
    return action_response


def plain_text_action_response(response: str):
    """Build the ActionResponse returned by executors that answer with plain text."""
    action_response = contract_pb2.ActionResponse()
    action_response.plainTextResponse.response = response
    return action_response


@pytest.fixture(name="client")
def client_fixture():
    outbox = Outbox()
    return ExecutorGitLabHttpClient(outbox)


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
    "method, path, body, params, parse_json, mock_return_value, expected_body",
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
    monkeypatch_execute_http_response,
    method,
    path,
    body,
    params,
    parse_json,
    mock_return_value,
    expected_body,
):
    # ExecutorGitLabHttpClient always returns GitLabHttpResponse
    action_response = contract_pb2.ActionResponse()
    action_response.httpResponse.statusCode = 200
    action_response.httpResponse.body = mock_return_value
    monkeypatch_execute_http_response.return_value = action_response

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
        )
    elif method == "POST":
        result = await client.apost(path, body, parse_json=parse_json)
    elif method == "PUT":
        result = await client.aput(path, body, parse_json=parse_json)
    elif method == "PATCH":
        result = await client.apatch(path, body, parse_json=parse_json)
    else:
        pytest.fail(f"Unexpected HTTP method: {method}")
        result = None

    # Verify the action was called correctly
    monkeypatch_execute_http_response.assert_called_once()
    call_args = monkeypatch_execute_http_response.call_args[0]

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
        ), f"Expected body to be None or empty string, got: {actual_body!r}"
    else:
        assert actual_body == body

    # ExecutorGitLabHttpClient always returns GitLabHttpResponse
    assert isinstance(result, GitLabHttpResponse)
    assert result.status_code == 200
    assert result.body == expected_body


@pytest.mark.asyncio
async def test_executor_gitlab_http_client_json_decode_error(
    client,
    monkeypatch_execute_http_response,
):
    # Setup non-JSON response
    invalid_json = "This is not valid JSON"

    action_response = contract_pb2.ActionResponse()
    action_response.httpResponse.statusCode = 200
    action_response.httpResponse.body = invalid_json
    monkeypatch_execute_http_response.return_value = action_response

    # Call with parse_json=True to trigger JSON decode error
    result = await client.aget("/api/v4/test", parse_json=True)

    # Should return empty dict when JSON parsing fails
    assert isinstance(result, GitLabHttpResponse)
    assert result.status_code == 200
    assert result.body == {}


@pytest.mark.asyncio
async def test_executor_gitlab_http_client_with_object_hook(
    client,
    monkeypatch_execute_http_response,
):
    # Setup a JSON string that will be decoded
    json_str = '{"key": "value", "nested": {"id": 1}}'

    action_response = contract_pb2.ActionResponse()
    action_response.httpResponse.statusCode = 200
    action_response.httpResponse.body = json_str
    monkeypatch_execute_http_response.return_value = action_response

    # Create a simple object hook
    def custom_hook(obj):
        if "id" in obj:
            obj["id"] = f"ID-{obj['id']}"
        return obj

    # Call with the object hook
    result = await client.aget("/api/v4/test", parse_json=True, object_hook=custom_hook)

    # Check that the hook was applied
    assert isinstance(result, GitLabHttpResponse)
    assert result.status_code == 200
    assert result.body["key"] == "value"
    assert result.body["nested"]["id"] == "ID-1"


@pytest.mark.asyncio
async def test_graphql_basic_query(client, monkeypatch_execute_http_response):
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
    monkeypatch_execute_http_response.return_value = http_action_response(mock_response)

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

    monkeypatch_execute_http_response.assert_called_once()
    call_args = monkeypatch_execute_http_response.call_args[0]

    assert call_args[0]["outbox"] == client.outbox

    http_request = call_args[1].runHTTPRequest
    assert http_request.path == "/api/graphql"
    assert http_request.method == "POST"

    payload = json.loads(http_request.body)
    assert payload["query"] == query
    assert payload["variables"] == variables


@pytest.mark.asyncio
async def test_graphql_without_variables(client, monkeypatch_execute_http_response):
    mock_response = json.dumps(
        {
            "data": {
                "currentUser": {"username": "test-user", "email": "test@example.com"}
            }
        }
    )
    monkeypatch_execute_http_response.return_value = http_action_response(mock_response)

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

    monkeypatch_execute_http_response.assert_called_once()
    call_args = monkeypatch_execute_http_response.call_args[0]

    http_request = call_args[1].runHTTPRequest
    assert http_request.path == "/api/graphql"
    assert http_request.method == "POST"

    payload = json.loads(http_request.body)
    assert payload["query"] == query
    assert payload["variables"] == {}  # Empty variables object


@pytest.mark.asyncio
async def test_graphql_invalid_json_response(client, monkeypatch_execute_http_response):
    monkeypatch_execute_http_response.return_value = http_action_response(
        "This is not valid JSON"
    )

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
async def test_graphql_with_errors(client, monkeypatch_execute_http_response):
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
    monkeypatch_execute_http_response.return_value = http_action_response(mock_response)

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
async def test_executor_gitlab_http_client_success(
    client, monkeypatch_execute_http_response
):
    action_response = contract_pb2.ActionResponse()
    action_response.httpResponse.statusCode = 200
    action_response.httpResponse.body = '{"key": "value"}'

    monkeypatch_execute_http_response.return_value = action_response

    result = await client.aget("/api/v4/test", parse_json=True)

    expected_response = GitLabHttpResponse(
        status_code=200,
        body={"key": "value"},
    )

    assert isinstance(result, GitLabHttpResponse)
    assert result.status_code == expected_response.status_code
    assert result.body == expected_response.body

    monkeypatch_execute_http_response.assert_called_once()


@pytest.mark.asyncio
async def test_executor_gitlab_http_client_http_connection_error(
    client, monkeypatch_execute_http_response
):
    """Test that non-retryable ToolExceptions propagate immediately."""
    monkeypatch_execute_http_response.side_effect = ToolException("Permission denied")

    with pytest.raises(ToolException, match="Permission denied"):
        await client.aget("/api/v4/test")

    monkeypatch_execute_http_response.assert_called_once()


@pytest.mark.asyncio
async def test_http_call_retries_on_timeout_and_succeeds(
    client, monkeypatch_execute_http_response
):
    """Test that _call retries when the executor returns a timeout ToolException."""
    success_response = contract_pb2.ActionResponse()
    success_response.httpResponse.statusCode = 200
    success_response.httpResponse.body = '{"key": "value"}'

    monkeypatch_execute_http_response.side_effect = [
        ToolException("HTTP action error: request timed out"),
        success_response,
    ]

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        result = await client.aget("/api/v4/test")

    assert result.status_code == 200
    assert result.body == {"key": "value"}
    assert monkeypatch_execute_http_response.call_count == 2


@pytest.mark.asyncio
async def test_http_call_exhausts_retries_on_repeated_timeouts(
    client, monkeypatch_execute_http_response
):
    """Test that _call raises after all retry attempts are exhausted."""
    monkeypatch_execute_http_response.side_effect = ToolException(
        "HTTP action error: request timed out"
    )

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        with pytest.raises(ToolException, match="request timed out"):
            await client.aget("/api/v4/test")

    assert monkeypatch_execute_http_response.call_count == 4


@pytest.mark.asyncio
async def test_http_call_does_not_retry_non_timeout_errors(
    client, monkeypatch_execute_http_response
):
    """Test that _call does NOT retry for non-timeout errors."""
    monkeypatch_execute_http_response.side_effect = ToolException("Permission denied")

    with pytest.raises(ToolException, match="Permission denied"):
        await client.aget("/api/v4/test")

    monkeypatch_execute_http_response.assert_called_once()


@pytest.mark.asyncio
async def test_http_call_retries_on_asyncio_timeout_and_succeeds(
    client, monkeypatch_execute_http_response
):
    """Test that _call retries when the executor raises asyncio.TimeoutError directly."""
    success_response = contract_pb2.ActionResponse()
    success_response.httpResponse.statusCode = 200
    success_response.httpResponse.body = '{"key": "value"}'

    monkeypatch_execute_http_response.side_effect = [
        asyncio.TimeoutError(),
        success_response,
    ]

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        result = await client.aget("/api/v4/test")

    assert result.status_code == 200
    assert result.body == {"key": "value"}
    assert monkeypatch_execute_http_response.call_count == 2


@pytest.mark.asyncio
async def test_graphql_retries_on_timeout_and_succeeds(
    client, monkeypatch_execute_http_response
):
    """Test that graphql retries when asyncio.TimeoutError is raised."""
    success_response = json.dumps({"data": {"currentUser": {"username": "alice"}}})

    call_count = 0

    async def side_effect(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise asyncio.TimeoutError()
        return http_action_response(success_response)

    monkeypatch_execute_http_response.side_effect = side_effect

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        result = await client.graphql("{ currentUser { username } }")

    assert result["currentUser"]["username"] == "alice"
    assert call_count == 2


@pytest.mark.asyncio
async def test_graphql_exhausts_retries_on_repeated_timeouts(
    client, monkeypatch_execute_http_response
):
    """Test that graphql raises after all retry attempts are exhausted."""
    monkeypatch_execute_http_response.side_effect = asyncio.TimeoutError()

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        with pytest.raises(Exception, match="GraphQL request timed out"):
            await client.graphql("{ currentUser { username } }")

    assert monkeypatch_execute_http_response.call_count == 4


# ---------------------------------------------------------------------------
# Tests for _is_retryable_error
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc, expected",
    [
        (asyncio.TimeoutError(), True),
        (_RetryableStatusError(500, "Internal Server Error", {}), True),
        (_RetryableStatusError(503, "Service Unavailable", {}), True),
        (Exception("HTTP action error: request timed out"), True),
        (Exception("GraphQL request timed out after 10.0 seconds"), True),
        (ToolException("HTTP action error: connection refused"), True),
        (ToolException("HTTP action error: connection reset by peer"), True),
        (ToolException("HTTP action error: broken pipe"), True),
        (ToolException("HTTP action error: network unreachable"), True),
        (
            ToolException("HTTP action error: failed to establish a new connection"),
            True,
        ),
        (ToolException("Permission denied"), False),
        (ToolException("Access denied"), False),
        (ToolException("Not found"), False),
        (Exception("JSON decode error"), False),
    ],
)
def test_is_retryable_error(exc, expected):
    assert _is_retryable_error(exc) == expected


# ---------------------------------------------------------------------------
# Tests for 5xx retry behaviour in _call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_call_retries_on_500_and_succeeds(
    client, monkeypatch_execute_http_response
):
    """Test that _call retries when the executor returns a 500 status code."""
    error_response = contract_pb2.ActionResponse()
    error_response.httpResponse.statusCode = 500
    error_response.httpResponse.body = "Internal Server Error"

    success_response = contract_pb2.ActionResponse()
    success_response.httpResponse.statusCode = 200
    success_response.httpResponse.body = '{"key": "value"}'

    monkeypatch_execute_http_response.side_effect = [error_response, success_response]

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        result = await client.aget("/api/v4/test")

    assert result.status_code == 200
    assert result.body == {"key": "value"}
    assert monkeypatch_execute_http_response.call_count == 2


@pytest.mark.asyncio
async def test_http_call_retries_on_503_and_succeeds(
    client, monkeypatch_execute_http_response
):
    """Test that _call retries when the executor returns a 503 status code."""
    error_response = contract_pb2.ActionResponse()
    error_response.httpResponse.statusCode = 503
    error_response.httpResponse.body = "Service Unavailable"

    success_response = contract_pb2.ActionResponse()
    success_response.httpResponse.statusCode = 200
    success_response.httpResponse.body = '{"result": "ok"}'

    monkeypatch_execute_http_response.side_effect = [error_response, success_response]

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        result = await client.aget("/api/v4/test")

    assert result.status_code == 200
    assert result.body == {"result": "ok"}
    assert monkeypatch_execute_http_response.call_count == 2


@pytest.mark.asyncio
async def test_http_call_exhausts_retries_on_repeated_500s(
    client, monkeypatch_execute_http_response
):
    """Test that _call returns a 500 GitLabHttpResponse after all retries are exhausted."""
    error_response = contract_pb2.ActionResponse()
    error_response.httpResponse.statusCode = 500
    error_response.httpResponse.body = "Internal Server Error"

    monkeypatch_execute_http_response.return_value = error_response

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        result = await client.aget("/api/v4/test", parse_json=False)

    assert isinstance(result, GitLabHttpResponse)
    assert result.status_code == 500
    assert result.body == "Internal Server Error"
    assert monkeypatch_execute_http_response.call_count == 4


@pytest.mark.asyncio
async def test_http_call_exhausts_retries_on_repeated_500s_with_json_body(
    client, monkeypatch_execute_http_response
):
    """Test that _call returns a parsed-JSON body when parse_json=True and retries are exhausted."""
    error_response = contract_pb2.ActionResponse()
    error_response.httpResponse.statusCode = 500
    error_response.httpResponse.body = '{"error": "Internal Server Error"}'

    monkeypatch_execute_http_response.return_value = error_response

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        result = await client.aget("/api/v4/test")  # parse_json=True by default

    assert isinstance(result, GitLabHttpResponse)
    assert result.status_code == 500
    assert result.body == {"error": "Internal Server Error"}
    assert monkeypatch_execute_http_response.call_count == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [400, 403, 404, 422, 429])
async def test_http_call_does_not_retry_non_retryable_4xx_errors(
    client, monkeypatch_execute_http_response, status_code
):
    """Test that _call does NOT retry 4xx client errors other than 401."""
    error_response = contract_pb2.ActionResponse()
    error_response.httpResponse.statusCode = status_code
    error_response.httpResponse.body = '{"message": "Client error"}'

    monkeypatch_execute_http_response.return_value = error_response

    result = await client.aget("/api/v4/test")

    assert result.status_code == status_code
    monkeypatch_execute_http_response.assert_called_once()


@pytest.mark.asyncio
async def test_http_call_retries_on_401_and_succeeds(
    client, monkeypatch_execute_http_response
):
    """_call retries a 401, which is transient while a read replica lags behind the primary.

    The same OAuth token succeeds once the replica catches up, so retrying turns a failed checkpoint or audit-event POST
    into a successful one.
    """
    unauthorized_response = contract_pb2.ActionResponse()
    unauthorized_response.httpResponse.statusCode = 401
    unauthorized_response.httpResponse.body = '{"message": "401 Unauthorized"}'

    success_response = contract_pb2.ActionResponse()
    success_response.httpResponse.statusCode = 201
    success_response.httpResponse.body = '{"id": 1}'

    monkeypatch_execute_http_response.side_effect = [
        unauthorized_response,
        success_response,
    ]

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        result = await client.apost(
            "/api/v4/ai/duo_workflows/workflows/1/checkpoints", body="{}"
        )

    assert result.status_code == 201
    assert result.body == {"id": 1}
    assert monkeypatch_execute_http_response.call_count == 2


@pytest.mark.asyncio
async def test_http_call_exhausts_retries_on_repeated_401s(
    client, monkeypatch_execute_http_response
):
    """A genuinely expired or revoked token keeps returning 401 and is surfaced to the caller."""
    unauthorized_response = contract_pb2.ActionResponse()
    unauthorized_response.httpResponse.statusCode = 401
    unauthorized_response.httpResponse.body = '{"message": "401 Unauthorized"}'

    monkeypatch_execute_http_response.return_value = unauthorized_response

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        result = await client.aget("/api/v4/test")

    assert isinstance(result, GitLabHttpResponse)
    assert result.status_code == 401
    assert result.body == {"message": "401 Unauthorized"}
    assert monkeypatch_execute_http_response.call_count == 4


# ---------------------------------------------------------------------------
# Tests for network-error retry behaviour in _call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_http_call_retries_on_connection_refused_and_succeeds(
    client, monkeypatch_execute_http_response
):
    """Test that _call retries when a connection-refused ToolException is raised."""
    success_response = contract_pb2.ActionResponse()
    success_response.httpResponse.statusCode = 200
    success_response.httpResponse.body = '{"key": "value"}'

    monkeypatch_execute_http_response.side_effect = [
        ToolException("HTTP action error: connection refused"),
        success_response,
    ]

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        result = await client.aget("/api/v4/test")

    assert result.status_code == 200
    assert result.body == {"key": "value"}
    assert monkeypatch_execute_http_response.call_count == 2


@pytest.mark.asyncio
async def test_http_call_exhausts_retries_on_repeated_network_errors(
    client, monkeypatch_execute_http_response
):
    """Test that _call raises after all retries are exhausted on network errors."""
    monkeypatch_execute_http_response.side_effect = ToolException(
        "HTTP action error: connection reset by peer"
    )

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        with pytest.raises(ToolException, match="connection reset by peer"):
            await client.aget("/api/v4/test")

    assert monkeypatch_execute_http_response.call_count == 4


# ---------------------------------------------------------------------------
# Tests for network-error retry behaviour in graphql
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_graphql_retries_on_network_error_and_succeeds(
    client, monkeypatch_execute_http_response
):
    """Test that graphql retries when a network ToolException is raised."""
    success_response = json.dumps({"data": {"currentUser": {"username": "bob"}}})

    call_count = 0

    async def side_effect(*_args, **_kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ToolException("HTTP action error: connection refused")
        return http_action_response(success_response)

    monkeypatch_execute_http_response.side_effect = side_effect

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        result = await client.graphql("{ currentUser { username } }")

    assert result["currentUser"]["username"] == "bob"
    assert call_count == 2


@pytest.mark.asyncio
async def test_graphql_exhausts_retries_on_repeated_network_errors(
    client, monkeypatch_execute_http_response
):
    """Test that graphql raises after all retries are exhausted on network errors."""
    monkeypatch_execute_http_response.side_effect = ToolException(
        "HTTP action error: connection reset by peer"
    )

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        with pytest.raises(ToolException, match="connection reset by peer"):
            await client.graphql("{ currentUser { username } }")

    assert monkeypatch_execute_http_response.call_count == 4


# ---------------------------------------------------------------------------
# Tests for HTTP-status retry behaviour in graphql
#
# GitLab answers `/api/graphql` with HTTP 401 and a body of
# `{"errors":[{"message":"Invalid token"}]}` when a Duo Agent Platform OAuth
# token is temporarily invisible to a lagging Postgres read replica.  Without a
# retry this surfaced as an unrecoverable `GraphQL errors: [...]` and killed the
# whole flow.
# ---------------------------------------------------------------------------


INVALID_TOKEN_BODY = json.dumps({"errors": [{"message": "Invalid token"}]})


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", [401, 500, 503])
async def test_graphql_retries_on_retryable_status_and_succeeds(
    client, monkeypatch_execute_http_response, status_code
):
    """Graphql retries retryable HTTP statuses and returns the eventual success."""
    monkeypatch_execute_http_response.side_effect = [
        http_action_response(INVALID_TOKEN_BODY, status_code=status_code),
        http_action_response(
            json.dumps({"data": {"currentUser": {"username": "carol"}}})
        ),
    ]

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        result = await client.graphql("{ currentUser { username } }")

    assert result["currentUser"]["username"] == "carol"
    assert monkeypatch_execute_http_response.call_count == 2


@pytest.mark.asyncio
async def test_graphql_exhausts_retries_on_repeated_401s(
    client, monkeypatch_execute_http_response
):
    """Graphql surfaces the HTTP status once the retry budget is spent."""
    monkeypatch_execute_http_response.return_value = http_action_response(
        INVALID_TOKEN_BODY, status_code=401
    )

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        with pytest.raises(Exception, match="GraphQL request failed with HTTP 401"):
            await client.graphql("{ currentUser { username } }")

    assert monkeypatch_execute_http_response.call_count == 4


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body, expected_message",
    [
        (
            json.dumps(
                {
                    "errors": [
                        {
                            "message": "Workflow not found",
                            "extensions": {"code": "WORKFLOW_NOT_FOUND"},
                        }
                    ]
                }
            ),
            "Workflow not found",
        ),
        (
            json.dumps(
                {"errors": [{"message": "You don't have permission to access this"}]}
            ),
            "You don't have permission",
        ),
    ],
)
async def test_graphql_does_not_retry_errors_returned_with_http_200(
    client, monkeypatch_execute_http_response, body, expected_message
):
    """GraphQL-level errors come back with HTTP 200 and must fail fast, not retry."""
    monkeypatch_execute_http_response.return_value = http_action_response(body)

    with pytest.raises(Exception) as excinfo:
        await client.graphql("{ currentUser { username } }")

    assert expected_message in str(excinfo.value)
    monkeypatch_execute_http_response.assert_called_once()


@pytest.mark.asyncio
async def test_graphql_does_not_retry_non_retryable_status(
    client, monkeypatch_execute_http_response
):
    """A 403 is a genuine authorization failure and must not be retried."""
    monkeypatch_execute_http_response.return_value = http_action_response(
        json.dumps({"errors": [{"message": "Forbidden"}]}), status_code=403
    )

    with pytest.raises(Exception, match="GraphQL errors"):
        await client.graphql("{ currentUser { username } }")

    monkeypatch_execute_http_response.assert_called_once()


@pytest.mark.asyncio
async def test_graphql_accepts_plain_text_response(
    client, monkeypatch_execute_http_response
):
    """Executors that answer runHTTPRequest with plainTextResponse carry no status code."""
    monkeypatch_execute_http_response.return_value = plain_text_action_response(
        json.dumps({"data": {"currentUser": {"username": "dave"}}})
    )

    result = await client.graphql("{ currentUser { username } }")

    assert result["currentUser"]["username"] == "dave"


@pytest.mark.asyncio
async def test_graphql_raises_when_response_type_missing(
    client, monkeypatch_execute_http_response
):
    """An ActionResponse with neither response type is a protocol violation."""
    monkeypatch_execute_http_response.return_value = contract_pb2.ActionResponse()

    with pytest.raises(ToolException, match="expected response fields"):
        await client.graphql("{ currentUser { username } }")


@pytest.mark.asyncio
async def test_retry_backoff_schedule_covers_replica_lag_window(
    client, monkeypatch_execute_http_response, mock_tenacity_sleep
):
    """The retry budget must be 3s/9s/27s across 4 attempts.

    GitLab pins a freshly minted OAuth token to the database primary for only 30s
    (Gitlab::Database::LoadBalancing::Sticking::EXPIRATION). These waits add 39s on top of that, so a
    replica lagging by up to ~69s no longer fails the flow.
    """
    monkeypatch_execute_http_response.return_value = http_action_response(
        INVALID_TOKEN_BODY, status_code=401
    )

    with patch("duo_workflow_service.gitlab.executor_http_client.logger"):
        with pytest.raises(Exception, match="GraphQL request failed with HTTP 401"):
            await client.graphql("{ currentUser { username } }")

    waits = [call.args[0] for call in mock_tenacity_sleep.call_args_list]

    assert waits == [3, 9, 27]
    assert monkeypatch_execute_http_response.call_count == 4
