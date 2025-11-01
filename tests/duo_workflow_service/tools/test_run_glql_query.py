import json
from unittest.mock import AsyncMock, patch

import pytest

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.run_glql_query import GLQLQueryInput, RunGLQLQuery


@pytest.fixture
def mock_gitlab_client():
    """Fixture providing a mocked GitLab client."""
    client = AsyncMock()
    client.apost = AsyncMock()
    return client


@pytest.fixture
def glql_tool(mock_gitlab_client):
    """Fixture providing a RunGLQLQuery tool instance."""
    return RunGLQLQuery(metadata={"gitlab_client": mock_gitlab_client})


@pytest.fixture
def mock_version_18_6():
    """Fixture that mocks GitLab version as 18.6.0."""
    with patch("duo_workflow_service.tools.run_glql_query.gitlab_version") as mock:
        mock.get.return_value = "18.6.0"
        yield mock


@pytest.fixture
def sample_glql_query():
    """Sample GLQL query in markdown format."""
    return """```glql
display: table
fields: title, state
sort: created desc
limit: 50
query: type = Issue
```"""


@pytest.mark.asyncio
async def test_successful_query_returns_data(
    glql_tool, mock_gitlab_client, mock_version_18_6, sample_glql_query
):
    """Test successful GLQL query returns expected data."""
    expected_data = {
        "data": {
            "count": 2,
            "nodes": [
                {"id": "gid://gitlab/Issue/123", "title": "Test Issue 1"},
                {"id": "gid://gitlab/Issue/124", "title": "Test Issue 2"},
            ],
        }
    }
    mock_gitlab_client.apost.return_value = GitLabHttpResponse(
        status_code=200, body=expected_data
    )

    response = await glql_tool.arun({"glql_yaml": sample_glql_query})
    parsed = json.loads(response)

    assert parsed == expected_data
    assert parsed["data"]["count"] == 2
    assert len(parsed["data"]["nodes"]) == 2


@pytest.mark.asyncio
async def test_request_body_contains_glql_yaml(
    glql_tool, mock_gitlab_client, mock_version_18_6, sample_glql_query
):
    """Test that the request body correctly includes the GLQL YAML."""
    mock_gitlab_client.apost.return_value = GitLabHttpResponse(
        status_code=200, body={"data": {"count": 0, "nodes": []}}
    )

    await glql_tool.arun({"glql_yaml": sample_glql_query})

    body = json.loads(mock_gitlab_client.apost.call_args.kwargs["body"])
    expected_yaml = """```glql
display: table
fields: title, state
sort: created desc
limit: 50
query: type = Issue
```"""

    assert body == {"glql_yaml": expected_yaml}


@pytest.mark.asyncio
async def test_api_error_returns_error_response(
    glql_tool, mock_gitlab_client, mock_version_18_6
):
    """Test that API errors are properly captured and returned."""
    mock_gitlab_client.apost.return_value = GitLabHttpResponse(
        status_code=500, body={"message": "Internal server error"}
    )

    response = await glql_tool.arun({"glql_yaml": "```glql\nquery: type = Issue\n```"})
    parsed = json.loads(response)

    assert "error" in parsed
    assert "500" in parsed["error"]


@pytest.mark.asyncio
async def test_connection_exception_propagates(
    glql_tool, mock_gitlab_client, mock_version_18_6
):
    """Test that connection exceptions are raised."""
    mock_gitlab_client.apost.side_effect = Exception("Connection timeout")

    with pytest.raises(Exception, match="Connection timeout"):
        await glql_tool.arun({"glql_yaml": "```glql\nquery: type = Issue\n```"})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "version,should_succeed",
    [
        ("18.5.0", False),
        ("18.6.0", True),
        ("18.7.0", True),
        ("19.0.0", True),
        (None, False),
        ("invalid", False),
    ],
)
async def test_version_check(version, should_succeed, glql_tool, mock_gitlab_client):
    """Test that GLQL query only works with GitLab 18.6+."""
    mock_gitlab_client.apost.return_value = GitLabHttpResponse(
        status_code=200, body={"data": {"count": 0, "nodes": []}}
    )

    with patch(
        "duo_workflow_service.tools.run_glql_query.gitlab_version"
    ) as mock_version:
        mock_version.get.return_value = version

        response = await glql_tool.arun(
            {"glql_yaml": "```glql\nquery: type = Issue\n```"}
        )
        parsed = json.loads(response)

        if should_succeed:
            assert (
                "error" not in parsed
                or "GLQL API is only available" not in parsed.get("error", "")
            )
            mock_gitlab_client.apost.assert_called_once()
        else:
            assert "error" in parsed
            assert (
                "GLQL API is only available in GitLab 18.6 and later" in parsed["error"]
            )
            mock_gitlab_client.apost.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_query,api_error_message",
    [
        (
            "no markdown wrapper",
            "400 Bad request - Error: Unexpected `markdown wrapper`, expected operator (one of IN, =, !=, >, or <)",
        ),
        (
            "```glql\n\n```",
            "400 Bad request - Error: Unexpected token near ````glql\n\n````",
        ),
        (
            "```glql\ninvalid: yaml: :\n```",
            "400 Bad request - Error: Unexpected token near ````glql\n\ninvalid: yaml: :\n````",
        ),
    ],
)
async def test_invalid_query_format_handled_by_api(
    glql_tool, mock_gitlab_client, mock_version_18_6, invalid_query, api_error_message
):
    """Test that invalid query formats are sent to API and API errors are returned."""
    # Mock API returning validation error
    mock_gitlab_client.apost.return_value = GitLabHttpResponse(
        status_code=400, body={"error": api_error_message}
    )

    response = await glql_tool.arun({"glql_yaml": invalid_query})
    parsed = json.loads(response)

    # Verify the query was sent to the API
    mock_gitlab_client.apost.assert_called_once()

    # Verify error response is returned
    assert "error" in parsed
    assert "400" in parsed["error"]


def test_format_display_message():
    """Test display message formatting."""
    tool = RunGLQLQuery(metadata={})
    query = GLQLQueryInput(glql_yaml="```glql\nquery: type = Issue\n```")

    assert tool.format_display_message(query) == "Execute GLQL query"


def test_tool_properties():
    """Test tool has correct name, description, and schema."""
    tool = RunGLQLQuery(metadata={})

    assert tool.name == "run_glql_query"
    assert "GLQL" in tool.description
    assert "GitLab Query Language" in tool.description
    assert tool.args_schema == GLQLQueryInput
