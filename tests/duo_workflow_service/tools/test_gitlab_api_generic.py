import json
from unittest.mock import AsyncMock, Mock
from urllib.parse import urlencode

import pytest

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.gitlab_api_generic import (
    GitLabApiGet,
    GitLabApiGetInput,
    GitLabGraphQL,
    GitLabGraphQLInput,
)


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    """Fixture for GitLab client mock."""
    return Mock()


@pytest.fixture(name="gitlab_api_get_tool")
def gitlab_api_get_tool_fixture(gitlab_client_mock):
    """Fixture for GitLabApiGet tool."""
    tool = GitLabApiGet()
    tool.metadata = {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "https://gitlab.com",
    }
    return tool


@pytest.fixture(name="gitlab_graphql_tool")
def gitlab_graphql_tool_fixture(gitlab_client_mock):
    """Fixture for GitLabGraphQL tool."""
    tool = GitLabGraphQL()
    tool.metadata = {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "https://gitlab.com",
    }
    return tool


class TestGitLabApiGet:
    """Tests for GitLabApiGet tool."""

    @pytest.mark.asyncio
    async def test_get_with_endpoint_success(
        self, gitlab_api_get_tool, gitlab_client_mock
    ):
        """Test successful API GET request with endpoint."""
        # Mock response
        response_data = {"id": 1, "title": "Test MR"}
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = True
        mock_response.body = response_data
        gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

        # Execute tool
        result = await gitlab_api_get_tool._execute(
            endpoint="/api/v4/projects/13/merge_requests/42"
        )

        # Verify
        gitlab_client_mock.aget.assert_called_once_with(
            path="/api/v4/projects/13/merge_requests/42"
        )
        result_json = json.loads(result)
        assert result_json["status"] == "success"
        assert result_json["data"] == response_data

    @pytest.mark.asyncio
    async def test_get_with_endpoint_and_params(
        self, gitlab_api_get_tool, gitlab_client_mock
    ):
        """Test API GET request with query parameters."""
        # Mock response
        response_data = [{"id": 1, "title": "MR 1"}, {"id": 2, "title": "MR 2"}]
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = True
        mock_response.body = response_data
        gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

        # Execute tool
        params = {"state": "opened", "per_page": 20}
        result = await gitlab_api_get_tool._execute(
            endpoint="/api/v4/projects/13/merge_requests", params=params
        )

        # Verify
        expected_query = "?" + urlencode(params, doseq=True)
        expected_path = f"/api/v4/projects/13/merge_requests{expected_query}"
        gitlab_client_mock.aget.assert_called_once_with(path=expected_path)
        result_json = json.loads(result)
        assert result_json["status"] == "success"
        assert result_json["data"] == response_data

    @pytest.mark.asyncio
    async def test_get_error_no_endpoint(self, gitlab_api_get_tool):
        """Test error when endpoint is not provided."""
        result = await gitlab_api_get_tool._execute()
        result_json = json.loads(result)
        assert "error" in result_json
        assert "The 'endpoint' parameter must be provided" in result_json["error"]

    @pytest.mark.asyncio
    async def test_get_error_invalid_endpoint_format(self, gitlab_api_get_tool):
        """Test error when endpoint doesn't start with /api/v4/."""
        result = await gitlab_api_get_tool._execute(endpoint="/invalid/endpoint")
        result_json = json.loads(result)
        assert "error" in result_json
        assert "Invalid endpoint format" in result_json["error"]

    @pytest.mark.asyncio
    async def test_get_error_api_failure(self, gitlab_api_get_tool, gitlab_client_mock):
        """Test handling of API failure response."""
        # Mock error response
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = False
        mock_response.status_code = 404
        mock_response.body = {"error": "Not found"}
        gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

        # Execute tool
        result = await gitlab_api_get_tool._execute(
            endpoint="/api/v4/projects/13/merge_requests/999"
        )

        # Verify
        result_json = json.loads(result)
        assert "error" in result_json
        assert result_json["status_code"] == 404

    @pytest.mark.asyncio
    async def test_get_exception_handling(
        self, gitlab_api_get_tool, gitlab_client_mock
    ):
        """Test handling of exceptions during API call."""
        # Mock exception
        gitlab_client_mock.aget = AsyncMock(side_effect=Exception("Connection error"))

        # Execute tool
        result = await gitlab_api_get_tool._execute(
            endpoint="/api/v4/projects/13/merge_requests/42"
        )

        # Verify
        result_json = json.loads(result)
        assert "error" in result_json
        assert "Connection error" in result_json["details"]

    def test_format_display_message_with_endpoint(self, gitlab_api_get_tool):
        """Test display message formatting with endpoint."""
        args = GitLabApiGetInput(endpoint="/api/v4/projects/13/merge_requests/42")
        message = gitlab_api_get_tool.format_display_message(args)
        assert "/api/v4/projects/13/merge_requests/42" in message


class TestGitLabGraphQL:
    """Tests for GitLabGraphQL tool."""

    @pytest.mark.asyncio
    async def test_graphql_query_success(self, gitlab_graphql_tool, gitlab_client_mock):
        """Test successful GraphQL query."""
        # Mock response
        response_data = {
            "data": {
                "project": {"name": "Test Project", "description": "Test description"}
            }
        }
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = True
        mock_response.body = response_data
        gitlab_client_mock.apost = AsyncMock(return_value=mock_response)

        # Execute tool
        query = """
        query($projectPath: ID!) {
            project(fullPath: $projectPath) {
                name
                description
            }
        }
        """
        variables = {"projectPath": "namespace/project"}
        result = await gitlab_graphql_tool._execute(query=query, variables=variables)

        # Verify
        gitlab_client_mock.apost.assert_called_once()
        call_args = gitlab_client_mock.apost.call_args
        assert call_args[1]["path"] == "/api/graphql"

        payload = json.loads(call_args[1]["body"])
        assert "query" in payload
        assert payload["variables"] == variables

        result_json = json.loads(result)
        assert result_json["status"] == "success"
        assert result_json["data"]["project"]["name"] == "Test Project"

    @pytest.mark.asyncio
    async def test_graphql_query_without_variables(
        self, gitlab_graphql_tool, gitlab_client_mock
    ):
        """Test GraphQL query without variables."""
        # Mock response
        response_data = {"data": {"currentUser": {"username": "testuser"}}}
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = True
        mock_response.body = response_data
        gitlab_client_mock.apost = AsyncMock(return_value=mock_response)

        # Execute tool
        query = "query { currentUser { username } }"
        result = await gitlab_graphql_tool._execute(query=query)

        # Verify
        gitlab_client_mock.apost.assert_called_once()
        call_args = gitlab_client_mock.apost.call_args
        payload = json.loads(call_args[1]["body"])
        assert "query" in payload
        assert "variables" not in payload

        result_json = json.loads(result)
        assert result_json["status"] == "success"

    @pytest.mark.asyncio
    async def test_graphql_mutation_blocked(self, gitlab_graphql_tool):
        """Test that mutations are blocked."""
        # Execute tool with mutation
        mutation = """
        mutation($input: UpdateIssueInput!) {
            updateIssue(input: $input) {
                issue { title }
            }
        }
        """
        result = await gitlab_graphql_tool._execute(query=mutation)

        # Verify
        result_json = json.loads(result)
        assert "error" in result_json
        assert "mutations and subscriptions are not allowed" in result_json["error"]

    @pytest.mark.asyncio
    async def test_graphql_subscription_blocked(self, gitlab_graphql_tool):
        """Test that subscriptions are blocked."""
        # Execute tool with subscription
        subscription = """
        subscription($projectPath: ID!) {
            mergeRequestUpdated(projectPath: $projectPath) {
                id
                title
                state
            }
        }
        """
        result = await gitlab_graphql_tool._execute(query=subscription)

        # Verify
        result_json = json.loads(result)
        assert "error" in result_json
        assert "mutations and subscriptions are not allowed" in result_json["error"]

    @pytest.mark.asyncio
    async def test_graphql_subscription_with_comments_blocked(
        self, gitlab_graphql_tool
    ):
        """Test that subscriptions with comments are blocked."""
        # Execute tool with subscription that has comments
        subscription = """
        # Subscribe to merge request updates
        subscription WatchMR($projectPath: ID!, $iid: String!) {
            # Watch for any changes to the MR
            mergeRequestUpdated(projectPath: $projectPath, iid: $iid) {
                id
                title  # MR title
                state  # Current state
                updatedAt
            }
        }
        """
        result = await gitlab_graphql_tool._execute(query=subscription)

        # Verify subscription is still detected
        result_json = json.loads(result)
        assert "error" in result_json
        assert "mutations and subscriptions are not allowed" in result_json["error"]

    @pytest.mark.asyncio
    async def test_graphql_mutation_with_comment_blocked(self, gitlab_graphql_tool):
        """Test that mutations with leading comments are blocked."""
        # Execute tool with mutation that has a comment at the start
        mutation = """
        # Update MR title
        mutation UpdateMRTitle($projectPath: ID!, $iid: String!, $title: String!) {
            mergeRequestUpdate(
                input: {
                    projectPath: $projectPath
                    iid: $iid
                    title: $title
                }
            ) {
                mergeRequest {
                    id
                    title
                }
                errors
            }
        }
        """
        result = await gitlab_graphql_tool._execute(query=mutation)

        # Verify mutation is still detected
        result_json = json.loads(result)
        assert "error" in result_json
        assert "mutations and subscriptions are not allowed" in result_json["error"]

    @pytest.mark.asyncio
    async def test_graphql_mutation_with_fragment_blocked(self, gitlab_graphql_tool):
        """Test that mutations with leading fragments are blocked."""
        # Execute tool with mutation that has a fragment at the start
        mutation = """
        fragment MRFields on MergeRequest {
            id
            title
            state
        }

        mutation UpdateTitle {
            mergeRequestUpdate(
                input: {
                    projectPath: "gitlab-duo/gdk-ai-gateway"
                    iid: "11"
                    title: "feat: Add comprehensive search with Elasticsearch (ES)"
                }
            ) {
                mergeRequest {
                    ...MRFields
                }
                errors
            }
        }
        """
        result = await gitlab_graphql_tool._execute(query=mutation)

        # Verify mutation is still detected
        result_json = json.loads(result)
        assert "error" in result_json
        assert "mutations and subscriptions are not allowed" in result_json["error"]

    @pytest.mark.parametrize(
        "exploit_query",
        [
            pytest.param(
                "#\rquery{\n"
                " mutation{\n"
                'createNote(input: {noteableId: "gid://gitlab/Issue/23101307",'
                ' body: "testing mutations"}){clientMutationId}}'
                " \n#\r}",
                id="cr_wraps_mutation_as_query_field",
            ),
            pytest.param(
                "#\rquery{\n"
                ' subscription { mergeRequestUpdated(projectPath: "foo") { id } }'
                " \n#\r}",
                id="cr_wraps_subscription_as_query_field",
            ),
            pytest.param(
                "#\r\rquery{\n"
                " mutation{\n"
                'createNote(input: {noteableId: "gid://gitlab/Issue/1",'
                ' body: "test"}){clientMutationId}}'
                " \n#\r\r}",
                id="multiple_cr_wraps_mutation_as_query_field",
            ),
        ],
    )
    @pytest.mark.asyncio
    async def test_graphql_cr_exploit_is_neutralized_via_normalization(
        self, gitlab_graphql_tool, gitlab_client_mock, exploit_query
    ):
        """Test that \\r-based parser-differential exploits are neutralized.

        Python's graphql-core treats \\r as a line terminator, so it parses these payloads as queries (not mutations).
        The tool normalizes the query via print_ast before sending to Rails, stripping comments and \\r characters. This
        ensures Ruby's parser sees the same AST as Python — a query with harmless unresolvable field names, not a
        mutation operation.
        """
        response_data = {"data": {"mutation": None}}
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = True
        mock_response.body = response_data
        gitlab_client_mock.apost = AsyncMock(return_value=mock_response)

        await gitlab_graphql_tool._execute(query=exploit_query)

        gitlab_client_mock.apost.assert_called_once()
        sent_payload = json.loads(gitlab_client_mock.apost.call_args[1]["body"])
        sent_query = sent_payload["query"]
        assert "\r" not in sent_query
        assert sent_query != exploit_query

    @pytest.mark.asyncio
    async def test_graphql_mutation_hidden_by_cr_comment_blocked(
        self, gitlab_graphql_tool, gitlab_client_mock
    ):
        """Test that a mutation hidden behind a \\r-terminated comment is blocked.

        '#\\rmutation {...}' is parsed by Python as comment '#' followed by
        a real mutation operation. After print_ast normalization the mutation
        keyword is preserved, and the existing operation-type check blocks it.
        """
        exploit_query = (
            "#\rmutation { createNote(input: "
            '{noteableId: "gid://gitlab/Issue/1", body: "test"}) '
            "{ clientMutationId } }"
        )
        gitlab_client_mock.apost = AsyncMock()
        result = await gitlab_graphql_tool._execute(query=exploit_query)

        result_json = json.loads(result)
        assert "error" in result_json
        assert "mutations and subscriptions are not allowed" in result_json["error"]
        gitlab_client_mock.apost.assert_not_called()

    @pytest.mark.asyncio
    async def test_graphql_query_with_crlf_line_endings_allowed(
        self, gitlab_graphql_tool, gitlab_client_mock
    ):
        """Test that a legitimate query with \\r\\n (Windows) line endings still works."""
        query = (
            'query {\r\n  project(fullPath: "foo/bar")'
            " {\r\n    id\r\n    title\r\n  }\r\n}"
        )
        response_data = {"data": {"project": {"id": "1", "title": "Test"}}}
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = True
        mock_response.body = response_data
        gitlab_client_mock.apost = AsyncMock(return_value=mock_response)

        result = await gitlab_graphql_tool._execute(query=query)

        result_json = json.loads(result)
        assert result_json["status"] == "success"
        gitlab_client_mock.apost.assert_called_once()
        sent_payload = json.loads(gitlab_client_mock.apost.call_args[1]["body"])
        assert "\r" not in sent_payload["query"]

    @pytest.mark.asyncio
    async def test_graphql_invalid_query_rejected(self, gitlab_graphql_tool):
        """Test that invalid GraphQL queries are rejected."""
        # Execute tool with malformed query
        invalid_query = "this is not valid GraphQL { syntax"
        result = await gitlab_graphql_tool._execute(query=invalid_query)

        # Verify it's rejected
        result_json = json.loads(result)
        assert "error" in result_json
        assert "Invalid GraphQL query" in result_json["error"]

    @pytest.mark.asyncio
    async def test_graphql_error_in_response(
        self, gitlab_graphql_tool, gitlab_client_mock
    ):
        """Test handling of GraphQL errors in response."""
        # Mock response with GraphQL errors
        response_data = {
            "errors": [
                {"message": "Field not found", "path": ["project", "invalidField"]}
            ],
            "data": None,
        }
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = True
        mock_response.body = response_data
        gitlab_client_mock.apost = AsyncMock(return_value=mock_response)

        # Execute tool
        query = 'query { project(fullPath: "test") { invalidField } }'
        result = await gitlab_graphql_tool._execute(query=query)

        # Verify
        result_json = json.loads(result)
        assert "error" in result_json
        assert "GraphQL query returned errors" in result_json["error"]
        assert len(result_json["graphql_errors"]) == 1

    @pytest.mark.asyncio
    async def test_graphql_api_failure(self, gitlab_graphql_tool, gitlab_client_mock):
        """Test handling of API failure."""
        # Mock error response
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = False
        mock_response.status_code = 500
        mock_response.body = "Internal server error"
        gitlab_client_mock.apost = AsyncMock(return_value=mock_response)

        # Execute tool
        query = "query { currentUser { username } }"
        result = await gitlab_graphql_tool._execute(query=query)

        # Verify
        result_json = json.loads(result)
        assert "error" in result_json
        assert result_json["status_code"] == 500

    @pytest.mark.asyncio
    async def test_graphql_exception_handling(
        self, gitlab_graphql_tool, gitlab_client_mock
    ):
        """Test handling of exceptions during GraphQL call."""
        # Mock exception
        gitlab_client_mock.apost = AsyncMock(side_effect=Exception("Network error"))

        # Execute tool
        query = "query { currentUser { username } }"
        result = await gitlab_graphql_tool._execute(query=query)

        # Verify
        result_json = json.loads(result)
        assert "error" in result_json
        assert "Network error" in result_json["details"]

    def test_format_display_message_with_variables(self, gitlab_graphql_tool):
        """Test display message formatting with variables."""
        args = GitLabGraphQLInput(
            query="query { project { name } }", variables={"projectPath": "test"}
        )
        message = gitlab_graphql_tool.format_display_message(args)
        assert "with variables" in message

    def test_format_display_message_without_variables(self, gitlab_graphql_tool):
        """Test display message formatting without variables."""
        args = GitLabGraphQLInput(query="query { currentUser { username } }")
        message = gitlab_graphql_tool.format_display_message(args)
        assert "GraphQL query" in message
