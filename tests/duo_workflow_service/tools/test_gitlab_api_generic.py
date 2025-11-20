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
    async def test_get_with_url_parsing_merge_request(
        self, gitlab_api_get_tool, gitlab_client_mock
    ):
        """Test API GET request with URL parsing - path extracted as-is."""
        # Mock response
        response_data = {"id": 42, "title": "Test MR"}
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = True
        mock_response.body = response_data
        gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

        # Execute tool with an API endpoint URL
        result = await gitlab_api_get_tool._execute(
            url="https://gitlab.com/api/v4/projects/13/merge_requests/42"
        )

        # Verify - path is extracted as-is from URL
        expected_path = "/api/v4/projects/13/merge_requests/42"
        gitlab_client_mock.aget.assert_called_once_with(path=expected_path)
        result_json = json.loads(result)
        assert result_json["status"] == "success"
        assert result_json["data"] == response_data

    @pytest.mark.asyncio
    async def test_get_with_url_parsing_issue(
        self, gitlab_api_get_tool, gitlab_client_mock
    ):
        """Test API GET request with URL parsing - path extracted as-is."""
        # Mock response
        response_data = {"iid": 10, "title": "Test Issue"}
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = True
        mock_response.body = response_data
        gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

        # Execute tool with an API endpoint URL
        result = await gitlab_api_get_tool._execute(
            url="https://gitlab.com/api/v4/projects/13/issues/10"
        )

        # Verify - path is extracted as-is from URL
        expected_path = "/api/v4/projects/13/issues/10"
        gitlab_client_mock.aget.assert_called_once_with(path=expected_path)
        result_json = json.loads(result)
        assert result_json["status"] == "success"

    @pytest.mark.asyncio
    async def test_get_with_url_parsing_project(
        self, gitlab_api_get_tool, gitlab_client_mock
    ):
        """Test API GET request with URL parsing - path extracted as-is."""
        # Mock response
        response_data = {"id": 13, "name": "Test Project"}
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = True
        mock_response.body = response_data
        gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

        # Execute tool with an API endpoint URL
        result = await gitlab_api_get_tool._execute(
            url="https://gitlab.com/api/v4/projects/13"
        )

        # Verify - path is extracted as-is from URL
        expected_path = "/api/v4/projects/13"
        gitlab_client_mock.aget.assert_called_once_with(path=expected_path)
        result_json = json.loads(result)
        assert result_json["status"] == "success"

    @pytest.mark.asyncio
    async def test_get_with_nested_group_url(
        self, gitlab_api_get_tool, gitlab_client_mock
    ):
        """Test API GET request with URL - path extracted as-is."""
        # Mock response
        response_data = {"id": 123, "name": "Nested Project"}
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = True
        mock_response.body = response_data
        gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

        # Execute tool with an API endpoint URL
        result = await gitlab_api_get_tool._execute(
            url="https://gitlab.com/api/v4/groups/123/projects"
        )

        # Verify - path is extracted as-is from URL
        expected_path = "/api/v4/groups/123/projects"
        gitlab_client_mock.aget.assert_called_once_with(path=expected_path)
        result_json = json.loads(result)
        assert result_json["status"] == "success"

    @pytest.mark.asyncio
    async def test_get_error_no_endpoint_or_url(self, gitlab_api_get_tool):
        """Test error when neither endpoint nor url is provided."""
        result = await gitlab_api_get_tool._execute()
        result_json = json.loads(result)
        assert "error" in result_json
        assert (
            "Either 'endpoint' or 'url' parameter must be provided"
            in result_json["error"]
        )

    @pytest.mark.asyncio
    async def test_get_error_invalid_endpoint_format(self, gitlab_api_get_tool):
        """Test error when endpoint doesn't start with /api/v4/."""
        result = await gitlab_api_get_tool._execute(endpoint="/invalid/endpoint")
        result_json = json.loads(result)
        assert "error" in result_json
        assert "Invalid endpoint format" in result_json["error"]

    @pytest.mark.asyncio
    async def test_get_error_invalid_url(self, gitlab_api_get_tool):
        """Test error when URL has no path."""
        # Use a URL with no path
        result = await gitlab_api_get_tool._execute(url="https://gitlab.com")
        result_json = json.loads(result)
        assert "error" in result_json
        assert "Failed to parse GitLab URL" in result_json["error"]

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

    def test_format_display_message_with_url(self, gitlab_api_get_tool):
        """Test display message formatting with URL."""
        args = GitLabApiGetInput(
            url="https://gitlab.com/namespace/project/-/merge_requests/42"
        )
        message = gitlab_api_get_tool.format_display_message(args)
        assert "https://gitlab.com/namespace/project/-/merge_requests/42" in message

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

    def test_strip_query_for_preview_single_line_comment(self, gitlab_graphql_tool):
        """Test stripping single-line comments from GraphQL query."""
        query = """
        # This is a comment
        query { currentUser { username } }
        """
        cleaned = gitlab_graphql_tool._strip_query_for_preview(query)
        assert cleaned == "query { currentUser { username } }"
        assert "#" not in cleaned

    def test_strip_query_for_preview_multiple_comments(self, gitlab_graphql_tool):
        """Test stripping multiple consecutive comment lines from GraphQL query."""
        query = """
        # This is the first comment
        # This is the second comment
        # This is the third comment
        query {
            # Another comment in the middle
            currentUser {
                username # inline comment
            }
        }
        """
        cleaned = gitlab_graphql_tool._strip_query_for_preview(query)
        assert cleaned == "query { currentUser { username } }"
        assert "#" not in cleaned
        assert "first comment" not in cleaned
        assert "second comment" not in cleaned
        assert "inline comment" not in cleaned

    def test_strip_query_for_preview_whitespace_collapsing(self, gitlab_graphql_tool):
        """Test collapsing excess whitespace in GraphQL query."""
        query = """
        query    {
            currentUser     {
                username

                email
            }
        }
        """
        cleaned = gitlab_graphql_tool._strip_query_for_preview(query)
        # All whitespace should be collapsed to single spaces
        assert "    " not in cleaned
        assert "     " not in cleaned
        assert "\n" not in cleaned
        # Should have single spaces between tokens
        assert cleaned == "query { currentUser { username email } }"

    def test_strip_query_for_preview_mutation_with_comments(self, gitlab_graphql_tool):
        """Test stripping comments from a mutation query."""
        query = """
        # Update merge request title
        mutation UpdateMRTitle($projectPath: ID!, $iid: String!, $title: String!) {
            # Call the mergeRequestUpdate mutation
            mergeRequestUpdate(
                input: {
                    projectPath: $projectPath  # Project path
                    iid: $iid  # MR IID
                    title: $title  # New title
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
        cleaned = gitlab_graphql_tool._strip_query_for_preview(query)
        # Should have no comments
        assert "#" not in cleaned
        # Should have mutation keyword
        assert "mutation" in cleaned
        # Should be collapsed to single line with single spaces
        assert "\n" not in cleaned
        assert "  " not in cleaned

    def test_strip_query_for_preview_empty_lines(self, gitlab_graphql_tool):
        """Test removal of empty lines from GraphQL query."""
        query = """

        query {

            currentUser {

                username

            }

        }

        """
        cleaned = gitlab_graphql_tool._strip_query_for_preview(query)
        assert cleaned == "query { currentUser { username } }"
        # Verify no extra spaces from empty lines
        assert cleaned.count("  ") == 0

    def test_strip_query_for_preview_mixed_comments_and_whitespace(
        self, gitlab_graphql_tool
    ):
        """Test comprehensive cleaning with comments, whitespace, and empty lines."""
        query = """
        # GraphQL query to fetch project details
        # Including merge requests and issues

        query GetProjectDetails($projectPath: ID!) {
            # Fetch the project
            project(fullPath: $projectPath) {
                name  # Project name
                description

                # Get merge requests
                mergeRequests(first: 10) {
                    nodes {
                        title
                        author {
                            username  # Author's username
                        }
                    }
                }
            }
        }
        """
        cleaned = gitlab_graphql_tool._strip_query_for_preview(query)
        # Should have no comments
        assert "#" not in cleaned
        # Should be collapsed
        assert "\n" not in cleaned
        # Should have single spaces only
        assert "  " not in cleaned
        # Should contain key parts
        assert "query GetProjectDetails" in cleaned
        assert "project(fullPath: $projectPath)" in cleaned
        assert "mergeRequests(first: 10)" in cleaned
