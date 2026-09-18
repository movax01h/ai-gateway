import json
from unittest.mock import AsyncMock, Mock
from urllib.parse import urlencode

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.gitlab_api_generic import (
    GitLabApiGet,
    GitLabApiGetInput,
    GitLabGraphQL,
    GitLabGraphQLInput,
    validate_api_endpoint,
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
        mock_response.body = json.dumps(response_data)
        mock_response.headers = {}
        gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

        # Execute tool
        result = await gitlab_api_get_tool._execute(
            endpoint="/api/v4/projects/13/merge_requests/42"
        )

        # Verify
        gitlab_client_mock.aget.assert_called_once_with(
            path="/api/v4/projects/13/merge_requests/42",
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
        mock_response.body = json.dumps(response_data)
        mock_response.headers = {}
        gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

        # Execute tool
        params = {"state": "opened", "per_page": 20}
        result = await gitlab_api_get_tool._execute(
            endpoint="/api/v4/projects/13/merge_requests", params=params
        )

        # Verify
        expected_query = "?" + urlencode(params, doseq=True)
        expected_path = f"/api/v4/projects/13/merge_requests{expected_query}"
        gitlab_client_mock.aget.assert_called_once_with(
            path=expected_path,
        )
        result_json = json.loads(result)
        assert result_json["status"] == "success"
        assert result_json["data"] == response_data

    @pytest.mark.asyncio
    async def test_get_error_no_endpoint(self, gitlab_api_get_tool):
        """Test error when endpoint is not provided."""
        with pytest.raises(ToolException) as exc_info:
            await gitlab_api_get_tool._execute()
        assert "The 'endpoint' parameter must be provided" in str(exc_info.value)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "valid_endpoint",
        [
            "/api/v4/projects/13/merge_requests/42",
            "/api/v4/projects/13/issues",
            "/api/v4/users/1",
            "/api/v4/projects/namespace%2Fproject",
            "/api/v4/projects/13/repository/files/src%2Ftest.py",
            "/api/v4/projects/my.project/merge_requests/42",
        ],
    )
    async def test_get_valid_endpoints_accepted(
        self, gitlab_api_get_tool, gitlab_client_mock, valid_endpoint
    ):
        """Test that legitimate endpoints are accepted."""
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = True
        mock_response.body = json.dumps({"status": "ok"})
        mock_response.headers = {}
        gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

        result = await gitlab_api_get_tool._execute(endpoint=valid_endpoint)
        result_json = json.loads(result)
        assert result_json["status"] == "success"

    @pytest.mark.asyncio
    async def test_get_error_api_failure(self, gitlab_api_get_tool, gitlab_client_mock):
        """Test handling of API failure response."""
        # Mock error response
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = False
        mock_response.status_code = 404
        mock_response.body = json.dumps({"error": "Not found"})
        gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

        # Execute tool
        with pytest.raises(ToolException) as exc_info:
            await gitlab_api_get_tool._execute(
                endpoint="/api/v4/projects/13/merge_requests/999"
            )

        # Verify
        assert "HTTP 404" in str(exc_info.value)
        assert "Not found" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_get_exception_handling(
        self, gitlab_api_get_tool, gitlab_client_mock
    ):
        """Test handling of exceptions during API call."""
        # Mock exception
        gitlab_client_mock.aget = AsyncMock(side_effect=Exception("Connection error"))

        # Execute tool - exceptions are not caught, they propagate
        with pytest.raises(Exception, match="Connection error"):
            await gitlab_api_get_tool._execute(
                endpoint="/api/v4/projects/13/merge_requests/42"
            )

    @pytest.mark.asyncio
    async def test_get_response_includes_pagination_metadata(
        self, gitlab_api_get_tool, gitlab_client_mock
    ):
        """Test that pagination headers are included in the response."""
        response_data = [{"id": 1}, {"id": 2}]
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = True
        mock_response.body = json.dumps(response_data)
        mock_response.headers = {
            "X-Next-Page": "2",
            "X-Total": "50",
            "X-Total-Pages": "3",
        }
        gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

        result = await gitlab_api_get_tool._execute(
            endpoint="/api/v4/projects/13/issues"
        )

        result_json = json.loads(result)
        assert result_json["status"] == "success"
        assert result_json["pagination"] == {
            "next_page": "2",
            "total": "50",
            "total_pages": "3",
        }

    @pytest.mark.asyncio
    async def test_get_response_no_pagination_when_headers_absent(
        self, gitlab_api_get_tool, gitlab_client_mock
    ):
        """Test that pagination key is omitted when no pagination headers exist."""
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = True
        mock_response.body = json.dumps({"id": 1})
        mock_response.headers = {}
        gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

        result = await gitlab_api_get_tool._execute(
            endpoint="/api/v4/projects/13/merge_requests/42"
        )

        result_json = json.loads(result)
        assert "pagination" not in result_json

    @pytest.mark.asyncio
    async def test_get_response_partial_pagination_headers(
        self, gitlab_api_get_tool, gitlab_client_mock
    ):
        """Test pagination with only some headers present (e.g. last page)."""
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = True
        mock_response.body = json.dumps([{"id": 3}])
        mock_response.headers = {
            "X-Next-Page": "",
            "X-Total": "3",
            "X-Total-Pages": "1",
        }
        gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

        result = await gitlab_api_get_tool._execute(
            endpoint="/api/v4/projects/13/issues"
        )

        result_json = json.loads(result)
        assert result_json["pagination"] == {
            "total": "3",
            "total_pages": "1",
        }

    @pytest.mark.asyncio
    async def test_get_response_pagination_with_lowercase_header_keys(
        self, gitlab_api_get_tool, gitlab_client_mock
    ):
        """Test that pagination extraction works with lowercase HTTP header keys.

        Some HTTP clients normalize header names to lowercase (e.g. httpx, aiohttp). Ensure _extract_pagination_headers
        handles this correctly.
        """
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = True
        mock_response.body = json.dumps([{"id": 1}])
        mock_response.headers = {
            "x-next-page": "2",
            "x-total": "42",
            "x-total-pages": "3",
        }
        gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

        result = await gitlab_api_get_tool._execute(
            endpoint="/api/v4/projects/13/issues"
        )

        result_json = json.loads(result)
        assert result_json["status"] == "success"
        assert result_json["pagination"] == {
            "next_page": "2",
            "total": "42",
            "total_pages": "3",
        }

    @pytest.mark.asyncio
    async def test_work_items_endpoint_redirected_to_issues(
        self, gitlab_api_get_tool, gitlab_client_mock
    ):
        """Test that work_items endpoints are redirected to issues endpoints."""
        response_data = {"id": 1, "title": "Test Issue"}
        mock_response = Mock(spec=GitLabHttpResponse)
        mock_response.is_success.return_value = True
        mock_response.body = json.dumps(response_data)
        mock_response.headers = {}
        gitlab_client_mock.aget = AsyncMock(return_value=mock_response)

        result = await gitlab_api_get_tool._execute(
            endpoint="/api/v4/projects/13/work_items/42"
        )

        gitlab_client_mock.aget.assert_called_once_with(
            path="/api/v4/projects/13/issues/42",
        )
        result_json = json.loads(result)
        assert result_json["status"] == "success"
        assert result_json["data"] == response_data

    def test_format_display_message_with_endpoint(self, gitlab_api_get_tool):
        """Test display message formatting with endpoint."""
        args = GitLabApiGetInput(endpoint="/api/v4/projects/13/merge_requests/42")
        message = gitlab_api_get_tool.format_display_message(args)
        assert "/api/v4/projects/13/merge_requests/42" in message

    def test_params_default_is_empty_dict(self):
        """Test that GitLabApiGetInput.params defaults to an empty dict (not None).

        The default_factory=dict pattern avoids Pydantic validation issues that
        arise when LangChain/tool callers receive None for a dict-typed field.
        """
        args = GitLabApiGetInput(endpoint="/api/v4/projects/13")
        assert args.params == {}
        assert isinstance(args.params, dict)

    def test_params_accepts_dict(self):
        """Test that GitLabApiGetInput.params accepts an explicit dict."""
        args = GitLabApiGetInput(
            endpoint="/api/v4/projects/13/issues",
            params={"state": "opened", "per_page": 20},
        )
        assert args.params == {"state": "opened", "per_page": 20}

    def test_params_rejects_none(self):
        """Test that GitLabApiGetInput.params no longer accepts None.

        With default_factory=dict, the field is non-optional and rejects None.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GitLabApiGetInput(
                endpoint="/api/v4/projects/13/issues", params=None
            )

    def test_params_coerces_json_string(self):
        """Test that JSON-encoded string params are coerced to a dict.

        Some LLMs (notably Qwen variants) double-serialize nested dict
        parameters in tool calls, so params arrives as a JSON-encoded
        string like '{"ref":"main"}' instead of a real dict. The
        before-validator parses the string so downstream code can
        iterate it as a normal dict.
        """
        args = GitLabApiGetInput(
            endpoint="/api/v4/projects/13/repository/files/src%2Flib%2Ffile.py",
            params='{"ref": "main"}',
        )
        assert args.params == {"ref": "main"}
        assert isinstance(args.params, dict)

    def test_params_invalid_json_string_still_rejected(self):
        """Test that a non-JSON string falls through to Pydantic's default error.

        The before-validator only coerces strings that successfully parse as
        JSON. Anything else is passed through unchanged so Pydantic raises
        its normal dict-type validation error rather than silently dropping
        the value.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GitLabApiGetInput(
                endpoint="/api/v4/projects/13/issues", params="not-a-json-string"
            )


class TestValidateApiEndpoint:
    """Tests for the validate_api_endpoint function."""

    @pytest.mark.parametrize(
        "valid_endpoint",
        [
            "/api/v4/projects/13/merge_requests/42",
            "/api/v4/projects/13/issues",
            "/api/v4/users/1",
            "/api/v4/projects/namespace%2Fproject",
            "/api/v4/projects/13/repository/files/src%2Ftest.py",
            "/api/v4/projects/my.project/merge_requests/42",
            "/api/v4/projects/joernchen%2Fnothing_to_see_here/repository/files/j%C3%B6rn.txt",
            "/api/v4/projects/13/repository/files/%E4%B8%AD%E6%96%87.txt",
            "/api/v4/projects/joernchen%2Fnothing_to_see_here/repository/files/%23does%20this%20work%3F.txt",
        ],
    )
    def test_accepts_valid_endpoints(self, valid_endpoint):
        """Test that legitimate endpoints pass validation with original encoding preserved."""
        assert validate_api_endpoint(valid_endpoint) == valid_endpoint

    @pytest.mark.parametrize(
        "malicious_endpoint",
        [
            "/api/v4/../../admin/users",
            "/api/v4/../../../etc/passwd",
            "/api/v4/%2e%2e/%2e%2e/admin/users",
            "/api/v4/%2E%2E/%2E%2E/admin/users",
            "/api/v4/..%2f..%2fadmin/users",
            "/api/v4/..\\..\\admin\\users",
            "/api/v4/..\\admin\\users",
            "/api/v4/projects/13/../../../../../../admin/users",
            "/api/v4/projects/%2e%2e/../../admin/users",
            "/api/v4/\u002e\u002e/admin/users",
            "/invalid/endpoint",
            "/api/v4admin/users",
        ],
    )
    def test_rejects_path_traversal(self, malicious_endpoint):
        """Test that path traversal attempts raise ToolException."""
        with pytest.raises(ToolException, match="Invalid endpoint"):
            validate_api_endpoint(malicious_endpoint)

    @pytest.mark.parametrize(
        "payload",
        [
            "/api/v4/projects/1\n/admin/users",
            "/api/v4/projects/1\r\n/admin/users",
        ],
    )
    def test_rejects_newline_injection(self, payload):
        """Test that newline/CRLF injection raises ToolException."""
        with pytest.raises(ToolException, match="Invalid endpoint"):
            validate_api_endpoint(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            "/api/v4/projects/1\x00/admin",
            "/api/v4/projects/1%00/admin",
        ],
    )
    def test_rejects_null_bytes(self, payload):
        """Test that null bytes (literal and percent-encoded) raise ToolException."""
        with pytest.raises(ToolException, match="Invalid endpoint"):
            validate_api_endpoint(payload)

    @pytest.mark.parametrize(
        "payload",
        [
            "/api/v4/projects/13?state=opened",
            "/api/v4/projects/13#fragment",
            "/api/v4/projects/13/issues?state=opened&per_page=20",
        ],
    )
    def test_rejects_query_string_and_fragment(self, payload):
        """Test that query strings and fragments raise ToolException."""
        with pytest.raises(ToolException, match="path only"):
            validate_api_endpoint(payload)


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
        with pytest.raises(ToolException) as exc_info:
            await gitlab_graphql_tool._execute(query=mutation)

        # Verify
        assert "mutations and subscriptions are not allowed" in str(exc_info.value)

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
        with pytest.raises(ToolException) as exc_info:
            await gitlab_graphql_tool._execute(query=subscription)

        # Verify
        assert "mutations and subscriptions are not allowed" in str(exc_info.value)

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
        with pytest.raises(ToolException) as exc_info:
            await gitlab_graphql_tool._execute(query=subscription)

        # Verify subscription is still detected
        assert "mutations and subscriptions are not allowed" in str(exc_info.value)

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
        with pytest.raises(ToolException) as exc_info:
            await gitlab_graphql_tool._execute(query=mutation)

        # Verify mutation is still detected
        assert "mutations and subscriptions are not allowed" in str(exc_info.value)

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
        with pytest.raises(ToolException) as exc_info:
            await gitlab_graphql_tool._execute(query=mutation)

        # Verify mutation is still detected
        assert "mutations and subscriptions are not allowed" in str(exc_info.value)

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
        with pytest.raises(ToolException) as exc_info:
            await gitlab_graphql_tool._execute(query=exploit_query)

        assert "mutations and subscriptions are not allowed" in str(exc_info.value)
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
        # Execute tool with malformed query - parsing exceptions propagate
        invalid_query = "this is not valid GraphQL { syntax"

        # The graphql parser will raise an exception for invalid syntax
        with pytest.raises(Exception):
            await gitlab_graphql_tool._execute(query=invalid_query)

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
        with pytest.raises(ToolException) as exc_info:
            await gitlab_graphql_tool._execute(query=query)

        # Verify
        error_message = str(exc_info.value)
        assert "GraphQL query returned errors" in error_message
        assert "Field not found" in error_message

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
        with pytest.raises(ToolException) as exc_info:
            await gitlab_graphql_tool._execute(query=query)

        # Verify
        assert "HTTP 500" in str(exc_info.value)
        assert "Internal server error" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_graphql_exception_handling(
        self, gitlab_graphql_tool, gitlab_client_mock
    ):
        """Test handling of exceptions during GraphQL call."""
        # Mock exception
        gitlab_client_mock.apost = AsyncMock(side_effect=Exception("Network error"))

        # Execute tool - exceptions are not caught, they propagate
        query = "query { currentUser { username } }"
        with pytest.raises(Exception, match="Network error"):
            await gitlab_graphql_tool._execute(query=query)

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

    def test_variables_default_is_empty_dict(self):
        """Test that GitLabGraphQLInput.variables defaults to an empty dict (not None).

        The default_factory=dict pattern avoids Pydantic validation issues that
        arise when LangChain/tool callers receive None for a dict-typed field.
        """
        args = GitLabGraphQLInput(query="query { currentUser { username } }")
        assert args.variables == {}
        assert isinstance(args.variables, dict)

    def test_variables_accepts_dict(self):
        """Test that GitLabGraphQLInput.variables accepts an explicit dict."""
        args = GitLabGraphQLInput(
            query="query GetProject($projectPath: ID!) { project(fullPath: $projectPath) { name } }",
            variables={"projectPath": "namespace/project"},
        )
        assert args.variables == {"projectPath": "namespace/project"}

    def test_variables_rejects_none(self):
        """Test that GitLabGraphQLInput.variables no longer accepts None.

        With default_factory=dict, the field is non-optional and rejects None.
        """
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GitLabGraphQLInput(query="query { currentUser { username } }", variables=None)

    def test_variables_coerces_json_string(self):
        """Test that JSON-encoded string variables are coerced to a dict.

        Mirrors GitLabApiGetInput: a model that double-serializes `variables`
        should not fail validation.
        """
        args = GitLabGraphQLInput(
            query="query GetProject($projectPath: ID!) { project(fullPath: $projectPath) { name } }",
            variables='{"projectPath": "namespace/project", "iid": "42"}',
        )
        assert args.variables == {"projectPath": "namespace/project", "iid": "42"}
        assert isinstance(args.variables, dict)

    def test_variables_invalid_json_string_still_rejected(self):
        """Test that a non-JSON string falls through to Pydantic's default error."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            GitLabGraphQLInput(
                query="query { currentUser { username } }",
                variables="not-a-json-string",
            )
