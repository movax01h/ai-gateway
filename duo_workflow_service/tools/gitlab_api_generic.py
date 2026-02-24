"""Generic GitLab API tools for read-only operations.

These tools provide generic access to GitLab REST API and GraphQL API for read operations, reducing token consumption
and maintenance overhead compared to specialized tools.
"""

import json
from typing import Any, Dict, Optional, Type
from urllib.parse import urlencode

import structlog
from gitlab_cloud_connector import GitLabUnitPrimitive
from graphql import parse as parse_graphql
from graphql.language.ast import DocumentNode, OperationDefinitionNode, OperationType
from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

log = structlog.stdlib.get_logger("workflow")


class GitLabApiGetInput(BaseModel):
    """Input schema for gitlab_api_get tool."""

    endpoint: Optional[str] = Field(
        default=None,
        description=(
            "The GitLab API endpoint path (e.g., '/api/v4/projects/13/merge_requests/42'). "
            "Must start with '/api/v4/'."
        ),
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional query parameters as a dictionary. "
            "Example: {'state': 'opened', 'per_page': 20, 'page': 1}"
        ),
    )


class GitLabGraphQLInput(BaseModel):
    """Input schema for gitlab_graphql tool."""

    query: str = Field(
        description=(
            "The GraphQL query string. Only queries are supported, not mutations or subscriptions. "
            "Best practice: Always name your queries for better traceability (e.g., 'query GetProject'). "
            "Example: 'query GetProject($projectPath: ID!) { project(fullPath: $projectPath) { name description } }'"
        )
    )
    variables: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional variables for the GraphQL query as a dictionary.",
    )


class GitLabApiGet(DuoBaseTool):
    """Generic tool for making read-only GitLab REST API GET requests.

    This tool provides access to any GitLab REST API GET endpoint, reducing the need
    for specialized tools for every read operation.

    Security: Only supports GET requests. All write operations (POST, PUT, DELETE) must
    use specialized tools with proper validation.

    Examples:
        # Using endpoint directly
        gitlab_api_get(endpoint="/api/v4/projects/13/merge_requests/42")

        # Using endpoint with parameters
        gitlab_api_get(
            endpoint="/api/v4/projects/13/merge_requests",
            params={"state": "opened", "author_username": "janedoe"}
        )

        # Get project information
        gitlab_api_get(endpoint="/api/v4/projects/13")

        # List merge request notes
        gitlab_api_get(endpoint="/api/v4/projects/13/merge_requests/42/notes")

        # List issues with filters
        gitlab_api_get(
            endpoint="/api/v4/projects/13/issues",
            params={"state": "opened", "labels": "bug,urgent"}
        )
    """

    name: str = "gitlab_api_get"
    description: str = (
        "Make read-only GET requests to any GitLab REST API endpoint of the GitLab instance "
        "the user is currently using. Use this to retrieve information about projects, "
        "merge requests, issues, pipelines, commits, epics, work items, users, todos, wikis "
        "or any other GitLab resource. "
        "\n\nCommon API patterns:\n"
        "- Projects: /api/v4/projects/{id}\n"
        "- Merge Requests: /api/v4/projects/{id}/merge_requests/{iid}\n"
        "- Issues: /api/v4/projects/{id}/issues/{iid}\n"
        "- Pipelines: /api/v4/projects/{id}/pipelines/{id}\n"
        "- Commits: /api/v4/projects/{id}/repository/commits/{sha}\n"
        "- Repository Tree: /api/v4/projects/{id}/repository/tree with params={'path': 'sub/dir', 'ref': 'HEAD'}\n"
        "- Repository File: /api/v4/projects/{id}/repository/files/{file_path} with params={'ref': 'HEAD'}\n"
        "- Users: /api/v4/users/{id}\n"
        "\nTips:\n"
        "- Use ref=HEAD for repository file and tree requests to automatically resolve the default branch.\n"
        "- Use the params dict for query parameters (e.g., path, ref, per_page),"
        " don't append them to the endpoint URL.\n"
        "- For /repository/files/{file_path} endpoints, the file_path MUST be URL-encoded."
        " For example: /api/v4/projects/1/repository/files/src%%2Flib%%2Fmy%%20file.py\n"
        "\nSee https://docs.gitlab.com/ee/api/ for full API documentation."
    )
    args_schema: Type[BaseModel] = GitLabApiGetInput
    unit_primitive: Optional[GitLabUnitPrimitive] = None

    async def _execute(
        self,
        endpoint: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> str:
        """Execute the generic GitLab API GET request.

        Args:
            endpoint: The API endpoint path (must always start with /api/v4/, don't add the full URL)
            params: Optional query parameters
            **kwargs: Additional keyword arguments (ignored)

        Returns:
            JSON string with the API response or error information
        """
        # Validate that endpoint is provided
        if not endpoint:
            return json.dumps(
                {
                    "error": "The 'endpoint' parameter must be provided",
                    "details": "Please provide an API endpoint path",
                }
            )

        # Validate endpoint format
        if not endpoint.startswith("/api/v4/"):
            return json.dumps(
                {
                    "error": "Invalid endpoint format",
                    "endpoint": endpoint,
                    "details": "Endpoint must start with '/api/v4/'",
                }
            )

        # Build query string if params are provided
        query_string = ""
        if params:
            # Filter out None values
            filtered_params = {k: v for k, v in params.items() if v is not None}
            if filtered_params:
                query_string = "?" + urlencode(filtered_params, doseq=True)

        # Construct full path
        full_path = f"{endpoint}{query_string}"

        try:
            log.info("Making generic GitLab API GET request", extra={"path": full_path})

            response = await self.gitlab_client.aget(
                path=full_path,
            )

            if not response.is_success():
                # Safely handle response body for logging
                body_preview = None
                if response.body:
                    if isinstance(response.body, str):
                        body_preview = response.body[:500]
                    else:
                        body_preview = str(response.body)[:500]

                log.warning(
                    "GitLab API GET request failed",
                    extra={
                        "status_code": response.status_code,
                        "path": full_path,
                        "response_body": body_preview,
                    },
                )
                return json.dumps(
                    {
                        "error": f"API request failed with status {response.status_code}",
                        "status_code": response.status_code,
                        "endpoint": endpoint,
                        "details": (
                            response.body
                            if response.body
                            else "No error details provided"
                        ),
                    }
                )

            # Parse response body
            try:
                # Response body might already be parsed
                if isinstance(response.body, (dict, list)):
                    result = response.body
                else:
                    result = json.loads(response.body)

                return json.dumps(
                    {
                        "status": "success",
                        "data": result,
                    }
                )
            except json.JSONDecodeError:
                # If response is not JSON, return as string
                return json.dumps(
                    {
                        "status": "success",
                        "data": response.body,
                    }
                )

        except Exception as e:
            log.error(
                "Exception during GitLab API GET request",
                extra={"path": full_path, "error": str(e)},
                exc_info=True,
            )
            return json.dumps(
                {
                    "error": "Request failed with exception",
                    "endpoint": endpoint,
                    "details": str(e),
                }
            )

    def format_display_message(
        self, args: GitLabApiGetInput, _tool_response: Any = None
    ) -> str:
        """Format a display message for the tool execution."""
        if args.endpoint:
            params_str = f" with params {args.params}" if args.params else ""
            return f"Making GitLab API request: {args.endpoint}{params_str}"
        return "Making GitLab API request"


class GitLabGraphQL(DuoBaseTool):
    """Generic tool for making read-only GitLab GraphQL queries.

    This tool provides access to GitLab's GraphQL API for complex read operations.
    It only supports queries, not mutations or subscriptions, maintaining read-only access.

    Security: Only supports GraphQL queries. All mutations (write operations) and
    subscriptions (real-time data streaming) must use specialized tools with proper validation.

    Best Practices:
        - Always name your queries (e.g., "query GetMergeRequest") for better traceability in logs
        - Use descriptive operation names that indicate the query's purpose
        - Consider using @skip or @include directives for conditional fields based on GitLab version

    Examples:
        # Named query for merge request details (RECOMMENDED)
        gitlab_graphql(
            query='''
                query GetMergeRequestDetails($projectPath: ID!, $iid: String!) {
                    project(fullPath: $projectPath) {
                        mergeRequest(iid: $iid) {
                            title
                            description
                            author { username }
                            diffStats { additions deletions }
                            notes { nodes { body author { username } } }
                        }
                    }
                }
            ''',
            variables={"projectPath": "namespace/project", "iid": "42"}
        )

        # Named query with multiple resources
        gitlab_graphql(
            query='''
                query GetProjectOverview($projectPath: ID!) {
                    project(fullPath: $projectPath) {
                        name
                        description
                        mergeRequests(first: 10, state: opened) {
                            nodes {
                                title
                                author { username }
                                createdAt
                            }
                        }
                        issues(first: 5, state: opened) {
                            nodes {
                                title
                                assignees { nodes { username } }
                            }
                        }
                    }
                }
            ''',
            variables={"projectPath": "namespace/project"}
        )

        # Query with conditional field using @include (version-aware pattern)
        gitlab_graphql(
            query='''
                query GetUserDetails($username: String!, $includeStatus: Boolean!) {
                    user(username: $username) {
                        name
                        username
                        status @include(if: $includeStatus) {
                            message
                            availability
                        }
                    }
                }
            ''',
            variables={"username": "johndoe", "includeStatus": True}
        )
    """

    name: str = "gitlab_graphql"
    description: str = (
        "Execute read-only GraphQL queries against the GitLab GraphQL API. "
        "Use this for complex queries that need to fetch data from multiple related resources "
        "or when you need more flexibility than the REST API provides. "
        "\n\nBest Practice: Always name your queries (e.g., 'query GetMergeRequest') for better "
        "traceability in logs and debugging. Use descriptive operation names that indicate the "
        "query's purpose."
        "\n\nOnly queries are supported; mutations (write operations) and subscriptions "
        "(real-time data streaming) are not allowed. "
    )
    args_schema: Type[BaseModel] = GitLabGraphQLInput
    unit_primitive: Optional[GitLabUnitPrimitive] = None

    def _strip_query_for_preview(self, query: str) -> str:
        """Strip comments and excess whitespace from a GraphQL query for logging.

        Args:
            query: The GraphQL query string

        Returns:
            Cleaned query string with comments and excess whitespace removed
        """
        lines = query.split("\n")
        cleaned_lines = []

        for line in lines:
            # Remove comments (content after #)
            if "#" in line:
                line = line[: line.index("#")]

            # Strip leading/trailing whitespace
            line = line.strip()

            # Skip empty lines
            if line:
                cleaned_lines.append(line)

        # Join with single space and collapse multiple spaces
        cleaned = " ".join(cleaned_lines)
        cleaned = " ".join(cleaned.split())

        return cleaned

    def _query_contains_mutation_or_subscription(self, query: str) -> bool:
        """Check if a GraphQL query contains any mutation or subscription operations.

        This method properly parses the GraphQL query to detect mutations and subscriptions,
        including those hidden behind comments or fragments.

        Args:
            query: The GraphQL query string to check

        Returns:
            True if the query contains any mutations or subscriptions, False otherwise

        Raises:
            Exception: If the query cannot be parsed
        """
        # Parse the GraphQL query into an AST
        document: DocumentNode = parse_graphql(query)

        # Check all operation definitions in the document
        for definition in document.definitions:
            if isinstance(definition, OperationDefinitionNode):
                # Check if this operation is a mutation or subscription
                if definition.operation == OperationType.MUTATION:
                    cleaned_query = self._strip_query_for_preview(query)
                    log.warning(
                        "Blocked GraphQL mutation attempt",
                        extra={"query_preview": cleaned_query[:200]},
                    )
                    return True
                if definition.operation == OperationType.SUBSCRIPTION:
                    cleaned_query = self._strip_query_for_preview(query)
                    log.warning(
                        "Blocked GraphQL subscription attempt",
                        extra={"query_preview": cleaned_query[:200]},
                    )
                    return True

        return False

    async def _execute(  # pylint: disable=too-many-return-statements
        self, query: str, variables: Optional[Dict[str, Any]] = None, **kwargs: Any
    ) -> str:
        """Execute a GraphQL query against the GitLab GraphQL API.

        Args:
            query: The GraphQL query string
            variables: Optional variables for the query
            **kwargs: Additional keyword arguments (ignored)

        Returns:
            JSON string with the GraphQL response or error information
        """
        # Validate that the query is not a mutation or subscription by parsing the GraphQL
        try:
            if self._query_contains_mutation_or_subscription(query):
                return json.dumps(
                    {
                        "error": "GraphQL mutations and subscriptions are not allowed",
                        "details": (
                            "This tool only supports read-only queries. "
                            "Mutations and subscriptions are not supported. "
                            "Use specialized tools for write operations."
                        ),
                    }
                )
        except Exception as e:
            # If we can't parse the query, reject it for safety
            return json.dumps(
                {
                    "error": "Invalid GraphQL query",
                    "details": f"Failed to parse query: {str(e)}",
                }
            )

        # Build the GraphQL request payload
        payload: Dict[str, Any] = {"query": query}
        if variables:
            payload["variables"] = variables

        try:
            log.info(
                "Making GitLab GraphQL query", extra={"has_variables": bool(variables)}
            )

            response = await self.gitlab_client.apost(
                path="/api/graphql",
                body=json.dumps(payload),
            )

            if not response.is_success():
                log.warning(
                    "GitLab GraphQL query failed",
                    extra={
                        "status_code": response.status_code,
                        "response_body": response.body[:500] if response.body else None,
                    },
                )
                return json.dumps(
                    {
                        "error": f"GraphQL query failed with status {response.status_code}",
                        "status_code": response.status_code,
                        "details": (
                            response.body
                            if response.body
                            else "No error details provided"
                        ),
                    }
                )

            # Parse response body
            try:
                if isinstance(response.body, dict):
                    result = response.body
                else:
                    result = json.loads(response.body)

                # Check for GraphQL errors in the response
                if "errors" in result and result["errors"]:
                    return json.dumps(
                        {
                            "error": "GraphQL query returned errors",
                            "graphql_errors": result["errors"],
                            "data": result.get("data"),
                        }
                    )

                return json.dumps(
                    {
                        "status": "success",
                        "data": result.get("data"),
                    }
                )

            except json.JSONDecodeError as e:
                return json.dumps(
                    {
                        "error": "Failed to parse GraphQL response",
                        "details": str(e),
                    }
                )

        except Exception as e:
            log.error(
                "Exception during GitLab GraphQL query",
                extra={"error": str(e)},
                exc_info=True,
            )
            return json.dumps(
                {
                    "error": "GraphQL query failed with exception",
                    "details": str(e),
                }
            )

    def format_display_message(
        self, args: GitLabGraphQLInput, _tool_response: Any = None
    ) -> str:
        """Format a display message for the tool execution."""
        has_vars = " with variables" if args.variables else ""
        return f"Executing GitLab GraphQL query{has_vars}"
