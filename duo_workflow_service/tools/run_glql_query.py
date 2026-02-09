import json
from typing import Any, Type

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tracking.errors import log_exception
from lib.context import gitlab_version


class GLQLQueryInput(BaseModel):
    glql_yaml: str = Field(
        description="""Complete GLQL query block including all parameters.

        Example:
        ```glql
        display: table
        fields: title, state, author, created
        title: "All Issues in Test Project"
        sort: created desc
        limit: 100
        query: type = Issue and project = "gitlab-duo/test"
        ```

        The query parameter is required. All other parameters (display, fields, title, sort, limit, etc)
        are optional and will use API defaults if not specified.
        """
    )
    after: str | None = Field(
        default=None,
        description="""Cursor for forward pagination. Should use the 'endCursor' value from a previous query's
        pageInfo to fetch the next page of results.

        Example workflow:
        1. Execute initial query without 'after' parameter
        2. Check response pageInfo.hasNextPage
        3. If true, use pageInfo.endCursor as 'after' value for next query

        Leave empty for the first page of results.
        """,
    )


class RunGLQLQuery(DuoBaseTool):
    name: str = "run_glql_query"
    description: str = """Execute a GLQL (GitLab Query Language) query and return results with pagination support.

    Accepts a complete GLQL query block as a string. GLQL is transformed internally
    to GraphQL and returns structured data about GitLab issues, merge requests, and epics.

    Example GLQL query block:
    ```glql
    display: table
    fields: title, state, author, created
    sort: created desc
    limit: 100
    query: type = Issue and project = "gitlab-duo/test"
    ```

    Returns structured data including:
    - data.nodes: Array of results
    - data.count: Total count of matching items
    - data.pageInfo: Pagination information:
        - hasNextPage: Boolean indicating if more results exist
        - endCursor: Cursor to use for fetching next page (pass as 'after' parameter)
        - hasPreviousPage: Boolean indicating if previous page exists
        - startCursor: Cursor for the start of current page
    - fields: Field metadata

    For pagination, use the 'after' parameter with the endCursor from previous response.
    """
    args_schema: Type[BaseModel] = GLQLQueryInput
    handle_tool_error: bool = True

    async def _execute(self, glql_yaml: str, after: str | None = None) -> str:
        """Execute a GLQL query and return the results.

        Args:
            glql_yaml: Complete GLQL query block as a string
            after: Optional cursor for pagination (endCursor from previous response)

        Returns:
            JSON string containing query results and pagination info
        """
        # GLQL API is only available from 18.6+
        version_18_6 = Version("18.6.0")
        version_18_5 = Version("18.5.0")

        try:
            gl_version = Version(str(gitlab_version.get() or ""))
        except (InvalidVersion, TypeError) as ex:
            log_exception(ex)
            gl_version = version_18_5

        if gl_version < version_18_6:
            return json.dumps(
                {
                    "error": "GLQL API is only available in GitLab 18.6 and later. "
                    f"Current GitLab version: {gitlab_version.get() or 'unknown'}"
                }
            )

        try:
            request_body = {"glql_yaml": glql_yaml}
            if after:
                request_body["after"] = after

            response = await self.gitlab_client.apost(
                path="/api/v4/glql",
                body=json.dumps(request_body),
            )

            if not response.is_success():
                return json.dumps(
                    {
                        "error": f"GLQL API response status {response.status_code}: {response.body}"
                    }
                )

            return json.dumps(response.body)

        except ValueError as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: GLQLQueryInput, _tool_response: Any = None
    ) -> str:
        if args.after:
            return "Execute GLQL query (fetching next page)"
        return "Execute GLQL query"
