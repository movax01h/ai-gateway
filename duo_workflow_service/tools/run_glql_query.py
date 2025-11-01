import json
from typing import Any, Type

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field

from ai_gateway.instrumentators.model_requests import gitlab_version
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tracking.errors import log_exception


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


class RunGLQLQuery(DuoBaseTool):
    name: str = "run_glql_query"
    description: str = """Execute a GLQL (GitLab Query Language) query and return results.

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

    Returns structured data including query results, pagination info, and field metadata.
    """
    args_schema: Type[BaseModel] = GLQLQueryInput
    handle_tool_error: bool = True

    async def _execute(self, glql_yaml: str) -> str:
        """Execute a GLQL query and return the results."""
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
            response = await self.gitlab_client.apost(
                path="/api/v4/glql",
                body=json.dumps({"glql_yaml": glql_yaml}),
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
        return "Execute GLQL query"
