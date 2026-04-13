from typing import Any, ClassVar, Optional, Type

from pydantic import BaseModel, Field

from duo_workflow_service.tools.ascp.queries import LIST_ASCP_SCANS_QUERY
from duo_workflow_service.tools.ascp.types import ScanTypeLiteral
from duo_workflow_service.tools.duo_base_tool import (
    LICENSED_FEATURE_SECURITY_DASHBOARD,
    DuoBaseTool,
)

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100


class ListAscpScansResponseBody(BaseModel):
    """Nested response: scans list, page_info, and raw API payload."""

    scans: Optional[list[dict[str, Any]]] = None
    page_info: Optional[dict[str, Any]] = None
    raw_response: Optional[dict[str, Any]] = None


class ListAscpScansResponse(BaseModel):
    """Unified response shape for success and error."""

    errors: list[str] = Field(default_factory=list)
    response: ListAscpScansResponseBody = Field(
        default_factory=ListAscpScansResponseBody
    )


class ListAscpScansInput(BaseModel):
    """Input model for the ListAscpScans tool."""

    project_path: str = Field(
        description='Full path of the project (e.g., "namespace/project").',
    )
    scan_type: Optional[ScanTypeLiteral] = Field(
        default=None,
        description='Optional filter: "FULL" or "INCREMENTAL". Omit to list all scans.',
    )
    first: int = Field(
        default=DEFAULT_PAGE_SIZE,
        ge=1,
        le=MAX_PAGE_SIZE,
        description=f"Number of scans per page (default {DEFAULT_PAGE_SIZE}, max {MAX_PAGE_SIZE}).",
    )
    after: Optional[str] = Field(
        default=None,
        description="Cursor for pagination (from previous response page_info.end_cursor).",
    )


class ListAscpScans(DuoBaseTool):
    """Tool for listing ASCP (Application Security Context and Patterns) scans for a project.

    Returned JSON uses a single shape for success and error: {"errors": list[str],
    "response": {"scans": ... | null, "page_info": ... | null, "raw_response": ... | null}}.
    Success when errors is empty and response.scans / response.page_info are set; on error,
    errors is non-empty and response contains raw_response (raw GraphQL payload) when available.
    """

    tier_check_licensed_feature: ClassVar[str] = LICENSED_FEATURE_SECURITY_DASHBOARD

    name: str = "ascp_list_scans"
    description: str = """
    List ASCP scans for a GitLab project.

    Use this tool when you need to see existing full or incremental ASCP scans for
    a project. Provide the project full path (e.g., 'namespace/project').
    Optionally filter by scan_type ('FULL' or 'INCREMENTAL') and use first/after
    for pagination. To create a new scan, use ascp_create_scan instead.

    Example:
        ascp_list_scans(project_path="my-group/my-project")
        ascp_list_scans(project_path="my-group/my-project", scan_type="FULL", first=10)
    """
    args_schema: Type[BaseModel] = ListAscpScansInput

    def format_display_message(
        self, args: ListAscpScansInput, _tool_response: Any = None
    ) -> str:
        if args.scan_type:
            return f"List ASCP scans for {args.project_path} (type={args.scan_type})"
        return f"List ASCP scans for {args.project_path}"

    async def _execute(self, **kwargs: Any) -> str:
        project_path = kwargs["project_path"]
        scan_type = kwargs.get("scan_type")
        first = kwargs.get("first", DEFAULT_PAGE_SIZE)
        after = kwargs.get("after")

        variables: dict[str, Any] = {
            "fullPath": project_path,
            "first": first,
        }
        if scan_type is not None:
            variables["scanType"] = scan_type
        if after is not None:
            variables["after"] = after

        try:
            response = await self.gitlab_client.graphql(
                LIST_ASCP_SCANS_QUERY,
                variables,
            )
        except Exception as e:
            return ListAscpScansResponse(
                errors=[
                    f"ascp_list_scans failed: {type(e).__name__}: {e!s}",
                ],
                response=ListAscpScansResponseBody(
                    scans=None, page_info=None, raw_response=None
                ),
            ).model_dump_json()

        if not isinstance(response, dict):
            return ListAscpScansResponse(
                errors=["GraphQL returned no response or invalid format"],
                response=ListAscpScansResponseBody(
                    scans=None, page_info=None, raw_response=None
                ),
            ).model_dump_json()

        graphql_errors = response.get("errors")
        if graphql_errors:
            if not isinstance(graphql_errors, list):
                graphql_errors = [str(graphql_errors)]
            else:
                graphql_errors = [
                    e.get("message", str(e)) if isinstance(e, dict) else str(e)
                    for e in graphql_errors
                ]
            return ListAscpScansResponse(
                errors=graphql_errors,
                response=ListAscpScansResponseBody(
                    scans=None, page_info=None, raw_response=response
                ),
            ).model_dump_json()

        project = response.get("project")
        if project is None:
            return ListAscpScansResponse(
                errors=["Project not found or access denied"],
                response=ListAscpScansResponseBody(
                    scans=None, page_info=None, raw_response=response
                ),
            ).model_dump_json()

        ascp_scans = project.get("ascpScans") or {}
        nodes = ascp_scans.get("nodes") or []
        page_info = ascp_scans.get("pageInfo") or {}

        page_info_dict = {
            "has_next_page": page_info.get("hasNextPage", False),
            "end_cursor": page_info.get("endCursor"),
        }
        return ListAscpScansResponse(
            errors=[],
            response=ListAscpScansResponseBody(
                scans=nodes, page_info=page_info_dict, raw_response=None
            ),
        ).model_dump_json()
