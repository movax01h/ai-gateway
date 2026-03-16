from typing import Any, Literal, Optional, Type

from pydantic import BaseModel, Field

from duo_workflow_service.tools.ascp.queries import CREATE_ASCP_SCAN_MUTATION
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

ScanTypeLiteral = Literal["FULL", "INCREMENTAL"]


class CreateAscpScanResponseBody(BaseModel):
    """Nested response: scan entity and raw API payload."""

    scan: Optional[dict[str, Any]] = None
    raw_response: Optional[dict[str, Any]] = None


class CreateAscpScanResponse(BaseModel):
    """Unified response shape for success and error."""

    errors: list[str] = Field(default_factory=list)
    response: CreateAscpScanResponseBody = Field(
        default_factory=CreateAscpScanResponseBody
    )


class CreateAscpScanInput(BaseModel):
    """Input model for the CreateAscpScan tool."""

    project_path: str = Field(
        description='Full path of the project (e.g., "namespace/project").',
    )
    commit_sha: str = Field(
        description="Commit SHA for the scan (e.g., full 40-character Git SHA).",
    )
    scan_type: Optional[ScanTypeLiteral] = Field(
        default="FULL",
        description='Type of scan: "FULL" or "INCREMENTAL". Sent as scanType to the API. Defaults to "FULL".',
    )
    base_scan_id: Optional[str] = Field(
        default=None,
        description="GraphQL ID of the base scan (for INCREMENTAL scans).",
    )
    base_commit_sha: Optional[str] = Field(
        default=None,
        description="Base commit SHA (for INCREMENTAL scans).",
    )


class CreateAscpScan(DuoBaseTool):
    """Tool for creating an ASCP (Application Security Collaboration Platform) scan.

    Returned JSON uses a single shape for success and error: {"errors": list[str],
    "response": {"scan": ... | null, "raw_response": ... | null}}. Success when
    errors is empty and response.scan is set; on error, errors is non-empty and
    response contains scan and/or raw_response (raw API payload) when available.
    """

    name: str = "ascp_create_scan"
    description: str = """
    Create a new ASCP scan for a project at a given commit.

    Use this tool when you need to record a full or incremental ASCP scan for
    a GitLab project. Provide the project full path (e.g., 'namespace/project'),
    the commit SHA to scan, and optionally scan_type ('FULL' or 'INCREMENTAL').
    For incremental scans, you can optionally provide base_scan_id and base_commit_sha.

    Example:
        ascp_create_scan(
            project_path="my-group/my-project",
            commit_sha="abc123def456...",
            scan_type="FULL"
        )
    """
    args_schema: Type[BaseModel] = CreateAscpScanInput

    def format_display_message(
        self, args: CreateAscpScanInput, _tool_response: Any = None
    ) -> str:
        return f"Create ASCP scan for {args.project_path} at {args.commit_sha}"

    async def _execute(self, **kwargs: Any) -> str:
        project_path = kwargs["project_path"]
        commit_sha = kwargs["commit_sha"]
        scan_type = kwargs.get("scan_type", "FULL")
        base_scan_id = kwargs.get("base_scan_id")
        base_commit_sha = kwargs.get("base_commit_sha")

        input_data: dict[str, Any] = {
            "projectPath": project_path,
            "commitSha": commit_sha,
            "scanType": scan_type,
        }

        if base_scan_id is not None:
            input_data["baseScanId"] = base_scan_id
        if base_commit_sha is not None:
            input_data["baseCommitSha"] = base_commit_sha

        variables = {"input": input_data}

        try:
            response = await self.gitlab_client.graphql(
                CREATE_ASCP_SCAN_MUTATION,
                variables,
            )
        except Exception as e:
            return CreateAscpScanResponse(
                errors=[
                    f"ascp_create_scan failed: {type(e).__name__}: {e!s}",
                ],
                response=CreateAscpScanResponseBody(scan=None, raw_response=None),
            ).model_dump_json()

        if not isinstance(response, dict):
            return CreateAscpScanResponse(
                errors=["GraphQL returned no response or invalid format"],
                response=CreateAscpScanResponseBody(scan=None, raw_response=None),
            ).model_dump_json()

        payload = response.get("ascpScanCreate") or {}

        scan = payload.get("scan")
        errors = payload.get("errors")

        if errors:
            if not isinstance(errors, list):
                errors = [str(errors)]
            return CreateAscpScanResponse(
                errors=errors,
                response=CreateAscpScanResponseBody(scan=scan, raw_response=payload),
            ).model_dump_json()

        if not scan or not scan.get("id"):
            return CreateAscpScanResponse(
                errors=["Failed to create ASCP scan."],
                response=CreateAscpScanResponseBody(scan=scan, raw_response=payload),
            ).model_dump_json()

        return CreateAscpScanResponse(
            errors=[],
            response=CreateAscpScanResponseBody(scan=scan, raw_response=None),
        ).model_dump_json()
