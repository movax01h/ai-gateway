import json
from collections import Counter
from enum import StrEnum
from typing import Any, Optional, Type

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.findings.queries.security_findings import (
    LIST_SECURITY_FINDINGS_QUERY,
)


class SecurityFindingState(StrEnum):
    """Valid security finding states."""

    DETECTED = "DETECTED"
    DISMISSED = "DISMISSED"
    CONFIRMED = "CONFIRMED"
    RESOLVED = "RESOLVED"


class SecurityFindingSeverity(StrEnum):
    """Valid security finding severity levels."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"
    UNKNOWN = "UNKNOWN"


class SecurityFindingReportType(StrEnum):
    """Valid security finding report types."""

    SAST = "SAST"
    DEPENDENCY_SCANNING = "DEPENDENCY_SCANNING"
    CONTAINER_SCANNING = "CONTAINER_SCANNING"
    DAST = "DAST"
    SECRET_DETECTION = "SECRET_DETECTION"
    COVERAGE_FUZZING = "COVERAGE_FUZZING"
    API_FUZZING = "API_FUZZING"
    CLUSTER_IMAGE_SCANNING = "CLUSTER_IMAGE_SCANNING"
    GENERIC = "GENERIC"


class ListSecurityFindingsInput(BaseModel):
    """Input for listing security findings from a pipeline."""

    project_full_path: str = Field(
        description="The full path of the GitLab project (e.g., 'namespace/project' or 'group/subgroup/project')",
    )
    pipeline_id: str = Field(description="Pipeline ID to get findings from (required)")
    severity: Optional[list[SecurityFindingSeverity | str]] = Field(
        default=None,
        description="Filter by severity levels (e.g., [CRITICAL, HIGH]). If not specified, all severities returned.",
    )
    report_type: Optional[list[SecurityFindingReportType | str]] = Field(
        default=None,
        description="Filter by report types (e.g., [SAST, DAST]). If not specified, all types returned.",
    )
    scanner: Optional[list[str]] = Field(
        default=None,
        description="Filter by scanner IDs. If not specified, all scanners returned.",
    )
    state: Optional[list[SecurityFindingState | str]] = Field(
        default=None,
        description="Filter by states (e.g., [DETECTED, DISMISSED]). Default includes all states.",
    )
    per_page: Optional[int] = Field(
        default=100,
        description="Number of results per page (default: 100, max: 100)",
        ge=1,
        le=100,
    )
    fetch_all_pages: Optional[bool] = Field(
        default=True,
        description="Whether to fetch all pages of results (default: True)",
    )
    include_dismissed: Optional[bool] = Field(
        default=True, description="Include dismissed findings (default: True)"
    )


class ListSecurityFindings(DuoBaseTool):
    """Tool for listing security findings from a specific pipeline scan."""

    name: str = "list_security_findings"
    description: str = """List ephemeral security findings from a specific GitLab pipeline security scan.

    Use this tool to see all potential vulnerabilities found in a single pipeline run, such as for a Merge Request.
    This tool requires a `pipeline_id` to operate.

    **Do NOT use this tool to list vulnerabilities for an entire project; use 'list_vulnerabilities' for that.**

    For example:
    - List all findings in a pipeline:
        list_security_findings(
            project_full_path="gitlab-org/gitlab",
            pipeline_id="gid://gitlab/Ci::Pipeline/12345"
        )

    - List only critical SAST findings:
        list_security_findings(
            project_full_path="gitlab-org/gitlab",
            pipeline_id="gid://gitlab/Ci::Pipeline/12345",
            severity=[SecurityFindingSeverity.CRITICAL],
            report_type=[SecurityFindingReportType.SAST]
        )

    - List non-dismissed findings:
        list_security_findings(
            project_full_path="gitlab-org/gitlab",
            pipeline_id="gid://gitlab/Ci::Pipeline/12345",
            state=[SecurityFindingState.DETECTED, SecurityFindingState.CONFIRMED]
        )
    """
    args_schema: Type[BaseModel] = ListSecurityFindingsInput

    def _normalize_enum_value(self, item: Any) -> str:
        """Convert enum or string to string value."""
        return item.value if isinstance(item, StrEnum) else item

    def _prepare_state_filter(
        self, include_dismissed: bool, state: Optional[list[SecurityFindingState | str]]
    ) -> Optional[list[SecurityFindingState | str]]:
        """Prepare the state filter based on include_dismissed flag."""
        if not include_dismissed and not state:
            return [
                SecurityFindingState.DETECTED,
                SecurityFindingState.CONFIRMED,
                SecurityFindingState.RESOLVED,
            ]
        return state

    def _build_query_variables(
        self,
        project_path: str,
        pipeline_id: str,
        per_page: int,
        cursor: Optional[str],
        severity: Optional[list[SecurityFindingSeverity | str]],
        report_type: Optional[list[SecurityFindingReportType | str]],
        scanner: Optional[list[str]],
        state: Optional[list[SecurityFindingState | str]],
    ) -> dict[str, Any]:
        """Build GraphQL query variables."""
        variables: dict[str, Any] = {
            "fullPath": project_path,
            "pipelineId": pipeline_id,
            "first": per_page,
        }

        if cursor:
            variables["after"] = cursor
        if severity:
            variables["severity"] = [self._normalize_enum_value(s) for s in severity]
        if report_type:
            variables["reportType"] = [
                self._normalize_enum_value(rt) for rt in report_type
            ]
        if scanner:
            variables["scanner"] = scanner
        if state:
            variables["state"] = [self._normalize_enum_value(s) for s in state]

        return variables

    def _extract_pipeline_info(self, pipeline_data: dict[str, Any]) -> dict[str, Any]:
        """Extract pipeline information from response."""
        return {
            "id": pipeline_data["id"],
            "iid": pipeline_data.get("iid"),
            "sha": pipeline_data.get("sha"),
            "ref": pipeline_data.get("ref"),
            "status": pipeline_data.get("status"),
        }

    def _compute_summary(
        self, findings: list[dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, list[str]]]:
        """Compute summary statistics and SAST file mapping."""
        severity_counts = Counter(f.get("severity", "UNKNOWN") for f in findings)
        report_type_counts = Counter(f.get("reportType", "UNKNOWN") for f in findings)
        state_counts = Counter(f.get("state", "UNKNOWN") for f in findings)

        ai_resolvable = sum(
            1 for f in findings if f.get("aiResolutionAvailable", False)
        )
        promoted = sum(1 for f in findings if f.get("vulnerability") is not None)
        false_positive = sum(1 for f in findings if f.get("falsePositive", False))

        sast_by_file: dict[str, list[str]] = {}
        for finding in findings:
            if finding.get("reportType") == "SAST":
                location = finding.get("location", {})
                file_path = location.get("file")
                if file_path:
                    if file_path not in sast_by_file:
                        sast_by_file[file_path] = []
                    sast_by_file[file_path].append(finding["uuid"])

        summary = {
            "total": len(findings),
            "by_severity": dict(severity_counts),
            "by_report_type": dict(report_type_counts),
            "by_state": dict(state_counts),
            "ai_resolvable": ai_resolvable,
            "promoted_to_vulnerability": promoted,
            "false_positives": false_positive,
            "sast_files_affected": len(sast_by_file),
        }

        return summary, sast_by_file

    async def _fetch_findings(
        self,
        project_path: str,
        pipeline_id: str,
        fetch_all_pages: bool,
        per_page: int,
        severity: Optional[list[SecurityFindingSeverity | str]],
        report_type: Optional[list[SecurityFindingReportType | str]],
        scanner: Optional[list[str]],
        state: Optional[list[SecurityFindingState | str]],
    ) -> dict[str, Any]:
        """Fetch findings from GitLab API with pagination."""
        all_findings = []
        cursor = None
        pipeline_info = None

        while True:
            variables = self._build_query_variables(
                project_path,
                pipeline_id,
                per_page,
                cursor,
                severity,
                report_type,
                scanner,
                state,
            )

            response = await self.gitlab_client.apost(
                path="/api/graphql",
                body=json.dumps(
                    {"query": LIST_SECURITY_FINDINGS_QUERY, "variables": variables}
                ),
            )

            if "errors" in response:
                return {"error": "GraphQL errors", "errors": response["errors"]}

            project_data = response.get("data", {}).get("project")
            if not project_data:
                return {
                    "error": "Project not found or access denied",
                    "project": project_path,
                }

            pipeline_data = project_data.get("pipeline")
            if not pipeline_data:
                return {
                    "error": "Pipeline not found",
                    "pipeline_id": pipeline_id,
                    "project": project_path,
                }

            if pipeline_info is None:
                pipeline_info = self._extract_pipeline_info(pipeline_data)

            findings_data = pipeline_data.get("securityReportFindings", {})
            findings = findings_data.get("nodes", [])
            all_findings.extend(findings)

            page_info = findings_data.get("pageInfo", {})
            has_next = page_info.get("hasNextPage", False)
            cursor = page_info.get("endCursor")

            if not fetch_all_pages or not has_next or not cursor:
                break

        return {"findings": all_findings, "pipeline_info": pipeline_info}

    async def _execute(self, **kwargs: Any) -> str:
        """Execute the security findings listing."""
        project_path = kwargs.pop("project_full_path")
        pipeline_id = kwargs.pop("pipeline_id")

        try:
            fetch_all_pages = kwargs.pop("fetch_all_pages", True)
            per_page = kwargs.pop("per_page", 100)
            severity = kwargs.pop("severity", None)
            report_type = kwargs.pop("report_type", None)
            scanner = kwargs.pop("scanner", None)
            include_dismissed = kwargs.pop("include_dismissed", True)
            state = kwargs.pop("state", None)

            state = self._prepare_state_filter(include_dismissed, state)

            result = await self._fetch_findings(
                project_path,
                pipeline_id,
                fetch_all_pages,
                per_page,
                severity,
                report_type,
                scanner,
                state,
            )

            if "error" in result:
                return json.dumps(result)

            findings = result["findings"]
            pipeline_info = result["pipeline_info"]

            summary, _sast_by_file = self._compute_summary(findings)

            return json.dumps(
                {
                    "findings": findings,
                    "summary": summary,
                    "pipeline": pipeline_info,
                    "metadata": {
                        "total_pages": 1 if not fetch_all_pages else "all",
                        "filters_applied": {
                            "severity": (
                                [self._normalize_enum_value(s) for s in severity]
                                if severity
                                else None
                            ),
                            "report_type": (
                                [self._normalize_enum_value(rt) for rt in report_type]
                                if report_type
                                else None
                            ),
                            "state": (
                                [self._normalize_enum_value(s) for s in state]
                                if state
                                else None
                            ),
                            "scanner": scanner,
                        },
                    },
                }
            )

        except Exception as e:
            raise ToolException(f"Failed to list security findings: {str(e)}")

    def format_display_message(
        self, args: ListSecurityFindingsInput, _tool_response: Any = None
    ) -> str:
        """Format user-friendly display message."""
        message = f"List security findings from pipeline {args.pipeline_id} in {args.project_full_path}"

        filters = []
        if args.severity:
            filters.append(
                f"severity: {', '.join([self._normalize_enum_value(s) for s in args.severity])}"
            )
        if args.report_type:
            filters.append(
                f"type: {', '.join([self._normalize_enum_value(rt) for rt in args.report_type])}"
            )
        if args.state:
            filters.append(
                f"state: {', '.join([self._normalize_enum_value(s) for s in args.state])}"
            )

        if filters:
            message += f" ({', '.join(filters)})"

        return message
