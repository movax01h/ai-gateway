import json
from collections import Counter
from enum import StrEnum
from typing import Any, Optional, Type

from packaging.version import InvalidVersion, Version
from pydantic import BaseModel, Field, field_validator

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.vulnerabilities.queries.vulnerabilities import (
    LIST_VULNERABILITIES_QUERY,
)
from duo_workflow_service.tracking.errors import log_exception
from lib.context import gitlab_version

PROJECT_IDENTIFICATION_DESCRIPTION = """
The project must be specified using its full path (e.g., 'namespace/project' or 'group/subgroup/project').
"""


class VulnerabilitySeverity(StrEnum):
    """Valid vulnerability severity levels."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"
    UNKNOWN = "UNKNOWN"


class VulnerabilityReportType(StrEnum):
    """Valid vulnerability report types."""

    SAST = "SAST"
    DEPENDENCY_SCANNING = "DEPENDENCY_SCANNING"
    CONTAINER_SCANNING = "CONTAINER_SCANNING"
    DAST = "DAST"
    SECRET_DETECTION = "SECRET_DETECTION"
    COVERAGE_FUZZING = "COVERAGE_FUZZING"
    API_FUZZING = "API_FUZZING"
    CLUSTER_IMAGE_SCANNING = "CLUSTER_IMAGE_SCANNING"
    CONTAINER_SCANNING_FOR_REGISTRY = "CONTAINER_SCANNING_FOR_REGISTRY"
    GENERIC = "GENERIC"


class FpDetectionStatus(StrEnum):
    """Valid FP detection status values."""

    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    DETECTED_AS_FP = "DETECTED_AS_FP"
    DETECTED_AS_NOT_FP = "DETECTED_AS_NOT_FP"
    FAILED = "FAILED"


__all__ = [
    "ListVulnerabilities",
    "DismissVulnerability",
    "LinkVulnerabilityToIssue",
    "LinkVulnerabilityToMergeRequest",
    "ConfirmVulnerability",
    "RevertToDetectedVulnerability",
    "CreateVulnerabilityIssue",
]


class ListVulnerabilitiesInput(BaseModel):
    """Input validation for list vulnerabilities operation."""

    project_full_path: str = Field(
        description="The full path of the GitLab project (e.g., 'namespace/project' or 'group/subgroup/project')",
    )
    severity: Optional[list[VulnerabilitySeverity]] = Field(
        default=None,
        description="""
        Filter vulnerabilities by severity levels. Can specify multiple values (e.g., [CRITICAL, HIGH, MEDIUM]).
        If not specified, all severities will be returned.
        """,
    )
    report_type: Optional[list[VulnerabilityReportType]] = Field(
        default=None,
        description="""
        Filter vulnerabilities by report types. Can specify multiple values (e.g., [SAST, DAST]).
        If not specified, all report types will be returned.
        """,
    )
    per_page: Optional[int] = Field(
        default=100,
        description="Number of results per page (default: 100, max: 100).",
        ge=1,
        le=100,
    )
    page: Optional[int] = Field(
        default=1,
        description="Page number to fetch (default: 1).",
        ge=1,
    )
    fetch_all_pages: Optional[bool] = Field(
        default=True,
        description="Whether to fetch all pages of results (default: True).",
    )

    @field_validator("project_full_path")
    @classmethod
    def validate_project_path(cls, v: str) -> str:
        """Basic validation for project path."""
        if not v or len(v) < 3:
            raise ValueError("Project path must be at least 3 characters")
        if ".." in v or v.startswith("/") or v.endswith("/"):
            raise ValueError("Invalid project path format")
        return v


class ListVulnerabilities(DuoBaseTool):
    """Tool for listing persisted project vulnerabilities."""

    name: str = "list_vulnerabilities"
    description: str = f"""List persisted vulnerabilities for an entire project from its main Vulnerability Report.

    {PROJECT_IDENTIFICATION_DESCRIPTION}

    Use this tool to get an overview of the project's current security debt on its default branch.
    This tool operates on the project level and does not require a pipeline ID.

    The tool supports filtering vulnerabilities by:
    - Severity levels (can specify multiple: CRITICAL, HIGH, MEDIUM, LOW, INFO, UNKNOWN)
    - Report type (SAST, DAST, DEPENDENCY_SCANNING, etc.)

    The tool exposes FP detection metadata in the latestFlag field including:
    - Detection status and confidence scores
    - AI reasoning for FP detection
    - Detection service origin and timestamps

    **Do NOT use this tool to list security findings from a specific pipeline; to list security findings from a specific pipeline, use the 'list_security_findings' tool.**

    For example:
    - List all vulnerabilities in a project:
        list_vulnerabilities(project_full_path="namespace/project")

    - List only critical and high vulnerabilities in a project:
        list_vulnerabilities(
            project_full_path="namespace/project",
            severity=[VulnerabilitySeverity.CRITICAL, VulnerabilitySeverity.HIGH]
        )

    - List only SAST vulnerabilities in a project:
        list_vulnerabilities(
            project_full_path="namespace/project",
            report_type=[VulnerabilityReportType.SAST]
        )

    - List only critical SAST vulnerabilities in a project:
        list_vulnerabilities(
            project_full_path="namespace/project",
            severity=[VulnerabilitySeverity.CRITICAL]
            report_type=[VulnerabilityReportType.SAST]
        )
    """
    args_schema: Type[BaseModel] = ListVulnerabilitiesInput

    async def _execute(self, **kwargs: Any) -> str:
        """Execute the vulnerability listing."""
        try:
            project_full_path = kwargs.pop("project_full_path")
            fetch_all_pages = kwargs.pop("fetch_all_pages", True)
            per_page = kwargs.pop("per_page", 100)
            severity = kwargs.pop("severity", None)
            report_type = kwargs.pop("report_type", None)

            all_vulnerabilities: list[dict[str, Any]] = []
            cursor = None

            while True:
                variables = {
                    "projectFullPath": project_full_path,
                    "first": per_page,
                }

                if cursor is not None:
                    variables["after"] = cursor

                if severity:
                    variables["severity"] = [s.value for s in severity]

                if report_type:
                    variables["reportType"] = [rt.value for rt in report_type]

                response = await self.gitlab_client.apost(
                    path="/api/graphql",
                    body=json.dumps(
                        {"query": LIST_VULNERABILITIES_QUERY, "variables": variables}
                    ),
                )

                response = self._process_http_response(
                    identifier="query", response=response
                )

                # Parse JSON if response is a string
                if isinstance(response, str):
                    response = json.loads(response)

                if not response or "data" not in response:
                    raise ValueError("Invalid GraphQL response")

                project_data = response.get("data", {}).get("project")
                if not project_data:
                    return json.dumps(
                        {
                            "error": "Project not found or access denied",
                            "project_path": project_full_path,
                        }
                    )

                vulnerabilities_data = project_data.get("vulnerabilities", {})
                vulnerabilities = vulnerabilities_data.get("nodes") or []

                all_vulnerabilities.extend(vulnerabilities)

                page_info = vulnerabilities_data.get("pageInfo", {})
                if not fetch_all_pages or not page_info.get("hasNextPage"):
                    break

                cursor = page_info.get("endCursor")
                if not cursor:
                    break

            severity_counts = Counter(
                vuln.get("severity", "UNKNOWN") for vuln in all_vulnerabilities
            )
            report_type_counts = Counter(
                vuln.get("reportType", "UNKNOWN") for vuln in all_vulnerabilities
            )
            state_counts = Counter(
                vuln.get("state", "UNKNOWN") for vuln in all_vulnerabilities
            )

            return json.dumps(
                {
                    "vulnerabilities": all_vulnerabilities,
                    "summary": {
                        "total": len(all_vulnerabilities),
                        "by_severity": dict(severity_counts),
                        "by_report_type": dict(report_type_counts),
                        "by_state": dict(state_counts),
                    },
                    "pagination": {
                        "total_items": len(all_vulnerabilities),
                    },
                }
            )

        except Exception as e:
            return json.dumps(
                {
                    "error": "An error occurred while listing vulnerabilities",
                    "error_type": type(e).__name__,
                }
            )

    def format_display_message(
        self, args: ListVulnerabilitiesInput, _tool_response: Any = None
    ) -> str:
        """Format a user-friendly display message."""
        message = f"List vulnerabilities in project {args.project_full_path}"
        filters = []

        if args.severity:
            filters.append(f"severity: {', '.join([s.value for s in args.severity])}")
        if args.report_type:
            filters.append(
                f"report type: {', '.join([rt.value for rt in args.report_type])}"
            )

        if filters:
            message += f" ({', '.join(filters)})"

        return message


class DismissVulnerabilityInput(BaseModel):
    vulnerability_id: str = Field(description="ID of the vulnerability to be dismissed")
    comment: str = Field(
        description="Comment why vulnerability was dismissed (maximum 50,000 characters)."
    )
    dismissal_reason: str = Field(
        description="Reason why vulnerability should be dismissed (ACCEPTABLE_RISK, FALSE_POSITIVE, MITIGATING_CONTROL,"
        " USED_IN_TESTS, NOT_APPLICABLE)"
    )


class DismissVulnerability(DuoBaseTool):
    name: str = "dismiss_vulnerability"
    description: str = f"""Dismiss a security vulnerability in a GitLab project using GraphQL.

    {PROJECT_IDENTIFICATION_DESCRIPTION}

    The tool supports dismissing a vulnerability by ID, with a dismissal reason, and comment.
    The dismiss reason must be one of: ACCEPTABLE_RISK, FALSE_POSITIVE, MITIGATING_CONTROL, USED_IN_TESTS, NOT_APPLICABLE.
    If a dismissal reason is not given, you will need to ask for one.

    A comment explaining the reason for the dismissal is required and can be up to 50,000 characters.
    If a comment is not given, you will need to ask for one.

    For example:
    - Dismiss a vulnerability for being a false positive:
        dismiss_vulnerability(
            vulnerability_id="gid://gitlab/Vulnerability/123",
            dismissal_reason="FALSE_POSITIVE",
            comment="Security review deemed this a false positive"
        )
    """
    args_schema: Type[BaseModel] = DismissVulnerabilityInput

    async def _execute(self, **kwargs: Any) -> str:
        vulnerability_id = kwargs.pop("vulnerability_id")
        comment = kwargs.pop("comment")
        dismissal_reason = kwargs.pop("dismissal_reason")

        # Validate severity value
        valid_dismissal_reasons = {
            "ACCEPTABLE_RISK",
            "FALSE_POSITIVE",
            "MITIGATING_CONTROL",
            "USED_IN_TESTS",
            "NOT_APPLICABLE",
        }
        if dismissal_reason not in valid_dismissal_reasons:
            return json.dumps(
                {
                    "error": f"""
                        Invalid dismissal reason '{dismissal_reason}'.
                        Must be one of: {', '.join(valid_dismissal_reasons)}
                        """
                }
            )

        # Validate comment length
        if len(comment) > 50000:
            return json.dumps({"error": "Comment must be 50,000 characters or less"})

        # editorconfig-checker-disable
        # Build GraphQL mutation
        mutation = """
mutation($vulnerabilityId: VulnerabilityID!, $comment: String, $dismissalReason: VulnerabilityDismissalReason) {
    vulnerabilityDismiss(input: {
    id: $vulnerabilityId,
    comment: $comment,
    dismissalReason: $dismissalReason
    }) {
    errors
    vulnerability {
        id
        description
        state
        dismissedAt
        dismissalReason
    }
    }
}
"""
        # editorconfig-checker-enable

        # Ensure vulnerability_id has proper GraphQL format
        if not vulnerability_id.startswith("gid://gitlab/Vulnerability/"):
            vulnerability_id = f"gid://gitlab/Vulnerability/{vulnerability_id}"

        variables = {
            "vulnerabilityId": vulnerability_id,
            "comment": comment,
            "dismissalReason": dismissal_reason,
        }

        response = await self.gitlab_client.apost(
            path="/api/graphql",
            body=json.dumps({"query": mutation, "variables": variables}),
        )

        response = self._process_http_response(identifier="mutation", response=response)

        errors = response["data"]["vulnerabilityDismiss"]["errors"]
        if errors:
            return json.dumps({"error": "; ".join(errors)})

        return json.dumps(
            {"vulnerability": response["data"]["vulnerabilityDismiss"]["vulnerability"]}
        )

    def format_display_message(
        self, args: DismissVulnerabilityInput, _tool_response: Any = None
    ) -> str:
        return f"Dismiss vulnerability {args.vulnerability_id}"


class CreateVulnerabilityIssueInput(BaseModel):
    project_full_path: str = Field(
        description="The full path of the GitLab project (e.g., 'namespace/project' or 'group/subgroup/project')",
    )
    vulnerability_ids: list[str] = Field(
        description="Array of vulnerability IDs that will be linked to the created issue. Up to 100 can be provided."
    )


class CreateVulnerabilityIssue(DuoBaseTool):
    name: str = "create_vulnerability_issue"
    description: str = f"""Create a GitLab issue linked to security vulnerabilities in a GitLab project using GraphQL.

    {PROJECT_IDENTIFICATION_DESCRIPTION}

    The tool supports creating a GitLab issue linked to vulnerabilities by ID.
    Up to 100 IDs of vulnerabilities can be provided.

    For example:
    - Create an issue for project ID 1 linked with vulnerabilities with ID 2 and 3:
        create_vulnerability_issue(
            project_full_path="namespace/project",
            vulnerability_ids=["gid://gitlab/Vulnerability/2", "gid://gitlab/Vulnerability/3"]
        )
    """
    args_schema: Type[BaseModel] = CreateVulnerabilityIssueInput
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    # pylint: disable-next=too-many-return-statements
    async def _execute(self, **kwargs: Any) -> str:
        project_full_path = kwargs.pop("project_full_path")
        vulnerability_ids = kwargs.pop("vulnerability_ids")

        project_query = """
        query($projectFullPath: ID!) {
            project(fullPath: $projectFullPath) {
                id
            }
        }
        """

        project_variables = {"projectFullPath": project_full_path}

        project_response = await self.gitlab_client.apost(
            path="/api/graphql",
            body=json.dumps({"query": project_query, "variables": project_variables}),
        )

        try:
            project_response = self._process_http_response(
                identifier="query", response=project_response
            )
        except ValueError as e:
            return json.dumps({"error": str(e)})

        if not project_response or "data" not in project_response:
            return json.dumps(
                {
                    "error": "Invalid GraphQL response",
                    "project_path": project_full_path,
                }
            )

        project_data = project_response.get("data", {}).get("project")
        if not project_data:
            return json.dumps(
                {
                    "error": "Project not found or access denied",
                    "project_path": project_full_path,
                }
            )

        project_id = project_data["id"]

        mutation = """
        mutation($vulnerabilityIds: [VulnerabilityID!]!, $projectId: ProjectID!) {
            vulnerabilitiesCreateIssue(input: { project: $projectId, vulnerabilityIds: $vulnerabilityIds }) {
                issue {
                    id,
                    title,
                    name
                }
                errors
            }
        }
        """

        vulnerability_ids = [
            (
                f"gid://gitlab/Vulnerability/{vid}"
                if not str(vid).startswith("gid://gitlab/Vulnerability/")
                else str(vid)
            )
            for vid in vulnerability_ids
        ]

        variables = {"projectId": project_id, "vulnerabilityIds": vulnerability_ids}

        response = await self.gitlab_client.apost(
            path="/api/graphql",
            body=json.dumps({"query": mutation, "variables": variables}),
        )

        try:
            response = self._process_http_response(
                identifier="mutation", response=response
            )
        except ValueError as e:
            return json.dumps({"error": str(e)})

        if (
            not response
            or "data" not in response
            or "vulnerabilitiesCreateIssue" not in response["data"]
        ):
            return json.dumps({"error": "Invalid GraphQL response"})

        errors = response["data"]["vulnerabilitiesCreateIssue"]["errors"]
        if errors:
            return json.dumps({"error": "; ".join(errors)})

        return json.dumps(
            {"issue": response["data"]["vulnerabilitiesCreateIssue"]["issue"]}
        )

    def format_display_message(
        self, args: CreateVulnerabilityIssueInput, _tool_response: Any = None
    ) -> str:
        return f"Create issue for vulnerabilities in project {args.project_full_path}"


class LinkVulnerabilityToIssueInput(BaseModel):
    issue_id: str = Field(description="ID of the issue to link to.")
    vulnerability_ids: list[str] = Field(
        description="Array of vulnerability IDs to link to the given issue. Up to 100 can be provided."
    )


class LinkVulnerabilityToIssue(DuoBaseTool):
    name: str = "link_vulnerability_to_issue"
    description: str = f"""Link a GitLab issue to security vulnerabilities in a GitLab project using GraphQL.

    {PROJECT_IDENTIFICATION_DESCRIPTION}

    The tool supports linking a GitLab issue to vulnerabilities by ID.
    Up to 100 IDs of vulnerabilities can be provided.

    For example:
    - Link issue with ID 1 to a vulnerabilities with ID 23 and 10:
        link_vulnerability_to_issue(
            issue_id="gid://gitlab/Issue/1",
            vulnerability_ids=["gid://gitlab/Vulnerability/23", "gid://gitlab/Vulnerability/10"]
        )
    """
    args_schema: Type[BaseModel] = LinkVulnerabilityToIssueInput

    async def _execute(self, **kwargs: Any) -> str:
        issue_id = kwargs.pop("issue_id")
        vulnerability_ids = kwargs.pop("vulnerability_ids")

        # editorconfig-checker-disable
        # Build GraphQL mutation
        mutation = """
        mutation($vulnerabilityIds: [VulnerabilityID!]!, $issueId: IssueID!) {
          vulnerabilityIssueLinkCreate(input: { issueId: $issueId, vulnerabilityIds: $vulnerabilityIds }) {
            issueLinks {
              id
              issue {
                id,
                title,
                name
              }
              linkType
            }
            errors
          }
        }
        """
        # editorconfig-checker-enable

        # Ensure vulnerability_ids have proper GraphQL format
        vulnerability_ids = [
            (
                f"gid://gitlab/Vulnerability/{vid}"
                if not str(vid).startswith("gid://gitlab/Vulnerability/")
                else str(vid)
            )
            for vid in vulnerability_ids
        ]

        issue_id = (
            f"gid://gitlab/Issue/{issue_id}"
            if not str(issue_id).startswith("gid://gitlab/Issue/")
            else str(issue_id)
        )

        variables = {"issueId": issue_id, "vulnerabilityIds": vulnerability_ids}

        response = await self.gitlab_client.apost(
            path="/api/graphql",
            body=json.dumps({"query": mutation, "variables": variables}),
        )

        try:
            response = self._process_http_response(
                identifier="mutation", response=response
            )
        except ValueError as e:
            return json.dumps({"error": str(e)})

        errors = response["data"]["vulnerabilityIssueLinkCreate"]["errors"]
        if errors:
            return json.dumps({"error": "; ".join(errors)})

        return json.dumps(
            {
                "issueLinks": response["data"]["vulnerabilityIssueLinkCreate"][
                    "issueLinks"
                ]
            }
        )

    def format_display_message(
        self, args: LinkVulnerabilityToIssueInput, _tool_response: Any = None
    ) -> str:
        return f"Link issue to vulnerability {args.vulnerability_ids}"


class LinkVulnerabilityToMergeRequestInput(BaseModel):
    vulnerability_id: str = Field(
        description="ID of the vulnerability to link to the merge request"
    )
    merge_request_id: str = Field(
        description="ID of the merge request to link to the vulnerability"
    )

    @field_validator("vulnerability_id", "merge_request_id", mode="before")
    @classmethod
    def coerce_to_string(cls, v):
        return str(v)


class LinkVulnerabilityToMergeRequest(DuoBaseTool):
    name: str = "link_vulnerability_to_merge_request"
    description: str = f"""Link a security vulnerability to a merge request in a GitLab project using GraphQL.

    {PROJECT_IDENTIFICATION_DESCRIPTION}

    The tool supports linking a vulnerability to a merge request by their respective IDs.
    This creates a relationship between the vulnerability and the merge request that addresses it.
    The Merge Request ID used is the global ID, not the IID.

    The merge request ID given must include `gid://gitlab/MergeRequest/` in the prefix.
    If the ID does not include `gid://gitlab/MergeRequest/` in the prefix:
        - ASK THE USER WHAT THEY HAVE GIVEN YOU
        - If they have given you the MR IID (which is what is shown in the UI), fetch the ID

    For example:
    - Link vulnerability with ID 123 to merge request with ID 456 (IID 245):
        link_vulnerability_to_merge_request(
            vulnerability_id="gid://gitlab/Vulnerability/123",
            merge_request_id="gid://gitlab/MergeRequest/456"
        )
    """
    args_schema: Type[BaseModel] = LinkVulnerabilityToMergeRequestInput

    async def _execute(self, **kwargs: Any) -> str:
        version_18_2 = Version("18.2.0")
        version_18_5 = Version("18.5.0")

        try:
            gl_version = Version(gitlab_version.get())  # type: ignore[arg-type]
        except (InvalidVersion, TypeError) as ex:
            log_exception(ex)
            gl_version = version_18_2

        if gl_version < version_18_5:
            return json.dumps({"error": "This tool is not available"})

        vulnerability_id = str(kwargs.pop("vulnerability_id"))
        merge_request_id = str(kwargs.pop("merge_request_id"))

        # Ensure vulnerability_id has proper GraphQL format
        if not vulnerability_id.startswith("gid://gitlab/Vulnerability/"):
            vulnerability_id = f"gid://gitlab/Vulnerability/{vulnerability_id}"

        # Ensure merge_request_id has proper GraphQL format
        if not merge_request_id.startswith("gid://gitlab/MergeRequest/"):
            merge_request_id = f"gid://gitlab/MergeRequest/{merge_request_id}"

        # editorconfig-checker-disable
        # Build GraphQL mutation
        mutation = """
        mutation($vulnerabilityId: VulnerabilityID!, $mergeRequestId: MergeRequestID!) {
          vulnerabilityLinkMergeRequest(input: {
            vulnerabilityId: $vulnerabilityId,
            mergeRequestId: $mergeRequestId
          }) {
            vulnerability {
              id
              mergeRequests {
                nodes {
                  id
                  title
                }
              }
            }
            errors
          }
        }
        """
        # editorconfig-checker-enable

        variables = {
            "vulnerabilityId": vulnerability_id,
            "mergeRequestId": merge_request_id,
        }

        response = await self.gitlab_client.apost(
            path="/api/graphql",
            body=json.dumps({"query": mutation, "variables": variables}),
        )

        response = self._process_http_response(identifier="mutation", response=response)

        try:
            errors = response["data"]["vulnerabilityLinkMergeRequest"]["errors"]
            if errors:
                return json.dumps({"error": "; ".join(errors)})

            return json.dumps(
                {
                    "vulnerability": response["data"]["vulnerabilityLinkMergeRequest"][
                        "vulnerability"
                    ]
                }
            )
        except KeyError as e:
            return json.dumps(
                {
                    "error": f"Unexpected response structure: {str(e)}",
                    "response": response,
                }
            )

    def format_display_message(
        self, args: LinkVulnerabilityToMergeRequestInput, _tool_response: Any = None
    ) -> str:
        return f"Link vulnerability {args.vulnerability_id} to merge request {args.merge_request_id}"


class ConfirmVulnerabilityInput(BaseModel):
    vulnerability_id: str = Field(
        description="The ID of the vulnerability to be confirmed "
        "(e.g., either digit ids like '123' or gid like 'gid://gitlab/Vulnerability/123')",
    )
    comment: Optional[str] = Field(
        default=None,
        description="Comment explaining why the vulnerability was confirmed (maximum 50,000 characters)",
    )


class ConfirmVulnerability(DuoBaseTool):
    name: str = "confirm_vulnerability"
    description: str = """Confirm a security vulnerability in a GitLab project.

This tool marks a vulnerability as confirmed, changing its state to CONFIRMED.
This is typically done when a security team has verified that the vulnerability is a real issue
that needs to be addressed.
"""
    args_schema: Type[BaseModel] = ConfirmVulnerabilityInput

    async def _execute(self, **kwargs: Any) -> str:
        vulnerability_id = kwargs.pop("vulnerability_id")
        comment = kwargs.pop("comment", None)

        # Validate comment length
        if comment is not None and len(comment) > 50000:
            return json.dumps({"error": "Comment must be 50,000 characters or less"})

        # Build GraphQL mutation
        mutation = """
mutation($vulnerabilityId: VulnerabilityID!, $comment: String) {
    vulnerabilityConfirm(input: { id: $vulnerabilityId, comment: $comment }) {
    vulnerability {
        id
        state
        title
        severity
        reportType
    }
    errors
    }
}
"""

        try:
            # Ensure vulnerability_id has proper GraphQL format
            if not vulnerability_id.startswith("gid://gitlab/Vulnerability/"):
                vulnerability_id = f"gid://gitlab/Vulnerability/{vulnerability_id}"

            variables = {
                "vulnerabilityId": vulnerability_id,
                "comment": comment,
            }

            response = await self.gitlab_client.apost(
                path="/api/graphql",
                body=json.dumps({"query": mutation, "variables": variables}),
            )

            try:
                response = self._process_http_response(
                    identifier="mutation", response=response
                )
            except ValueError as e:
                return json.dumps({"error": str(e)})

            mutation_result = response["data"]["vulnerabilityConfirm"]

            if mutation_result["errors"]:
                return json.dumps(
                    {"error": f"GraphQL errors: {mutation_result['errors']}"}
                )

            return json.dumps(
                {
                    "vulnerability": mutation_result["vulnerability"],
                    "success": True,
                    "message": "Vulnerability confirmed successfully",
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: ConfirmVulnerabilityInput, _tool_response: Any = None
    ) -> str | None:
        return f"Confirm vulnerability {args.vulnerability_id}"


class RevertToDetectedVulnerabilityInput(BaseModel):
    vulnerability_id: str = Field(
        description="The ID of the vulnerability to revert to detected state (e.g., 'gid://gitlab/Vulnerability/123')",
    )
    comment: Optional[str] = Field(
        default=None,
        description="Optional comment explaining why the vulnerability was reverted to detected state.",
    )


class RevertToDetectedVulnerability(DuoBaseTool):
    name: str = "revert_to_detected_vulnerability"
    description: str = """Revert a vulnerability's state back to 'detected' status in GitLab using GraphQL.

    The vulnerability is identified by its GitLab internal ID, which can be obtained from the
    list_vulnerabilities tool. An optional comment can be provided to explain the reason for reverting.

    For example:
    - Revert a vulnerability without comment:
        revert_to_detected_vulnerability(vulnerability_id="gid://gitlab/Vulnerability/123")
    - Revert with explanation:
        revert_to_detected_vulnerability(vulnerability_id="gid://gitlab/Vulnerability/123", comment="Reverting for re-assessment after code changes")
    """
    args_schema: Type[BaseModel] = RevertToDetectedVulnerabilityInput

    async def _execute(self, **kwargs: Any) -> str:
        vulnerability_id = kwargs.pop("vulnerability_id")
        comment = kwargs.pop("comment", None)

        # editorconfig-checker-disable
        # Build GraphQL mutation

        mutation = """
        mutation($vulnerabilityId: VulnerabilityID!, $comment: String) {
          vulnerabilityRevertToDetected(input: {
            id: $vulnerabilityId
            comment: $comment
          }) {
            errors
            vulnerability {
              id
              title
              state
              severity
            }
          }
        }
        """
        # editorconfig-checker-enable

        variables = {
            "vulnerabilityId": vulnerability_id,
        }

        if comment:
            variables["comment"] = comment

        try:
            response = await self.gitlab_client.apost(
                path="/api/graphql",
                body=json.dumps({"query": mutation, "variables": variables}),
            )

            try:
                response = self._process_http_response(
                    identifier="mutation", response=response
                )
            except ValueError as e:
                return json.dumps({"error": str(e)})

            mutation_result = response["data"]["vulnerabilityRevertToDetected"]

            if mutation_result["errors"]:
                return json.dumps({"error": mutation_result["errors"]})

            return json.dumps(
                {
                    "vulnerability": mutation_result["vulnerability"],
                    "status": "reverted_to_detected",
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: RevertToDetectedVulnerabilityInput, _tool_response: Any = None
    ) -> str:
        base_message = f"Revert vulnerability {args.vulnerability_id} to detected state"

        if args.comment:
            display_comment = (
                args.comment if len(args.comment) <= 100 else f"{args.comment[:97]}..."
            )
            return f"{base_message} - Reason: {display_comment}"

        return base_message
