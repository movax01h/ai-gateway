import json
from collections import Counter
from enum import StrEnum
from typing import Any, Optional, Type

from pydantic import BaseModel, Field, field_validator

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

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


__all__ = [
    "ListVulnerabilities",
    "DismissVulnerability",
    "LinkVulnerabilityToIssue",
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
    """Tool for listing GitLab project vulnerabilities with filtering."""

    name: str = "list_vulnerabilities"
    description: str = f"""List security vulnerabilities in a GitLab project using GraphQL.

    {PROJECT_IDENTIFICATION_DESCRIPTION}

    The tool supports filtering vulnerabilities by:
    - Severity levels (can specify multiple: CRITICAL, HIGH, MEDIUM, LOW, INFO, UNKNOWN)
    - Report type (SAST, DAST, DEPENDENCY_SCANNING, etc.)

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

    async def _arun(self, **kwargs: Any) -> str:
        """Execute the vulnerability listing."""
        try:
            project_full_path = kwargs.pop("project_full_path")
            fetch_all_pages = kwargs.pop("fetch_all_pages", True)
            per_page = kwargs.pop("per_page", 100)
            severity = kwargs.pop("severity", None)
            report_type = kwargs.pop("report_type", None)

            # editorconfig-checker-disable
            # Build GraphQL query with enhanced location details
            query = """
            query($projectFullPath: ID!, $first: Int, $after: String, $severity: [VulnerabilitySeverity!], $reportType: [VulnerabilityReportType!]) {
                project(fullPath: $projectFullPath) {
                    vulnerabilities(first: $first, after: $after, severity: $severity, reportType: $reportType) {
                        pageInfo {
                            hasNextPage
                            endCursor
                        }
                        nodes {
                            id
                            title
                            reportType
                            severity
                            state
                            location{
                                ... on VulnerabilityLocationSast {
                                    file
                                    startLine
                                }
                                ... on VulnerabilityLocationDependencyScanning {
                                    file
                                    dependency {
                                        package {
                                            name
                                        }
                                        version
                                    }
                                }
                                ... on VulnerabilityLocationContainerScanning {
                                    image
                                    operatingSystem
                                    dependency {
                                        package {
                                            name
                                        }
                                        version
                                    }
                                }
                                ... on VulnerabilityLocationSecretDetection {
                                    file
                                    startLine
                                }
                            }
                        }
                    }
                }
            }
            """
            # editorconfig-checker-enable

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
                    body=json.dumps({"query": query, "variables": variables}),
                )

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

    async def _arun(self, **kwargs: Any) -> str:
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

    async def _arun(self, **kwargs: Any) -> str:
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
