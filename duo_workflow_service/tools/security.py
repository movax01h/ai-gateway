import json
from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

PROJECT_IDENTIFICATION_DESCRIPTION = """The project must be specified using its full path (e.g., 'namespace/project' or 'group/subgroup/project')."""


class ListVulnerabilitiesInput(BaseModel):
    project_full_path: str = Field(
        description="The full path of the GitLab project (e.g., 'namespace/project' or 'group/subgroup/project')",
    )
    severity: Optional[str] = Field(
        default=None,
        description="Filter vulnerabilities by severity (CRITICAL, HIGH, MEDIUM, LOW, INFO, UNKNOWN). If not specified, all severities will be returned.",
    )
    per_page: Optional[int] = Field(
        default=100,
        description="Number of results per page (default: 100, max: 100).",
    )
    page: Optional[int] = Field(
        default=1,
        description="Page number to fetch (default: 1).",
    )
    fetch_all_pages: Optional[bool] = Field(
        default=True,
        description="Whether to fetch all pages of results (default: True).",
    )


class ListVulnerabilities(DuoBaseTool):
    name: str = "list_vulnerabilities"
    description: str = f"""List security vulnerabilities in a GitLab project using GraphQL.

    {PROJECT_IDENTIFICATION_DESCRIPTION}

    The tool supports filtering vulnerabilities by severity level (CRITICAL, HIGH, MEDIUM, LOW, INFO, UNKNOWN).
    If no severity is specified, vulnerabilities of all severity levels will be returned.

    For example:
    - List all vulnerabilities in a project:
        list_vulnerabilities(project_full_path="namespace/project")
    - List only critical vulnerabilities:
        list_vulnerabilities(project_full_path="namespace/project", severity="CRITICAL")
    """
    args_schema: Type[BaseModel] = ListVulnerabilitiesInput  # type: ignore

    async def _arun(self, **kwargs: Any) -> str:
        project_full_path = kwargs.pop("project_full_path")
        fetch_all_pages = kwargs.pop("fetch_all_pages", True)
        per_page = kwargs.pop("per_page", 100)
        severity = kwargs.pop("severity", None)

        # editorconfig-checker-disable
        # Build GraphQL query
        query = """
        query($projectFullPath: ID!, $first: Int, $after: String, $severity: [VulnerabilitySeverity!]) {
          project(fullPath: $projectFullPath) {
            vulnerabilities(first: $first, after: $after, severity: $severity) {
              pageInfo {
                hasNextPage
                endCursor
              }
              nodes {
                id
                title
                reportType
                severity
                location{
                  ... on VulnerabilityLocationSast {
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

        all_vulnerabilities = []
        cursor = None

        try:
            while True:
                variables = {
                    "projectFullPath": project_full_path,
                    "first": per_page,
                    "after": cursor,
                    "severity": severity,
                }

                response = await self.gitlab_client.apost(
                    path="/api/graphql",
                    body=json.dumps({"query": query, "variables": variables}),
                )

                vulnerabilities = response["data"]["project"]["vulnerabilities"][
                    "nodes"
                ]
                all_vulnerabilities.extend(vulnerabilities)

                page_info = response["data"]["project"]["vulnerabilities"]["pageInfo"]

                if not fetch_all_pages or not page_info["hasNextPage"]:
                    break

                cursor = page_info["endCursor"]

            return json.dumps(
                {
                    "vulnerabilities": all_vulnerabilities,
                    "pagination": {"total_items": len(all_vulnerabilities)},
                }
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: ListVulnerabilitiesInput) -> str:
        return f"List vulnerabilities in project {args.project_full_path}"


class DismissVulnerabilityInput(BaseModel):
    vulnerability_id: str = Field(description="ID of the vulnerability to be dismissed")
    comment: str = Field(
        description="Comment why vulnerability was dismissed (maximum 50,000 characters)."
    )
    dismissal_reason: str = Field(
        description="Reason why vulnerability should be dismissed (ACCEPTABLE_RISK, FALSE_POSITIVE, MITIGATING_CONTROL, USED_IN_TESTS, NOT_APPLICABLE)"
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
                    "error": f"Invalid dismissal reason '{dismissal_reason}'. Must be one of: {', '.join(valid_dismissal_reasons)}"
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

    def format_display_message(self, args: DismissVulnerabilityInput) -> str:
        return f"Dismiss vulnerability {args.vulnerability_id}"
