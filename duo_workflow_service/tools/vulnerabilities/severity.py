import json
from typing import Any, ClassVar, List, Type

from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

PROJECT_IDENTIFICATION_DESCRIPTION = """The project must be specified using its full path (e.g., 'namespace/project' or 'group/subgroup/project')."""

__all__ = [
    "UpdateVulnerabilitySeverity",
]


class UpdateVulnerabilitySeverityInput(BaseModel):
    vulnerability_ids: List[str] = Field(
        description="List of vulnerability IDs to update severity for. These should be the vulnerability IDs as returned by list_vulnerabilities.",
    )
    severity: str = Field(
        description="New severity level for the vulnerabilities. Must be one of: CRITICAL, HIGH, MEDIUM, LOW, INFO, UNKNOWN.",
    )
    comment: str = Field(
        description="Comment explaining why the vulnerability severity was changed (required, maximum 50,000 characters).",
    )


class UpdateVulnerabilitySeverity(DuoBaseTool):
    VALID_SEVERITIES: ClassVar[set[str]] = {
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
        "INFO",
        "UNKNOWN",
    }
    MAX_COMMENT_LENGTH: ClassVar[int] = 50000

    name: str = "update_vulnerability_severity"
    unit_primitive: GitLabUnitPrimitive = (
        GitLabUnitPrimitive.UPDATE_VULNERABILITY_SEVERITY
    )
    description: str = f"""Update the severity level of vulnerabilities in a GitLab project using GraphQL.

    {PROJECT_IDENTIFICATION_DESCRIPTION}

    This tool allows you to override the severity level of one or more vulnerabilities and provide a comment explaining the change.
    The severity must be one of: CRITICAL, HIGH, MEDIUM, LOW, INFO, UNKNOWN.

    A comment explaining the reason for the severity change is required and can be up to 50,000 characters.

    For example:
    - Update single vulnerability severity:
        update_vulnerability_severity(
            vulnerability_ids=["gid://gitlab/Vulnerability/123"],
            severity="HIGH",
            comment="Updated severity based on security review"
        )
    - Update multiple vulnerabilities:
        update_vulnerability_severity(
            vulnerability_ids=["gid://gitlab/Vulnerability/123", "gid://gitlab/Vulnerability/456"],
            severity="LOW",
            comment="These are false positives based on code analysis"
        )
    """
    args_schema: Type[BaseModel] = UpdateVulnerabilitySeverityInput

    def ensure_list(self, value):
        if isinstance(value, list):
            return value
        return [value]

    def validate_inputs(self, vulnerability_ids, severity, comment):
        if not isinstance(severity, str):
            raise ValueError(
                f"Severity must be a string, got {type(severity).__name__}"
            )
        if severity not in self.VALID_SEVERITIES:
            raise ValueError(
                f"Invalid severity '{severity}'. Must be one of: {', '.join(self.VALID_SEVERITIES)}"
            )

        if not isinstance(comment, str):
            raise ValueError(f"Comment must be a string, got {type(comment).__name__}")
        if len(comment) > self.MAX_COMMENT_LENGTH:
            raise ValueError(
                f"Comment must be {self.MAX_COMMENT_LENGTH:,} characters or less"
            )

        if not vulnerability_ids:
            raise ValueError("At least one vulnerability ID must be provided")
        # Filter out falsy values and check if any valid IDs remain
        valid_ids = [id for id in vulnerability_ids if id and str(id).strip()]
        if not valid_ids:
            raise ValueError("At least one valid vulnerability ID must be provided")

    async def _execute(self, **kwargs: Any) -> str:
        vulnerability_ids = self.ensure_list(kwargs.pop("vulnerability_ids"))
        severity = kwargs.pop("severity")
        comment = kwargs.pop("comment")

        try:
            self.validate_inputs(vulnerability_ids, severity, comment)
        except ValueError as e:
            return json.dumps({"error": str(e)})

        # editorconfig-checker-disable
        mutation = """
        mutation vulnerabilitiesSeverityOverride($vulnerabilityIds: [VulnerabilityID!]!, $severity: VulnerabilitySeverity!, $comment: String!) {
          vulnerabilitiesSeverityOverride(
            input: {
              vulnerabilityIds: $vulnerabilityIds,
              severity: $severity,
              comment: $comment
            }
          ) {
            errors
            vulnerabilities {
              id
              severity
            }
          }
        }
        """
        # editorconfig-checker-enable

        try:
            variables = {
                "vulnerabilityIds": vulnerability_ids,
                "severity": severity,
                "comment": comment,
            }

            response = await self.gitlab_client.apost(
                path="/api/graphql",
                body=json.dumps({"query": mutation, "variables": variables}),
            )

            response = self._process_http_response(
                identifier="vulnerabilitiesSeverityOverride", response=response
            )

            if "errors" in response:
                return json.dumps({"error": f"GraphQL errors: {response['errors']}"})

            mutation_result = response["data"]["vulnerabilitiesSeverityOverride"]

            if mutation_result["errors"]:
                return json.dumps(
                    {"error": f"Mutation errors: {mutation_result['errors']}"}
                )

            return json.dumps(
                {
                    "success": True,
                    "updated_vulnerabilities": mutation_result["vulnerabilities"],
                    "message": f"Successfully updated severity to {severity} for {len(mutation_result['vulnerabilities'])} vulnerability(s)",
                }
            )

        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: Any, _tool_response: Any = None
    ) -> str | None:
        if not isinstance(args, UpdateVulnerabilitySeverityInput):
            return None
        return f"Update severity to {args.severity} for {len(args.vulnerability_ids)} vulnerability(s)"
