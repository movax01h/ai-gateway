import json
from typing import Any, ClassVar, Type

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.security_inputs import (
    MAX_GITLAB_GLOBAL_ID_LENGTH,
    MAX_VULNERABILITY_COMMENT_LENGTH,
    MAX_VULNERABILITY_IDS,
    VulnerabilityComment,
    VulnerabilityIdList,
    VulnerabilitySeverity,
)
from duo_workflow_service.tools.tier_access_checker import (
    LICENSED_FEATURE_SECURITY_DASHBOARD,
)

PROJECT_IDENTIFICATION_DESCRIPTION = """The project must be specified using its full path (e.g., 'namespace/project' or 'group/subgroup/project')."""

__all__ = [
    "UpdateVulnerabilitySeverity",
]


class UpdateVulnerabilitySeverityInput(BaseModel):
    vulnerability_ids: VulnerabilityIdList = Field(
        description="List of vulnerability IDs to update severity for. These should be the vulnerability IDs as "
        f"returned by list_vulnerabilities. Up to {MAX_VULNERABILITY_IDS} can be provided, "
        f"each at most {MAX_GITLAB_GLOBAL_ID_LENGTH} characters.",
    )
    severity: VulnerabilitySeverity = Field(
        description="New severity level for the vulnerabilities.",
    )
    comment: VulnerabilityComment = Field(
        description="Comment explaining why the vulnerability severity was changed "
        f"(required, maximum {MAX_VULNERABILITY_COMMENT_LENGTH:,} characters).",
    )


class UpdateVulnerabilitySeverity(DuoBaseTool):
    tier_check_licensed_feature: ClassVar[str] = LICENSED_FEATURE_SECURITY_DASHBOARD
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    name: str = "update_vulnerability_severity"
    description: str = f"""Update the severity level of vulnerabilities in a GitLab project using GraphQL.

    {PROJECT_IDENTIFICATION_DESCRIPTION}

    This tool allows you to override the severity level of one or more vulnerabilities and provide a comment explaining the change.
    The severity must be one of: CRITICAL, HIGH, MEDIUM, LOW, INFO, UNKNOWN.

    Up to {MAX_VULNERABILITY_IDS} vulnerabilities can be updated in one call.
    A comment explaining the reason for the severity change is required and can be up to {MAX_VULNERABILITY_COMMENT_LENGTH:,} characters.

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

    def validate_inputs(self, vulnerability_ids: list[str]) -> None:
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

        self.validate_inputs(vulnerability_ids)

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
            raise ToolException(f"GraphQL errors: {response['errors']}")

        mutation_result = response["data"]["vulnerabilitiesSeverityOverride"]

        if mutation_result["errors"]:
            raise ToolException(f"Mutation errors: {mutation_result['errors']}")

        return json.dumps(
            {
                "success": True,
                "updated_vulnerabilities": mutation_result["vulnerabilities"],
                "message": f"Successfully updated severity to {severity} for {len(mutation_result['vulnerabilities'])} vulnerability(s)",
            }
        )

    def format_display_message(
        self, args: Any, _tool_response: Any = None
    ) -> str | None:
        if not isinstance(args, UpdateVulnerabilitySeverityInput):
            return None
        return f"Update severity to {args.severity} for {len(args.vulnerability_ids)} vulnerability(s)"
