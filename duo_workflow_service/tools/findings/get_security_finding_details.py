import json
from typing import Any, ClassVar, Type

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.findings.queries import (
    GET_SECURITY_FINDING_DETAILS_QUERY,
)
from duo_workflow_service.tools.tier_access_checker import (
    LICENSED_FEATURE_SECURITY_DASHBOARD,
)


class GetSecurityFindingDetailsInput(BaseModel):
    """Input model for the GetSecurityFindingDetails tool."""

    project_full_path: str = Field(
        description="The full path of the project (e.g., 'group/project')."
    )
    uuid: str = Field(
        description="The UUID of the security finding (e.g., 'abc-123-def-456')."
    )
    ref: str = Field(
        description="The branch or tag ref the pipeline was run for. "
        "For an MR, use the MR source branch name (e.g., 'my-feature-branch')."
    )


class GetSecurityFindingDetails(DuoBaseTool):
    """Tool for fetching detailed information about a specific security finding from a pipeline scan."""

    tier_check_licensed_feature: ClassVar[str] = LICENSED_FEATURE_SECURITY_DASHBOARD
    name: str = "get_security_finding_details"
    description: str = """
    Use this tool to get details for a specific security finding identified by its UUID and branch ref.

    A "Security Finding" is a potential vulnerability discovered in a pipeline scan.
    It is an ephemeral object identified by a UUID.

    **Use this tool when you have a UUID and the ref (MR source branch name).**

    The `ref` is the source branch of the MR the pipeline ran on.

    This is different from a "Vulnerability", which is a persisted record on the default branch and has a numeric ID.
    **Do NOT use this tool for numeric vulnerability IDs; use the 'get_vulnerability_details' tool instead.**

    For example:
        get_security_finding_details(
            uuid="1e9a2bf7-0450-5894-8db5-895c98e39deb",
            ref="my-feature-branch",
            project_full_path="namespace/project"
        )
    """
    args_schema: Type[BaseModel] = GetSecurityFindingDetailsInput

    async def _execute(self, **kwargs: Any) -> str:
        project_path = kwargs.pop("project_full_path")
        uuid = kwargs.pop("uuid")
        ref = kwargs.pop("ref")

        try:
            return await self._fetch_finding_from_pipeline(project_path, ref, uuid)
        except ToolException:
            raise
        except Exception as e:
            raise ToolException(
                f"An unexpected error occurred while fetching the security finding: {e!s}"
            )

    async def _fetch_finding_from_pipeline(
        self, project_path: str, ref: str, finding_uuid: str
    ) -> str:
        """Fetch a security finding by UUID using project.pipelines(first: 1, ref:)."""
        variables = {
            "projectFullPath": project_path,
            "ref": ref,
            "findingUuid": finding_uuid,
        }

        response = await self.gitlab_client.apost(
            path="/api/graphql",
            body=json.dumps(
                {"query": GET_SECURITY_FINDING_DETAILS_QUERY, "variables": variables}
            ),
        )

        response = self._process_http_response(identifier="query", response=response)

        if "errors" in response:
            raise ToolException(
                f"GraphQL query failed: {json.dumps(response['errors'])}"
            )

        project = response.get("data", {}).get("project")
        if not project:
            raise ToolException(f"Project not found or access denied: {project_path}")

        pipeline_nodes = (project.get("pipelines") or {}).get("nodes") or []
        if not pipeline_nodes:
            raise ToolException(f"No pipeline found for ref '{ref}'")

        pipeline = pipeline_nodes[0]

        finding = pipeline.get("securityReportFinding")
        if not finding:
            raise ToolException(
                f"Security finding not found in the specified pipeline: "
                f"uuid={finding_uuid}, ref={ref}"
            )

        result = {
            "finding": finding,
            "pipeline_context": {
                "id": pipeline["id"],
                "sha": pipeline.get("sha"),
                "ref": pipeline.get("ref"),
                "status": pipeline.get("status"),
                "createdAt": pipeline.get("createdAt"),
            },
            "project_context": {
                "id": project["id"],
                "webUrl": project.get("webUrl"),
                "nameWithNamespace": project.get("nameWithNamespace"),
            },
            "metadata": {
                "is_promoted": finding.get("vulnerability") is not None,
                "is_dismissed": finding.get("dismissedAt") is not None,
                "is_false_positive": finding.get("falsePositive", False),
                "ai_resolution_available": finding.get("aiResolutionAvailable", False),
                "ai_resolution_enabled": finding.get("aiResolutionEnabled", False),
            },
        }

        return json.dumps(result)

    def format_display_message(
        self, args: GetSecurityFindingDetailsInput, _tool_response: Any = None
    ) -> str:
        """Formats a user-friendly message for the UI log."""
        return (
            f"Get details for security finding {args.uuid[:8]}... on ref '{args.ref}'"
        )
