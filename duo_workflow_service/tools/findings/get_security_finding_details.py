import json
from typing import Any, Type, Union

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.findings.queries.security_findings import (
    GET_SECURITY_FINDING_DETAILS_QUERY,
)


class GetSecurityFindingDetailsInput(BaseModel):
    """Input model for the GetSecurityFindingDetails tool."""

    project_full_path: str = Field(
        description="The full path of the project (e.g., 'group/project')."
    )

    uuid: str = Field(
        description="The UUID of the security finding (e.g., 'abc-123-def-456')."
    )
    pipeline_id: Union[int, str] = Field(
        description="""
        The pipeline ID as an integer (e.g., 12345) or GraphQL GID
        (e.g., 'gid://gitlab/Ci::Pipeline/12345').
        """,
    )

    def get_pipeline_gid(self) -> str:
        """Convert pipeline_id to GID format if needed."""
        if isinstance(self.pipeline_id, int):
            return f"gid://gitlab/Ci::Pipeline/{self.pipeline_id}"
        if isinstance(self.pipeline_id, str) and self.pipeline_id.startswith("gid://"):
            return self.pipeline_id
        return f"gid://gitlab/Ci::Pipeline/{self.pipeline_id}"


class GetSecurityFindingDetails(DuoBaseTool):
    """Tool for fetching detailed information about a specific security finding from a pipeline scan."""

    name: str = "get_security_finding_details"
    description: str = """
    Use this tool to get details for a specific security finding identified by its UUID and pipeline ID.

    A "Security Finding" is a potential vulnerability discovered in a pipeline scan.
    It is an ephemeral object identified by a UUID.

    **Use this tool when you have both a UUID and pipeline ID.**

    This is different from a "Vulnerability", which is a persisted record on the default branch and has a numeric ID.
    **Do NOT use this tool for numeric vulnerability IDs; when you have a numeric vulnerability ID, use the 'get_vulnerability_details' tool.**

    For example:
        get_security_finding_details(
            uuid="1e9a2bf7-0450-5894-8db5-895c98e39deb",
            pipeline_id=12345,
            project_full_path="namespace/project"
        )
    """
    args_schema: Type[BaseModel] = GetSecurityFindingDetailsInput

    async def _execute(self, **kwargs: Any) -> str:
        project_path = kwargs.pop("project_full_path")
        uuid = kwargs.pop("uuid")
        pipeline_id_raw = kwargs.pop("pipeline_id")

        try:
            input_model = GetSecurityFindingDetailsInput(
                project_full_path=project_path, uuid=uuid, pipeline_id=pipeline_id_raw
            )
            pipeline_gid = input_model.get_pipeline_gid()
            return await self._fetch_finding_from_pipeline(
                project_path, pipeline_gid, uuid
            )

        except Exception as e:
            raise ToolException(
                f"An unexpected error occurred while fetching the security finding: {str(e)}"
            )

    async def _fetch_finding_from_pipeline(
        self, project_path: str, pipeline_gid: str, finding_uuid: str
    ) -> str:
        """Fetch a security finding by UUID from a specific pipeline using the pipeline's GID.

        Args:
            project_path: Full path to the project (e.g., 'namespace/project')
            pipeline_gid: Pipeline GraphQL ID (e.g., 'gid://gitlab/Ci::Pipeline/12345')
            finding_uuid: Security finding UUID (e.g., '1e9a2bf7-0450-5894-8db5-895c98e39deb')
        """
        variables = {
            "projectFullPath": project_path,
            "pipelineId": pipeline_gid,
            "findingUuid": finding_uuid,
        }

        response = await self.gitlab_client.apost(
            path="/api/graphql",
            body=json.dumps(
                {"query": GET_SECURITY_FINDING_DETAILS_QUERY, "variables": variables}
            ),
        )

        if "errors" in response:
            return json.dumps(
                {"error": "GraphQL query failed", "errors": response["errors"]}
            )

        project = response.get("data", {}).get("project")
        if not project:
            return json.dumps(
                {"error": "Project not found or access denied", "project": project_path}
            )

        pipeline = project.get("pipeline")
        if not pipeline:
            return json.dumps(
                {"error": "Pipeline not found", "pipeline_id": pipeline_gid}
            )

        finding = pipeline.get("securityReportFinding")
        if not finding:
            return json.dumps(
                {
                    "error": "Security finding not found in the specified pipeline",
                    "uuid": finding_uuid,
                    "pipeline_id": pipeline_gid,
                }
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

        if isinstance(args.pipeline_id, int):
            pipeline_numeric_id = str(args.pipeline_id)
        else:
            pipeline_numeric_id = args.pipeline_id.split("/")[-1]

        return f"Get details for security finding {args.uuid[:8]}... from pipeline {pipeline_numeric_id}"
