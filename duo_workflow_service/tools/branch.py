import json
from typing import Any, Type

from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.gitlab_resource_input import ProjectResourceInput


class CreateBranchInput(ProjectResourceInput):
    """Input model for creating a branch in a GitLab repository."""

    branch: str = Field(description="Name of the new branch to create")
    ref: str = Field(
        description="The branch name or commit SHA to create the branch from"
    )


class CreateBranch(DuoBaseTool):
    """Tool to create a new branch in a GitLab repository."""

    name: str = "create_branch"
    unit_primitive: GitLabUnitPrimitive = GitLabUnitPrimitive.DUO_AGENT_PLATFORM

    description: str = """Create a new branch in a GitLab repository.

    To identify the project you must provide either:
    - project_id parameter, or
    - A GitLab URL like:
        - https://gitlab.com/namespace/project
        - https://gitlab.com/group/subgroup/project

    For example:
    - Given project_id 13, branch name "feature-branch", and ref "main", the tool call would be:
        create_branch(project_id=13, branch="feature-branch", ref="main")
    - Given the URL https://gitlab.com/namespace/project, the tool call would be:
        create_branch(url="https://gitlab.com/namespace/project", branch="feature-branch", ref="main")
    """

    args_schema: Type[BaseModel] = CreateBranchInput

    async def _execute(self, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        project_id = kwargs.pop("project_id", None)
        branch = kwargs.get("branch")
        ref = kwargs.get("ref")

        project_id, errors = self._validate_project_url(url, project_id)

        if errors:
            return json.dumps({"error": "; ".join(errors)})

        params = {
            "branch": branch,
            "ref": ref,
        }

        try:
            response = await self.gitlab_client.apost(
                path=f"/api/v4/projects/{project_id}/repository/branches",
                body=json.dumps(params),
            )

            self._process_http_response(
                identifier=f"/api/v4/projects/{project_id}/repository/branches",
                response=response,
            )

            return json.dumps({"status": "success", "branch": response.body})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: CreateBranchInput, _tool_response: Any = None
    ) -> str:
        """Format a user-friendly message describing the action being performed."""
        if args.url:
            return f"Create branch {args.branch} from {args.ref} in {args.url}"
        return (
            f"Create branch {args.branch} from {args.ref} in project {args.project_id}"
        )
