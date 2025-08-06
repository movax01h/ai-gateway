import json
from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


class CiLinterInput(BaseModel):
    project_id: int = Field(description="Id of the project")
    content: str = Field(
        description="The content of the CI/CD YAML configuration to validate."
    )
    ref: Optional[str] = Field(
        default=None,
        description="The branch, tag, or commit SHA to validate against. "
        "If not provided, validates against the default branch. "
        "Required when using include:local with files that exist only on specific branches.",
    )


class CiLinter(DuoBaseTool):
    name: str = "ci_linter"
    description: str = """Validates a CI/CD YAML configuration against GitLab CI syntax rules in the context of the
    project. This tool can be used when you have a project_id and the content of the CI/CD YAML configuration and will
    return a JSON response indicating whether the configuration is valid or not, along with any errors found.

    Note: When using 'include:' with files that don't exist on the default branch, you must provide the 'ref' parameter
    pointing to the branch where those files exist.
    """

    args_schema: Type[BaseModel] = CiLinterInput

    async def _arun(self, content: str, **kwargs: Any) -> str:
        url = kwargs.pop("url", None)
        ref = kwargs.pop("ref", "")
        project_id = kwargs.pop("project_id", None)

        project_id, errors = self._validate_project_url(url, project_id)

        if errors:
            return json.dumps({"error": "; ".join(errors)})

        try:
            body = {"content": content}
            if ref:
                body["ref"] = ref

            response = await self.gitlab_client.apost(
                path=f"/api/v4/projects/{project_id}/ci/lint",
                body=json.dumps(body),
            )
            return json.dumps(response)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: CiLinterInput, _tool_response: Any = None
    ) -> str:
        message = f"Validate CI/CD YAML configuration in context of project: {args.project_id}"
        if args.ref:
            message += f" (ref: {args.ref})"
        return message
