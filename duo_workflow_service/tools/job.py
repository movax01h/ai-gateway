import json
from typing import Any, List, NamedTuple, Optional, Type

from pydantic import BaseModel, Field

from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.gitlab_resource_input import ProjectResourceInput


class JobURLValidationResult(NamedTuple):
    project_id: Optional[str]
    job_id: Optional[int]
    errors: List[str]


class GetLogsFromJobInput(ProjectResourceInput):
    job_id: Optional[int] = Field(
        default=None,
        description="The ID of the job. Required if URL is not provided.",
    )


class GetLogsFromJob(DuoBaseTool):
    name: str = "get_job_logs"
    description: str = """Get the trace for a job.

    Use this tool to get more details for specific jobs within a pipeline.

    To identify a job you must provide either:
    - project_id and job_id, or
    - A GitLab URL like:
      - https://gitlab.com/namespace/project/-/jobs/42
      - https://gitlab.com/group/subgroup/project/-/jobs/42

    For example:
    - Given project_id 13 and job_id 9, the tool call would be:
        get_job_logs(project_id=13, job_id=9)
    - Given the URL https://gitlab.com/namespace/project/-/jobs/103, the tool call would be:
        get_job_logs(url="https://gitlab.com/namespace/project/-/jobs/103")

    You can obtain the project_id and job_id from the pipeline details.
    """
    args_schema: Type[BaseModel] = GetLogsFromJobInput  # type: ignore

    def _validate_job_url(
        self,
        url: Optional[str],
        project_id: Optional[int | str],
        job_id: Optional[int],
    ) -> JobURLValidationResult:
        """Validate job URL and extract project_id and job_id.

        Args:
            url: The GitLab URL to parse
            project_id: The project ID provided by the user
            job_id: The job ID provided by the user

        Returns:
            JobURLValidationResult containing:
                - The validated project_id (or None if validation failed)
                - The validated job_id (or None if validation failed)
                - A list of error messages (empty if validation succeeded)
        """
        errors = []

        if not url:
            if not project_id:
                errors.append("'project_id' must be provided when 'url' is not")
            if not job_id:
                errors.append("'job_id' must be provided when 'url' is not")
            return JobURLValidationResult(
                None if project_id is None else str(project_id), job_id, errors
            )

        try:
            # Parse URL and validate netloc against gitlab_host
            url_project_id, url_job_id = GitLabUrlParser.parse_job_url(
                url, self.gitlab_host
            )

            # If both URL and IDs are provided, check if they match
            if project_id is not None and str(project_id) != url_project_id:
                errors.append(
                    f"Project ID mismatch: provided '{project_id}' but URL contains '{url_project_id}'"
                )
            if job_id is not None and job_id != url_job_id:
                errors.append(
                    f"Job ID mismatch: provided '{job_id}' but URL contains '{url_job_id}'"
                )

            # Use the IDs from the URL
            return JobURLValidationResult(url_project_id, url_job_id, errors)
        except GitLabUrlParseError as e:
            errors.append(f"Failed to parse URL: {str(e)}")
            return JobURLValidationResult(
                None if project_id is None else str(project_id), job_id, errors
            )

    async def _arun(self, **kwargs: Any) -> str:
        url = kwargs.get("url", None)
        project_id = kwargs.get("project_id", None)
        job_id = kwargs.get("job_id", None)

        validation_result = self._validate_job_url(url, project_id, job_id)

        if validation_result.errors:
            return json.dumps({"error": "; ".join(validation_result.errors)})

        try:
            trace = await self.gitlab_client.aget(
                path=f"/api/v4/projects/{validation_result.project_id}/jobs/{validation_result.job_id}/trace",
                parse_json=False,
            )

            if not trace:
                return "No job found"

            return json.dumps({"job_id": validation_result.job_id, "trace": trace})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(self, args: GetLogsFromJobInput) -> str:
        if args.url:
            return f"Get logs for {args.url}"
        return f"Get logs for job #{args.job_id} in project {args.project_id}"
