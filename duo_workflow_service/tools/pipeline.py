import json
from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.gitlab_resource_input import ProjectResourceInput
from duo_workflow_service.tools.merge_request import (
    MERGE_REQUEST_IDENTIFICATION_DESCRIPTION,
)


class PipelineException(Exception):
    pass


class PipelinesNotFoundError(PipelineException):
    pass


class PipelineMergeRequestNotFoundError(PipelineException):
    pass


class GetPipelineErrorsInput(ProjectResourceInput):
    merge_request_iid: Optional[int] = Field(
        default=None,
        description="The IID of the merge request. Required if URL is not provided.",
    )


class GetPipelineErrorsForMergeRequest(DuoBaseTool):
    name: str = "get_pipeline_errors"
    description: str = f"""Get the logs for failed jobs in the latest pipeline in a merge request.
    This tool can be used when you have a project_id and merge_request_iid.
    Be careful to differentiate between a pipeline_id and a job_id when using this tool

    {MERGE_REQUEST_IDENTIFICATION_DESCRIPTION}

    For example:
    - Given project_id 13 and merge_request_iid 9, the tool call would be:
        get_pipeline_errors(project_id=13, merge_request_iid=9)
    - Given the URL https://gitlab.com/namespace/project/-/merge_requests/103, the tool call would be:
        get_pipeline_errors(url="https://gitlab.com/namespace/project/-/merge_requests/103")
    """
    args_schema: Type[BaseModel] = GetPipelineErrorsInput  # type: ignore

    async def _arun(self, **kwargs: Any) -> str:
        url = kwargs.get("url", None)
        project_id = kwargs.get("project_id", None)
        merge_request_iid = kwargs.get("merge_request_iid", None)

        validation_result = self._validate_merge_request_url(
            url, project_id, merge_request_iid
        )

        if validation_result.errors:
            return json.dumps({"error": "; ".join(validation_result.errors)})

        merge_request = await self.gitlab_client.aget(
            path=f"/api/v4/projects/{validation_result.project_id}/merge_requests/{validation_result.merge_request_iid}"
        )

        if isinstance(merge_request, dict) and merge_request.get("status") == 404:
            raise PipelineMergeRequestNotFoundError("Merge request not found")

        pipelines = await self.gitlab_client.aget(
            path=f"/api/v4/projects/{validation_result.project_id}/merge_requests/"
            f"{validation_result.merge_request_iid}/pipelines"
        )

        if not isinstance(pipelines, list) or len(pipelines) == 0:
            raise PipelinesNotFoundError("No pipelines found")

        last_pipeline = pipelines[0]
        last_pipeline_id = last_pipeline["id"]

        jobs = await self.gitlab_client.aget(
            path=f"/api/v4/projects/{validation_result.project_id}/pipelines/{last_pipeline_id}/jobs"
        )

        traces = "Failed Jobs:\n"
        for job in jobs:
            if job["status"] == "failed":
                job_id = job["id"]
                job_name = job["name"]
                traces += f"Name: {job_name}\nJob ID: {job_id}\n"
                try:
                    trace = await self.gitlab_client.aget(
                        path=f"/api/v4/projects/{validation_result.project_id}/jobs/{job_id}/trace",
                        parse_json=False,
                    )
                    traces += f"Trace: {trace}\n"
                except Exception as e:
                    traces += f"Error fetching trace: {str(e)}\n"

        return json.dumps({"merge_request": merge_request, "traces": traces})

    def format_display_message(self, args: GetPipelineErrorsInput) -> str:
        if args.url:
            return f"Get pipeline error logs for {args.url}"
        return f"Get pipeline error logs for merge request !{args.merge_request_iid} in project {args.project_id}"
