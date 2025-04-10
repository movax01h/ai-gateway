import json
from typing import Type

from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


class PipelineException(Exception):
    pass


class PipelinesNotFoundError(PipelineException):
    pass


class PipelineMergeRequestNotFoundError(PipelineException):
    pass


class GetPipelineErrorsInput(BaseModel):
    project_id: int = Field(description="Id of the project")
    merge_request_id: int = Field(description="Id of the MR")


class GetPipelineErrorsForMergeRequest(DuoBaseTool):
    name: str = "get_pipeline_errors"
    description: str = """Get the pipeline trace for a pipeline with errors. This tool can be used when you have a project_id and merge_request_id.
                    Be careful to differentiate between a pipeline_id and a job_id when using this tool"""
    args_schema: Type[BaseModel] = GetPipelineErrorsInput  # type: ignore

    async def _arun(self, project_id: int, merge_request_id: int) -> str:
        merge_request = await self.gitlab_client.aget(
            path=f"/api/v4/projects/{project_id}/merge_requests/{merge_request_id}"
        )

        if isinstance(merge_request, dict) and merge_request.get("status") == 404:
            raise PipelineMergeRequestNotFoundError("Merge request not found")

        pipelines = await self.gitlab_client.aget(
            path=f"/api/v4/projects/{project_id}/merge_requests/{merge_request_id}/pipelines"
        )

        if not isinstance(pipelines, list) or len(pipelines) == 0:
            raise PipelinesNotFoundError("No pipelines found")

        last_pipeline = pipelines[0]
        last_pipeline_id = last_pipeline["id"]

        jobs = await self.gitlab_client.aget(
            path=f"/api/v4/projects/{project_id}/pipelines/{last_pipeline_id}/jobs"
        )

        traces = "Failed Jobs:\n"
        for job in jobs:
            if job["status"] == "failed":
                job_id = job["id"]
                job_name = job["name"]
                traces += f"Name: {job_name}\nJob ID: {job_id}\n"
                try:
                    trace = await self.gitlab_client.aget(
                        path=f"/api/v4/projects/{project_id}/jobs/{job_id}/trace",
                        parse_json=False,
                    )
                    traces += f"Trace: {trace}\n"
                except Exception as e:
                    traces += f"Error fetching trace: {str(e)}\n"

        return json.dumps({"merge_request": merge_request, "traces": traces})

    def format_display_message(self, args: GetPipelineErrorsInput) -> str:
        return f"Get pipeline error logs for merge request !{args.merge_request_id} in project {args.project_id}"
