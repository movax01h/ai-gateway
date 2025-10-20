import json
from typing import Any, Optional, Type

import structlog
from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import (
    DuoBaseTool,
    MergeRequestValidationResult,
    PipelineValidationResult,
)
from duo_workflow_service.tools.gitlab_resource_input import ProjectResourceInput
from duo_workflow_service.tools.merge_request import (
    MERGE_REQUEST_IDENTIFICATION_DESCRIPTION,
)

log = structlog.stdlib.get_logger("workflow")


class GetPipelineErrorsInput(ProjectResourceInput):
    merge_request_iid: Optional[int] = Field(
        default=None,
        description="The IID of the merge request. Required if URL is not provided.",
    )


class GetPipelineErrors(DuoBaseTool):
    name: str = "get_pipeline_errors"
    description: str = f"""Get the logs for failed jobs in a pipeline.
    You can use this tool by passing in a merge request to get the failing jobs in the
    latest pipeline. You can also use this tool by identifying a pipeline directly.
    This tool can be used when you have a project_id and merge_request_iid.
    This tool can be used when you have a merge request URL.
    This tool can be used when you have a pipeline URL.
    Be careful to differentiate between a pipeline_id and a job_id when using this tool

    {MERGE_REQUEST_IDENTIFICATION_DESCRIPTION}

    To identify a pipeline you must provide:
    - A GitLab URL like:
        - https://gitlab.com/namespace/project/-/pipelines/33
        - https://gitlab.com/group/subgroup/project/-/pipelines/42

    For example:
    - Given project_id 13 and merge_request_iid 9, the tool call would be:
        get_pipeline_errors(project_id=13, merge_request_iid=9)
    - Given a merge request URL https://gitlab.com/namespace/project/-/merge_requests/103, the tool call would be:
        get_pipeline_errors(url="https://gitlab.com/namespace/project/-/merge_requests/103")
    - Given a pipeline URL https://gitlab.com/namespace/project/-/pipelines/33, the tool call would be:
        get_pipeline_errors(url="https://gitlab.com/namespace/project/-/pipelines/33")
    """
    args_schema: Type[BaseModel] = GetPipelineErrorsInput  # type: ignore

    unit_primitive: GitLabUnitPrimitive = GitLabUnitPrimitive.ASK_MERGE_REQUEST

    async def _execute(  # pylint: disable=too-many-return-statements,too-many-branches
        self, **kwargs: Any
    ) -> str:
        url = kwargs.get("url", None)
        project_id = kwargs.get("project_id", None)
        merge_request_iid = kwargs.get("merge_request_iid", None)

        pipeline_id = None
        merge_request = None
        validation_result: Optional[
            MergeRequestValidationResult | PipelineValidationResult
        ] = None
        try:
            if url and "/-/pipelines/" in url:
                validation_result = self._validate_pipeline_url(url)

                if validation_result.errors:
                    return json.dumps({"error": "; ".join(validation_result.errors)})

                pipeline_id = validation_result.pipeline_iid
            else:
                validation_result = self._validate_merge_request_url(
                    url, project_id, merge_request_iid
                )

                if validation_result.errors:
                    return json.dumps({"error": "; ".join(validation_result.errors)})

                merge_request_response = await self.gitlab_client.aget(
                    path=f"/api/v4/projects/{validation_result.project_id}/merge_requests/"
                    f"{validation_result.merge_request_iid}",
                    use_http_response=True,
                )

                if merge_request_response.status_code == 404:
                    return json.dumps(
                        {
                            "error": f"Merge request with iid {validation_result.merge_request_iid} not found"
                        }
                    )

                if not merge_request_response.is_success():
                    log.error(
                        "Failed to fetch merge request: status_code=%s, response=%s",
                        merge_request_response.status_code,
                        merge_request_response.body,
                    )

                merge_request = merge_request_response.body

                pipelines_response = await self.gitlab_client.aget(
                    path=f"/api/v4/projects/{validation_result.project_id}/merge_requests/"
                    f"{validation_result.merge_request_iid}/pipelines",
                    use_http_response=True,
                )

                if not pipelines_response.is_success():
                    log.error(
                        "Failed to fetch pipelines: status_code=%s, response=%s",
                        pipelines_response.status_code,
                        pipelines_response.body,
                    )

                pipelines = pipelines_response.body

                if not isinstance(pipelines, list) or len(pipelines) == 0:
                    return json.dumps(
                        {
                            "error": f"No pipelines found for merge request iid {validation_result.merge_request_iid}"
                        }
                    )

                last_pipeline = pipelines[0]
                pipeline_id = last_pipeline["id"]

            jobs_response = await self.gitlab_client.aget(
                path=f"/api/v4/projects/{validation_result.project_id}/pipelines/{pipeline_id}/jobs",
                use_http_response=True,
            )

            if not jobs_response.is_success():
                log.error(
                    "Failed to fetch jobs: status_code=%s, response=%s",
                    jobs_response.status_code,
                    jobs_response.body,
                )

            jobs = jobs_response.body
            if not isinstance(jobs, list):
                return json.dumps(
                    {
                        "error": f"Failed to fetch jobs for pipeline {pipeline_id}: {jobs}"
                    }
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

            if merge_request:
                return json.dumps({"merge_request": merge_request, "traces": traces})

            return json.dumps({"pipeline_id": pipeline_id, "traces": traces})
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: GetPipelineErrorsInput, _tool_response: Any = None
    ) -> str:
        if args.url:
            return f"Get pipeline error logs for {args.url}"
        return f"Get pipeline error logs for merge request !{args.merge_request_iid} in project {args.project_id}"
