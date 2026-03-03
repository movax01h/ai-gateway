import json
from typing import Any, Optional, Type

import structlog
from langchain_core.tools import ToolException
from lxml import etree
from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import (
    DuoBaseTool,
    MergeRequestValidationResult,
    PipelineValidationResult,
)
from duo_workflow_service.tools.gitlab_resource_input import (
    GitLabResourceInput,
    ProjectResourceInput,
)
from duo_workflow_service.tools.merge_request import (
    MERGE_REQUEST_IDENTIFICATION_DESCRIPTION,
)

log = structlog.stdlib.get_logger("workflow")

MAX_LOG_PAGES_FOR_PIPELINE = 20
MAX_JOBS_RETURNED = 20


class GetPipelineFailingJobsInput(ProjectResourceInput):
    merge_request_iid: Optional[int] = Field(
        default=None,
        description="The IID of the merge request. Required if URL is not provided.",
    )


class GetPipelineFailingJobs(DuoBaseTool):
    name: str = "get_pipeline_failing_jobs"
    description: str = f"""Get the IDs for failed jobs in a pipeline.
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
        get_pipeline_failing_jobs(project_id=13, merge_request_iid=9)
    - Given a merge request URL https://gitlab.com/namespace/project/-/merge_requests/103, the tool call would be:
        get_pipeline_failing_jobs(url="https://gitlab.com/namespace/project/-/merge_requests/103")
    - Given a pipeline URL https://gitlab.com/namespace/project/-/pipelines/33, the tool call would be:
        get_pipeline_failing_jobs(url="https://gitlab.com/namespace/project/-/pipelines/33")
    """
    args_schema: Type[BaseModel] = GetPipelineFailingJobsInput

    async def _execute(  # pylint: disable=too-many-return-statements
        self, **kwargs: Any
    ) -> str:
        url = kwargs.get("url", None)
        project_id = kwargs.get("project_id")
        merge_request_iid = kwargs.get("merge_request_iid", None)

        pipeline_id = None
        merge_request = None
        validation_result: Optional[
            MergeRequestValidationResult | PipelineValidationResult
        ] = None
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
            )

            if merge_request_response.status_code == 404:
                return json.dumps(
                    {
                        "error": f"Merge request with iid {validation_result.merge_request_iid} not found"
                    }
                )

            if not merge_request_response.is_success():
                error_str = (
                    f"Failed to fetch merge request: status_code={merge_request_response.status_code}, "
                    f"response={merge_request_response.body}"
                )
                log.error(error_str)
                return json.dumps({"error": error_str})

            merge_request = merge_request_response.body

            pipelines_response = await self.gitlab_client.aget(
                path=f"/api/v4/projects/{validation_result.project_id}/merge_requests/"
                f"{validation_result.merge_request_iid}/pipelines",
            )

            if not pipelines_response.is_success():
                error_str = (
                    f"Failed to fetch pipelines: status_code={pipelines_response.status_code}, "
                    f"response={pipelines_response.body}"
                )
                log.error(error_str)
                return json.dumps({"error": error_str})

            pipelines = pipelines_response.body

            if not isinstance(pipelines, list) or len(pipelines) == 0:
                return json.dumps(
                    {
                        "error": f"No pipelines found for merge request iid {validation_result.merge_request_iid}"
                    }
                )

            last_pipeline = pipelines[0]
            pipeline_id = last_pipeline["id"]

        failing_jobs = await self._get_failing_jobs(
            validation_result.project_id, pipeline_id
        )
        if len(failing_jobs) > MAX_JOBS_RETURNED:
            failing_jobs = failing_jobs[:MAX_JOBS_RETURNED]

        if len(failing_jobs) == 0:
            return json.dumps({"error": "No Failing Jobs Found."})

        xml_root = etree.Element("jobs")
        for job in failing_jobs:
            xml_job = etree.SubElement(xml_root, "job")
            job_id = job["id"]
            job_name = job["name"]

            job_name_elem = etree.SubElement(xml_job, "job_name")
            job_name_elem.text = job_name

            job_id_elem = etree.SubElement(xml_job, "job_id")
            job_id_elem.text = str(job_id)

        failed_jobs_str = "Failed Jobs:\n" + etree.tostring(
            xml_root, pretty_print=True, encoding="unicode"
        )

        if merge_request:
            return json.dumps(
                {"merge_request": merge_request, "failed_jobs": failed_jobs_str}
            )

        return json.dumps({"pipeline_id": pipeline_id, "failed_jobs": failed_jobs_str})

    async def _get_failing_jobs(
        self, project_id: str | None, pipeline_id: int | None
    ) -> list[dict]:
        next_page = "1"
        page_count = 0
        failing_jobs: list[dict] = []
        while next_page and page_count < MAX_LOG_PAGES_FOR_PIPELINE:
            page_count += 1
            jobs_response = await self.gitlab_client.aget(
                path=f"/api/v4/projects/{project_id}/pipelines/{pipeline_id}"
                f"/jobs?per_page=100&page={next_page}",
            )

            if not jobs_response.is_success():
                error_str = (
                    f"Failed to fetch jobs: status_code={jobs_response.status_code}, "
                    f"response={jobs_response.body}"
                )
                raise ToolException(error_str)

            jobs = jobs_response.body
            if not isinstance(jobs, list):
                raise ToolException(
                    f"Failed to fetch jobs for pipeline {pipeline_id}: {jobs}"
                )
            for job in jobs:
                if job["status"] == "failed":
                    failing_jobs.append(job)

            next_page = jobs_response.headers.get("X-Next-Page", "")

        return failing_jobs

    def format_display_message(
        self, args: GetPipelineFailingJobsInput, _tool_response: Any = None
    ) -> str:
        if args.url:
            return f"Get pipeline failing jobs for {args.url}"
        return f"Get pipeline failing jobs for merge request !{args.merge_request_iid} in project {args.project_id}"


class GetDownstreamPipelines(DuoBaseTool):
    name: str = "get_downstream_pipelines"
    description: str = """Get the URLs for downstream pipelines.
    This tool can be used when you have a pipeline URL.

    Be careful to differentiate between a pipeline_id and a job_id when using this tool.

    To identify a pipeline you must provide:
    - A GitLab URL like:
        - https://gitlab.com/namespace/project/-/pipelines/33
        - https://gitlab.com/group/subgroup/project/-/pipelines/42

    For example:
    - Given a pipeline URL https://gitlab.com/namespace/project/-/pipelines/33, the tool call would be:
        get_downstream_pipelines(url="https://gitlab.com/namespace/project/-/pipelines/33")
    """
    args_schema: Type[BaseModel] = GitLabResourceInput

    async def _execute(self, url: str) -> str:
        validation_result = self._validate_pipeline_url(url)

        if validation_result.errors:
            return json.dumps({"error": "; ".join(validation_result.errors)})

        project_id = validation_result.project_id
        pipeline_iid = validation_result.pipeline_iid
        response = await self.gitlab_client.aget(
            path=f"/api/v4/projects/{project_id}/pipelines/{pipeline_iid}/bridges",
        )

        if not response.is_success():
            error_str = (
                f"Failed to fetch downstream pipelines: status_code={response.status_code}, "
                f"response={response.body}"
            )
            raise ToolException(error_str)

        downstream_pipelines = response.body
        if not isinstance(downstream_pipelines, list):
            raise ToolException(
                f"Failed to fetch downstream pipelines for url: {url}: {downstream_pipelines}"
            )

        downstream_pipeline_urls: list[dict] = []
        for pipeline in downstream_pipelines:
            downstream_pipeline = pipeline.get("downstream_pipeline", None)

            if downstream_pipeline:
                # Only handle parent/child pipelines for now, and do not return
                # downstream pipelines across different projects:
                # https://docs.gitlab.com/ci/pipelines/downstream_pipelines/#multi-project-pipelines
                downstream_url = downstream_pipeline["web_url"]
                downstream_validation_result = self._validate_pipeline_url(
                    downstream_url
                )
                if downstream_validation_result.errors:
                    return json.dumps(
                        {"error": "; ".join(downstream_validation_result.errors)}
                    )

                if downstream_validation_result.project_id == project_id:
                    downstream_pipeline_urls.append({"url": downstream_url})

        return json.dumps(downstream_pipeline_urls)

    def format_display_message(
        self, args: GitLabResourceInput, _tool_response: Any = None
    ) -> str:
        return f"Get downstream pipelines for {args.url}"
