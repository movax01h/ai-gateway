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
    exclude_allow_failure: bool = Field(
        default=False,
        description="If true, exclude jobs where allow_failure is true.",
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

    If this tool returns no failing jobs, call `get_failing_bridge_jobs` (preferred) or
    `get_downstream_pipelines` with the same pipeline URL, then call this tool with each
    downstream pipeline URL. Only report no failing jobs after checking the original pipeline
    and all downstream pipelines. Do not ask the user before checking downstream pipelines.

    {MERGE_REQUEST_IDENTIFICATION_DESCRIPTION}

    To identify a pipeline you must provide:
    - A GitLab SaaS URL like:
        - https://gitlab.com/namespace/project/-/pipelines/33
        - https://gitlab.com/group/subgroup/project/-/pipelines/42
    - A self-managed GitLab URL like:
        - https://gitlab.example.com/namespace/project/-/pipelines/33
        - https://gitlab.example.com/group/subgroup/project/-/pipelines/42

    For example:
    - Given project_id 13 and merge_request_iid 9, the tool call would be:
        get_pipeline_failing_jobs(project_id=13, merge_request_iid=9)
    - Given a merge request URL https://gitlab.com/namespace/project/-/merge_requests/103, the tool call would be:
        get_pipeline_failing_jobs(url="https://gitlab.com/namespace/project/-/merge_requests/103")
    - Given a pipeline URL https://gitlab.com/namespace/project/-/pipelines/33, the tool call would be:
        get_pipeline_failing_jobs(url="https://gitlab.com/namespace/project/-/pipelines/33")
    """
    args_schema: Type[BaseModel] = GetPipelineFailingJobsInput

    async def _execute(self, **kwargs: Any) -> str:
        url = kwargs.get("url", None)
        project_id = kwargs.get("project_id")
        merge_request_iid = kwargs.get("merge_request_iid", None)
        exclude_allow_failure = kwargs.get("exclude_allow_failure", False)

        pipeline_id = None
        merge_request = None
        validation_result: Optional[
            MergeRequestValidationResult | PipelineValidationResult
        ] = None
        if url and "/-/pipelines/" in url:
            validation_result = self._validate_pipeline_url(url)

            if validation_result.errors:
                raise ToolException("; ".join(validation_result.errors))

            pipeline_id = validation_result.pipeline_iid
        else:
            validation_result = self._validate_merge_request_url(
                url, project_id, merge_request_iid
            )

            if validation_result.errors:
                raise ToolException("; ".join(validation_result.errors))

            merge_request_response = await self.gitlab_client.aget(
                path=f"/api/v4/projects/{validation_result.project_id}/merge_requests/"
                f"{validation_result.merge_request_iid}",
            )

            merge_request = self._process_http_response(
                "fetch merge request", merge_request_response, log
            )

            pipelines_response = await self.gitlab_client.aget(
                path=f"/api/v4/projects/{validation_result.project_id}/merge_requests/"
                f"{validation_result.merge_request_iid}/pipelines",
            )

            pipelines = self._process_http_response(
                "fetch pipelines", pipelines_response, log
            )

            if not isinstance(pipelines, list) or len(pipelines) == 0:
                raise ToolException(
                    f"No pipelines found for merge request iid {validation_result.merge_request_iid}"
                )

            last_pipeline = pipelines[0]
            pipeline_id = last_pipeline["id"]

        failing_jobs = await self._get_failing_jobs(
            validation_result.project_id, pipeline_id, exclude_allow_failure
        )
        if len(failing_jobs) > MAX_JOBS_RETURNED:
            failing_jobs = failing_jobs[:MAX_JOBS_RETURNED]

        if len(failing_jobs) == 0:
            no_failures_msg = "No failing jobs found in this pipeline."
            if merge_request:
                return json.dumps(
                    {
                        "merge_request": merge_request,
                        "pipeline_id": pipeline_id,
                        "failed_jobs": no_failures_msg,
                    }
                )
            return json.dumps(
                {"pipeline_id": pipeline_id, "failed_jobs": no_failures_msg}
            )

        xml_root = etree.Element("jobs")
        for job in failing_jobs:
            xml_job = etree.SubElement(xml_root, "job")
            job_id = job["id"]
            job_name = job["name"]

            job_name_elem = etree.SubElement(xml_job, "job_name")
            job_name_elem.text = job_name

            job_id_elem = etree.SubElement(xml_job, "job_id")
            job_id_elem.text = str(job_id)

            job_url = job.get("web_url")
            if job_url:
                job_url_elem = etree.SubElement(xml_job, "job_url")
                job_url_elem.text = job_url

        failed_jobs_str = "Failed Jobs:\n" + etree.tostring(
            xml_root, pretty_print=True, encoding="unicode"
        )

        if merge_request:
            return json.dumps(
                {"merge_request": merge_request, "failed_jobs": failed_jobs_str}
            )

        return json.dumps({"pipeline_id": pipeline_id, "failed_jobs": failed_jobs_str})

    async def _get_failing_jobs(
        self,
        project_id: str | None,
        pipeline_id: int | None,
        exclude_allow_failure: bool = False,
    ) -> list[dict]:
        jobs = await self._paginate_get(
            endpoint=f"/api/v4/projects/{project_id}/pipelines/{pipeline_id}/jobs",
            per_page=100,
            max_pages=MAX_LOG_PAGES_FOR_PIPELINE,
            extra_params={"scope[]": "failed"},
        )
        if exclude_allow_failure:
            jobs = [job for job in jobs if not job.get("allow_failure", False)]
        return jobs

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
    - A GitLab SaaS URL like:
        - https://gitlab.com/namespace/project/-/pipelines/33
        - https://gitlab.com/group/subgroup/project/-/pipelines/42
    - A self-managed GitLab URL like:
        - https://gitlab.example.com/namespace/project/-/pipelines/33
        - https://gitlab.example.com/group/subgroup/project/-/pipelines/42

    For example:
    - Given a pipeline URL https://gitlab.com/namespace/project/-/pipelines/33, the tool call would be:
        get_downstream_pipelines(url="https://gitlab.com/namespace/project/-/pipelines/33")
    """
    args_schema: Type[BaseModel] = GitLabResourceInput

    async def _execute(self, url: str) -> str:
        validation_result = self._validate_pipeline_url(url)

        if validation_result.errors:
            raise ToolException("; ".join(validation_result.errors))

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
                    raise ToolException("; ".join(downstream_validation_result.errors))

                if downstream_validation_result.project_id == project_id:
                    downstream_pipeline_urls.append(
                        {
                            "url": downstream_url,
                            "status": downstream_pipeline.get("status"),
                        }
                    )

        return json.dumps(downstream_pipeline_urls)

    def format_display_message(
        self, args: GitLabResourceInput, _tool_response: Any = None
    ) -> str:
        return f"Get downstream pipelines for {args.url}"


class GetFailingBridgeJobs(DuoBaseTool):
    name: str = "get_failing_bridge_jobs"
    description: str = """Get the failed bridge jobs in a pipeline.
    A bridge job is the upstream job that triggers a downstream (child or multi-project) pipeline.
    This tool returns ONLY bridges whose own status is `failed`, with the URL of each failed
    bridge's downstream pipeline. Use this to discover nested failed pipelines without paying
    the token cost of fetching successful bridges as well.

    Returns a JSON list (capped) of objects with these fields:
    - id: bridge job ID
    - name: bridge job name
    - stage: pipeline stage the bridge belongs to
    - failure_reason: GitLab's reason string for the failure (may be null)
    - downstream_pipeline_url: web URL of the downstream pipeline, or null when the bridge did not trigger one (e.g., failed before triggering) or when the downstream is in a different project (multi-project triggers are not followed).

    This tool can be used when you have a pipeline URL.

    To identify a pipeline you must provide:
    - A GitLab SaaS URL like:
        - https://gitlab.com/namespace/project/-/pipelines/33
        - https://gitlab.com/group/subgroup/project/-/pipelines/42
    - A self-managed GitLab URL like:
        - https://gitlab.example.com/namespace/project/-/pipelines/33
        - https://gitlab.example.com/group/subgroup/project/-/pipelines/42

    For example:
    - Given a pipeline URL https://gitlab.com/namespace/project/-/pipelines/33, the tool call would be:
        get_failing_bridge_jobs(url="https://gitlab.com/namespace/project/-/pipelines/33")
    """
    args_schema: Type[BaseModel] = GitLabResourceInput

    async def _execute(self, url: str) -> str:
        validation_result = self._validate_pipeline_url(url)

        if validation_result.errors:
            raise ToolException("; ".join(validation_result.errors))

        project_id = validation_result.project_id
        pipeline_iid = validation_result.pipeline_iid
        response = await self.gitlab_client.aget(
            path=f"/api/v4/projects/{project_id}/pipelines/{pipeline_iid}/bridges",
        )

        if not response.is_success():
            error_str = (
                f"Failed to fetch failing bridge jobs: status_code={response.status_code}, "
                f"response={response.body}"
            )
            raise ToolException(error_str)

        bridges = response.body
        if not isinstance(bridges, list):
            raise ToolException(
                f"Failed to fetch failing bridge jobs for url: {url}: {bridges}"
            )

        failed_bridges: list[dict] = []
        for bridge in bridges:
            if bridge.get("status") != "failed":
                continue

            downstream_pipeline = bridge.get("downstream_pipeline")
            downstream_url = None
            if downstream_pipeline:
                # Only handle parent/child pipelines for now, and do not return
                # downstream pipelines across different projects. URLs that fail
                # validation are treated like cross-project URLs and surface as
                # `null` rather than aborting the whole tool call.
                candidate_url = downstream_pipeline.get("web_url")
                if candidate_url:
                    downstream_validation_result = self._validate_pipeline_url(
                        candidate_url
                    )
                    if (
                        not downstream_validation_result.errors
                        and downstream_validation_result.project_id == project_id
                    ):
                        downstream_url = candidate_url

            failed_bridges.append(
                {
                    "id": bridge.get("id"),
                    "name": bridge.get("name"),
                    "stage": bridge.get("stage"),
                    "failure_reason": bridge.get("failure_reason"),
                    "downstream_pipeline_url": downstream_url,
                }
            )

        if len(failed_bridges) > MAX_JOBS_RETURNED:
            failed_bridges = failed_bridges[:MAX_JOBS_RETURNED]

        return json.dumps(failed_bridges)

    def format_display_message(
        self, args: GitLabResourceInput, _tool_response: Any = None
    ) -> str:
        return f"Get failing bridge jobs for {args.url}"
