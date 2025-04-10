import json
from typing import Type

from pydantic import BaseModel, Field

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


class GetLogsFromJobInput(BaseModel):
    project_id: int = Field(description="Id of the project")
    job_id: int = Field(description="Id of the Job")


class GetLogsFromJob(DuoBaseTool):
    name: str = "get_job_logs"
    description: str = """
        Get the trace for a job when one has the project_id and job_id. Use this tool to get more
        details for specific jobs within a pipeline. You will need to obtain the project_id and job_id
        from the pipeline details to use this tool.
    """
    args_schema: Type[BaseModel] = GetLogsFromJobInput  # type: ignore

    async def _arun(self, project_id: int, job_id: int) -> str:
        url = f"/api/v4/projects/{project_id}/jobs/{job_id}/trace"

        trace = await self.gitlab_client.aget(path=url, parse_json=False)

        if not trace:
            return "No job found"

        return json.dumps({"job_id": job_id, "trace": trace})

    def format_display_message(self, args: GetLogsFromJobInput) -> str:
        return f"Get logs for job #{args.job_id} in project {args.project_id}"
