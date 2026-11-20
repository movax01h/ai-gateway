import json
from typing import Any, Type

from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


class PostDuoCodeReviewInput(BaseModel):
    """Input schema for posting Duo Code Review."""

    project_id: int = Field(description="The project ID")
    merge_request_iid: int = Field(description="The merge request IID")
    review_output: str = Field(
        description="The complete review output containing review comments in XML format"
    )


class PostDuoCodeReview(DuoBaseTool):
    """Tool for posting Duo Code Review to a merge request."""

    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    name: str = "post_duo_code_review"
    description: str = (
        "Post a Duo Code Review to a merge request.\n"
        "Example: post_duo_code_review(project_id=123, merge_request_iid=45, "
        'review_output="<review>...</review>")'
    )
    args_schema: Type[BaseModel] = PostDuoCodeReviewInput

    async def _execute(
        self, project_id: int, merge_request_iid: int, review_output: str, **kwargs: Any
    ) -> str:
        """Execute the tool to post the code review."""
        response = await self._post_review(project_id, merge_request_iid, review_output)
        return self._format_response(response, merge_request_iid)

    async def _post_review(
        self, project_id: int, merge_request_iid: int, review_output: str
    ) -> dict:
        """Post review to GitLab API."""
        request_body = {
            "project_id": project_id,
            "merge_request_iid": merge_request_iid,
            "review_output": review_output,
            "workflow_id": self.workflow_id,
        }
        response = await self.gitlab_client.apost(
            path="/api/v4/ai/duo_workflows/code_review/add_comments",
            body=json.dumps(request_body),
            parse_json=False,
        )

        return json.loads(response.body)

    def _format_response(self, response: dict, merge_request_iid: int) -> str:
        """Format API response as JSON string."""
        if response.get("message") == "Comments added successfully":
            return json.dumps(
                {
                    "status": "success",
                    "message": f"Review posted to MR !{merge_request_iid}",
                }
            )
        raise ToolException(f"Failed to post review: {response}")

    def format_display_message(
        self, args: PostDuoCodeReviewInput, _tool_response: Any = None
    ) -> str:
        """Format a user-friendly display message."""
        return (
            f"Post Duo Code Review to merge request !{args.merge_request_iid} "
            f"in project {args.project_id}"
        )
