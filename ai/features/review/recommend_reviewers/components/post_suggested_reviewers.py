import json
from typing import Any, Literal, Optional, Type

import structlog
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.merge_request import MERGE_REQUESTS_API_PATH
from duo_workflow_service.tools.version_compatibility import (
    get_gitlab_version,
    supports_suggested_reviewers_endpoint,
)

log = structlog.stdlib.get_logger("workflow")


class SuggestedReviewerInput(BaseModel):
    user_id: int = Field(
        description="The numeric ID of the user being suggested as a reviewer."
    )
    reason: Optional[str] = Field(
        default=None,
        max_length=2048,
        description="Optional short rationale explaining why the user is suggested. "
        "At most 2048 characters.",
    )
    approval_rule_id: Optional[int] = Field(
        default=None,
        description="Optional ID of the approval rule the user was drawn from.",
    )
    approval_rule_type: Optional[Literal["merge_request_rule", "project_rule"]] = Field(
        default=None,
        description="Optional type of the approval rule the user was drawn from.",
    )


class PostSuggestedReviewersInput(BaseModel):
    project_id: int = Field(
        description="The numeric ID of the project the merge request belongs to."
    )
    merge_request_iid: int = Field(
        description="The internal ID (iid) of the merge request within the project."
    )
    suggestions: list[SuggestedReviewerInput] = Field(
        max_length=50,
        description="The suggested reviewers to persist, at most 50.",
    )


class PostSuggestedReviewers(DuoBaseTool):
    name: str = "post_suggested_reviewers"
    description: str = """Persist AI-suggested reviewers for a merge request via GitLab's API.

    Stores the suggested reviewers without assigning them, replacing any suggestions
    already stored for the merge request.

    For example:
    - Given project_id 13, merge_request_iid 9, and two suggestions, the tool call would be:
        post_suggested_reviewers(
            project_id=13,
            merge_request_iid=9,
            suggestions=[
                {"user_id": 42, "reason": "Owns the affected module"},
                {"user_id": 57, "reason": "Recently reviewed related code"}
            ]
        )
    """
    args_schema: Type[BaseModel] = PostSuggestedReviewersInput
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    async def _execute(
        self,
        project_id: int,
        merge_request_iid: int,
        suggestions: list[SuggestedReviewerInput],
        **_kwargs: Any,
    ) -> str:
        # Gated before the request below: an older instance answers 404, which tells
        # the model nothing about what to do differently.
        if not supports_suggested_reviewers_endpoint():
            raise ToolException(
                "Saving suggested reviewers requires GitLab 19.4 or later. "
                f"This instance reports {get_gitlab_version()}."
            )

        data = {"suggestions": [s.model_dump(exclude_none=True) for s in suggestions]}

        path = (
            f"{MERGE_REQUESTS_API_PATH.format(project_id=project_id)}/"
            f"{merge_request_iid}/suggested_reviewers"
        )
        response = await self.gitlab_client.apost(
            path=path,
            body=json.dumps(data),
        )

        response = self._process_http_response(
            identifier=path,
            response=response,
            logger=log,
        )

        return json.dumps({"suggested_reviewers": response})

    def format_display_message(
        self, args: PostSuggestedReviewersInput, _tool_response: Any = None
    ) -> str:
        return (
            f"Save {len(args.suggestions)} suggested reviewer(s) for "
            f"merge request !{args.merge_request_iid} in project {args.project_id}"
        )
