import json
from typing import Any, Optional, Type

import structlog
from pydantic import BaseModel, Field

from duo_workflow_service.entities.state import Context, WorkflowContext
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

log = structlog.stdlib.get_logger("workflow")


class GetSessionContextInput(BaseModel):
    previous_session_id: int = Field(
        description="The ID of a previously-run session to get context for"
    )


class GetSessionContext(DuoBaseTool):
    name: str = "get_previous_session_context"
    description: str = """Get context from a previously run session.

    This tool retrieves context from a previously run specified session.
    Only use it when prompted by the user to reference a previously executed session.
    Do not provide context for any other session unless explicitly asked.

    Args:
        previous_session_id: The ID of a previously-run session to get context for

    Returns:
        A JSON string containing context data from a previous session or an error message if the context could not be retrieved.
    """
    args_schema: Type[BaseModel] = GetSessionContextInput

    async def _execute(self, previous_session_id: int, **_kwargs: Any) -> str:
        try:
            response = await self.gitlab_client.aget(
                path=f"/api/v4/ai/duo_workflows/workflows/{previous_session_id}/checkpoints?per_page=1",
                parse_json=True,
            )

            if not response.is_success():
                log.error(
                    "Failed to fetch checkpoints: status_code=%s, response=%s",
                    response.status_code,
                    response.body,
                )
                return json.dumps({"error": "API Error"})

            checkpoints = response.body
            if not checkpoints or len(checkpoints) == 0:
                return json.dumps(
                    {"error": "Unable to find checkpoint for this session"}
                )

            return json.dumps(
                {"context": self._format_checkpoint_context(checkpoints[0])}
            )
        except Exception as e:
            return json.dumps({"error": str(e)})

    def format_display_message(
        self, args: GetSessionContextInput, _tool_response: Any = None
    ) -> Optional[str]:
        return f"Get context for session {args.previous_session_id}"

    def _format_checkpoint_context(self, checkpoint: dict) -> str:
        workflow_id = checkpoint.get("metadata", {}).get("thread_id", None)

        if not workflow_id:
            raise ValueError("Invalid checkpoint format. Valid session ID is required")

        if not checkpoint.get("checkpoint") or not checkpoint.get("checkpoint", {}).get(
            "channel_values"
        ):
            context = Context(
                workflow=WorkflowContext(
                    id=workflow_id,
                    plan={"steps": []},
                    goal="No goal available",
                    summary="No summary available",
                )
            )
            return json.dumps(context)

        channel_values = checkpoint["checkpoint"]["channel_values"]
        plan = channel_values.get("plan", {})

        goal = ""
        handover_messages = channel_values.get("handover", [])
        if not isinstance(handover_messages, list):
            raise ValueError(
                "Unable to parse context from last checkpoint for this session"
            )

        if (
            len(handover_messages) > 1
            and isinstance(handover_messages[1], dict)
            and "content" in handover_messages[1]
        ):
            content = handover_messages[1]["content"]
            if "Your goal is: " in content:
                goal = content.split("Your goal is: ")[1]
            else:
                goal = "No goal available"

        summary = ""
        if handover_messages and isinstance(handover_messages[-1], dict):
            summary = handover_messages[-1].get("content", "No summary available")

        context = Context(
            workflow=WorkflowContext(
                id=workflow_id,
                plan=plan,
                goal=goal,
                summary=summary,
            )
        )
        return json.dumps(context)
