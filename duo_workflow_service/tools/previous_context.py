import asyncio
import json
from typing import Any, Optional, Tuple, Type, Union

import structlog
from langchain_core.tools import ToolException
from pydantic import BaseModel, Field

from duo_workflow_service.gitlab.gitlab_workflow_params import fetch_workflow_config
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool

log = structlog.stdlib.get_logger("workflow")

# Maximum number of ui_chat_log entries to include in the returned context.
MAX_UI_CHAT_LOG_ENTRIES = 20

# Maximum characters for tool_response strings in returned log entries.
# Agent and user message content is not truncated.
MAX_TOOL_RESPONSE_CHARS = 300


class GetSessionContextInput(BaseModel):
    previous_session_id: int = Field(
        description=(
            "The ID of a previously-run session (also called a workflow, Duo "
            "Workflow, GitLab Workflow, or Duo Agent Platform session) to get "
            "context for."
        )
    )


class GetSessionContext(DuoBaseTool):
    name: str = "get_previous_session_context"
    description: str = (
        "Retrieve context from a previously run agent session so you can "
        "understand what happened in it. Returns the session's title, goal, "
        "current status, summary, and recent activity (messages and tool "
        "calls).\n"
        "\n"
        "Use this tool whenever the user refers to an earlier session by ID — a "
        '"session", "workflow", "Duo Workflow", "GitLab Workflow", or "Duo Agent '
        'Platform session" (e.g. "what happened in session 123?", "summarise '
        'workflow 4567", "based on agent session 3576, do the following '
        'changes: ...").\n'
        "\n"
        "The returned goal describes what that earlier session set out to do — "
        "it is context, not automatically the goal of the current session.\n"
        "\n"
        "This tool can only retrieve sessions of the same flow type as the "
        "current session (e.g. a chat session can only look up other chat "
        "sessions). Requesting a session of a different flow type returns an "
        "error instead of its context.\n"
        "\n"
        "Args:\n"
        "    previous_session_id: The ID of the previously-run session to "
        "retrieve context for.\n"
        "\n"
        "Returns:\n"
        "    A JSON string with the session context, or an error message if it "
        "cannot be found."
    )
    args_schema: Type[BaseModel] = GetSessionContextInput

    async def _execute(self, previous_session_id: int, **_kwargs: Any) -> str:
        if not self.workflow_id:
            raise ToolException(
                "Unable to determine the current session's flow type; "
                "cannot verify access to another session."
            )

        current_record, previous_record = await asyncio.gather(
            self._fetch_workflow_record(self.workflow_id),
            self._fetch_workflow_record(previous_session_id),
        )
        self._check_same_flow_type(current_record, previous_record, previous_session_id)

        response = await self.gitlab_client.aget(
            path=f"/api/v4/ai/duo_workflows/workflows/{previous_session_id}/checkpoints?per_page=1",
            parse_json=True,
        )

        checkpoints = self._process_http_response("fetch checkpoints", response, log)
        if not checkpoints or len(checkpoints) == 0:
            raise ToolException("Unable to find checkpoint for this session")

        context = self._format_checkpoint_context(
            checkpoints[0], previous_session_id, previous_record
        )
        return json.dumps(context)

    async def _fetch_workflow_record(self, session_id: Union[int, str]) -> dict:
        """Fetch a workflow record for the given session (current or previous).

        Fields such as title/summary/status/workflow_definition live on the
        workflow record (``GET .../workflows/{id}``), not in the checkpoint
        state. A failure here must not raise: callers treat an empty dict as
        "unknown", which the flow-type check below then fails closed on.
        """
        try:
            record = await fetch_workflow_config(self.gitlab_client, str(session_id))
            return record if isinstance(record, dict) else {}
        except Exception as exc:
            log.warning(
                "Failed to fetch workflow record for session context",
                session_id=session_id,
                error=str(exc),
            )
            return {}

    @staticmethod
    def _flow_type_key(record: dict) -> Optional[Tuple[str, Optional[str]]]:
        """Build a comparable "flow type" key from a workflow record.

        ``workflow_definition`` alone is not always a unique flow identifier:
        custom AI Catalog agent flows all share the generic value
        "ai_catalog_agent" (Rails: Ai::Catalog::ExecuteWorkflowService), so two
        unrelated custom flows would otherwise look identical. Pairing it with
        ``ai_catalog_item_version_id`` (null for non-catalog flows) disambiguates
        those cases. Returns None when the record can't establish a flow type at
        all (e.g. the fetch failed), so callers can fail closed.
        """
        workflow_definition = record.get("workflow_definition")
        if not workflow_definition:
            return None
        return (workflow_definition, record.get("ai_catalog_item_version_id"))

    def _check_same_flow_type(
        self,
        current_record: dict,
        previous_record: dict,
        previous_session_id: int,
    ) -> None:
        """Deny access unless both sessions have the same, known flow type."""
        current_key = self._flow_type_key(current_record)
        previous_key = self._flow_type_key(previous_record)

        if previous_key is None:
            reason = "previous session type unknown"
            message = "There was an issue retrieving the previous session."
        elif current_key is None:
            reason = "current session type unknown"
            message = "There was an issue verifying the current session."
        elif current_key != previous_key:
            reason = "flow type mismatch"
            message = (
                f"Session {previous_session_id} belongs to a different flow "
                "type and cannot be accessed from this session."
            )
        else:
            return None

        log.warning(
            "Denied session context access",
            reason=reason,
            current_workflow_id=self.workflow_id,
            previous_session_id=previous_session_id,
            current_flow_type=current_key,
            previous_flow_type=previous_key,
        )
        raise ToolException(message)

    def format_display_message(
        self, args: GetSessionContextInput, _tool_response: Any = None
    ) -> Optional[str]:
        return f"Get context for session {args.previous_session_id}"

    def _format_checkpoint_context(
        self,
        checkpoint: dict,
        previous_session_id: int,
        workflow_record: Optional[dict] = None,
    ) -> dict:
        record: dict = workflow_record or {}

        channel_values: dict = (checkpoint.get("checkpoint") or {}).get(
            "channel_values"
        ) or {}

        # Status: prefer the canonical value from the workflow record, falling
        # back to the checkpoint's channel_values when the record is unavailable.
        status: Optional[str] = record.get("status") or channel_values.get("status")

        # WorkflowState (SW-dev, issue-to-MR) stores goal as a top-level field.
        # FlowState (v1 flow registry) stores it nested under context["goal"].
        # Goal is not exposed on the workflow record, so it comes from the checkpoint.
        goal: Optional[str] = channel_values.get("goal") or (
            channel_values.get("context") or {}
        ).get("goal")

        # These fields live on the workflow record, not the checkpoint state.
        # Note: ``title`` currently defaults to ``workflow_definition`` (Rails
        # set_title_from_workflow_definition), so the two often match; it is still
        # surfaced as the canonical user-facing name and can diverge once titles
        # are customised.
        title: Optional[str] = record.get("title")
        summary: Optional[str] = record.get("summary")
        workflow_definition: Optional[str] = record.get("workflow_definition")

        raw_log: list[dict] = channel_values.get("ui_chat_log") or []
        recent_activity = self._build_recent_activity(raw_log)
        # Derived from the last agent entry in the chat log; distinct from the
        # record-level ``summary`` (which is typically only populated on error).
        last_message = self._extract_last_message(raw_log)

        return {
            "session_id": previous_session_id,
            "title": title,
            "workflow_definition": workflow_definition,
            "status": status,
            "goal": goal,
            "summary": summary,
            "last_message": last_message,
            "recent_activity": recent_activity,
        }

    def _build_recent_activity(self, ui_chat_log: list[dict]) -> list[dict]:
        """Return the last MAX_UI_CHAT_LOG_ENTRIES entries with tool_response truncated.

        If the log has more entries than the limit, a leading system note is inserted making the omission explicit,
        mirroring the "[Showing X of Y]" convention used by repository_files.py. Individual tool_response strings that
        exceed MAX_TOOL_RESPONSE_CHARS are truncated with an inline marker, mirroring mr_discussions.py's
        _truncate_note_body, so the LLM doesn't mistake a cut-off response for the complete output.
        """
        total_entries = len(ui_chat_log)
        entries = ui_chat_log[-MAX_UI_CHAT_LOG_ENTRIES:]
        result: list = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            item: dict = {
                "message_type": entry.get("message_type"),
                "content": entry.get("content"),
                "timestamp": entry.get("timestamp"),
            }
            tool_info = entry.get("tool_info")
            if tool_info and isinstance(tool_info, dict):
                truncated_tool_info: dict = {
                    "name": tool_info.get("name"),
                    "args": tool_info.get("args"),
                }
                tool_response = tool_info.get("tool_response")
                if tool_response is not None:
                    if (
                        isinstance(tool_response, str)
                        and len(tool_response) > MAX_TOOL_RESPONSE_CHARS
                    ):
                        dropped = len(tool_response) - MAX_TOOL_RESPONSE_CHARS
                        tool_response = (
                            tool_response[:MAX_TOOL_RESPONSE_CHARS]
                            + f"\n\n...<TRUNCATED: {dropped} CHARACTERS DROPPED DUE TO SIZE LIMIT>"
                        )
                    truncated_tool_info["tool_response"] = tool_response
                item["tool_info"] = truncated_tool_info
            result.append(item)

        if total_entries > MAX_UI_CHAT_LOG_ENTRIES:
            result.insert(
                0,
                {
                    "message_type": "system",
                    "content": (
                        f"[Showing last {MAX_UI_CHAT_LOG_ENTRIES} of {total_entries} "
                        "total activity entries. Earlier entries omitted.]"
                    ),
                },
            )

        return result

    def _extract_last_message(self, ui_chat_log: list[dict]) -> Optional[str]:
        """Return the content of the last agent-type entry, or None if not found."""
        for entry in reversed(ui_chat_log):
            if isinstance(entry, dict) and entry.get("message_type") == "agent":
                return entry.get("content")
        return None
