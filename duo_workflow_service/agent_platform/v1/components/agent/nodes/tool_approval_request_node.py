__all__ = ["ToolApprovalRequestNode", "approval_requests_key_for"]

import asyncio
from typing import Any, Optional

import structlog.stdlib
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from duo_workflow_service.agent_platform.utils.tool_event_tracker import (
    ToolEventTracker,
)
from duo_workflow_service.agent_platform.v1.components.agent.nodes._session import (
    DEFAULT_SESSION_ID_KEY,
    resolve_session_id,
)
from duo_workflow_service.agent_platform.v1.components.agent.ui_log import (
    UILogEventsAgent,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    FlowStateKeys,
    IOKey,
    RuntimeIOKey,
)
from duo_workflow_service.agent_platform.v1.state.base import BaseIOKey
from duo_workflow_service.agent_platform.v1.ui_log import DefaultUILogWriter, UIHistory
from duo_workflow_service.entities import (
    ApprovalSource,
    ToolInfo,
    WorkflowStatusEnum,
)
from duo_workflow_service.tools import (
    MalformedToolCallError,
    Toolset,
    UnknownToolError,
    format_tool_display_message,
)
from lib.internal_events.event_enum import EventEnum

log = structlog.stdlib.get_logger(__name__)


def approval_requests_key_for(decision_key: RuntimeIOKey) -> RuntimeIOKey:
    """Build the requests key the request node shares with the fetch node.

    The key resolves beside the decision key (same parent subkeys), so it inherits the decision key's subsession scoping
    under a supervisor.
    """

    def _factory(state: FlowState) -> IOKey:
        decision_iokey = decision_key.to_iokey(state)
        parent_subkeys = (decision_iokey.subkeys or [])[:-1]
        return IOKey(
            target=decision_iokey.target,
            subkeys=[*parent_subkeys, "tool_approval_requests"],
            optional=True,
        )

    return RuntimeIOKey(alias="tool_approval_requests", factory=_factory)


class ToolApprovalRequestNode:
    """Node that validates tool calls and requests human approval.

    This node:
    1. Reads last AIMessage from conversation history
    2. Validates tool calls exist and are well-formed
    3. Filters pre-approved and session-approved tool calls
    4. Generates UI chat log entries for approval
    5. Sets workflow status to TOOL_CALL_APPROVAL_REQUIRED (or EXECUTION
        when nothing needs approval)

    Entries must stay attributed: the notifier streams a PENDING card carrying
    ``component_name`` for each tool call, and ``_merge_ui_chat_log`` matches on
    ``message_id`` alone, so an unattributed entry reusing that id replaces an
    attributed one.

    Args:
        name: Node name
        conversation_history_key: RuntimeIOKey for conversation history
        toolset: Toolset containing available tools
        pre_approved_tools: List of tool names that skip approval
        status_key: RuntimeIOKey for workflow status
        ui_history: UI logging history. Its events must include
            ``ON_TOOL_APPROVAL_REQUEST``; ``run`` raises if nothing survives.
        session_id_key: IOKey resolving the active subsession ID.
        tracker: Optional tool event tracker for approval-request and blocked-tool events
        approval_requests_key: Optional RuntimeIOKey shared with
            ToolApprovalFetchNode; persists which tool calls required approval.
    """

    def __init__(
        self,
        *,
        name: str,
        conversation_history_key: RuntimeIOKey,
        toolset: Toolset,
        pre_approved_tools: list[str],
        status_key: RuntimeIOKey,
        ui_history: UIHistory[DefaultUILogWriter, UILogEventsAgent],
        session_id_key: BaseIOKey = DEFAULT_SESSION_ID_KEY,
        tracker: ToolEventTracker | None = None,
        approval_requests_key: RuntimeIOKey | None = None,
    ):
        self.name = name
        self._conversation_history_key = conversation_history_key
        self._toolset = toolset
        self._pre_approved_tools = set(pre_approved_tools)
        self._status_key = status_key
        self._ui_history = ui_history
        self._session_id_key = session_id_key
        self._tracker = tracker
        self._approval_requests_key = approval_requests_key

    async def run(self, state: FlowState) -> dict[str, Any]:
        """Validate tool calls and request approval."""
        # Get conversation history
        history_iokey = self._conversation_history_key.to_iokey(state)
        history = history_iokey.value_from_state(state) or []

        if not history:
            raise RuntimeError(
                f"No conversation history found for key {history_iokey.target}:{history_iokey.subkeys}"
            )

        last_message = history[-1]

        # Validate last message has tool calls
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            # Agent didn't generate tool calls - add error to history and return
            error_message = HumanMessage(
                content="No tool calls found. Please generate tool calls for the current task."
            )
            history_dict = history_iokey.to_nested_dict(history + [error_message])
            status_dict = self._status_key.to_nested_dict(
                WorkflowStatusEnum.EXECUTION, state
            )
            return {**history_dict, **status_dict}

        # Filter and validate tool calls
        valid_calls, invalid_calls = self._filter_tool_calls(last_message.tool_calls)

        # If any tool calls are invalid, reject the entire batch.
        if invalid_calls:
            # Denied tools are stripped from the toolset, so attempts to call
            # them surface here as invalid calls rather than in the tool node.
            if self._tracker:
                for error in invalid_calls:
                    blocked_name = error.tool_call.get("name")
                    if blocked_name in self._toolset.denied_tools:
                        self._tracker.track_tool_governance_event(
                            event_name=EventEnum.WORKFLOW_TOOL_BLOCKED,
                            tool_name=blocked_name,
                        )

            invalid_by_id = {e.tool_call["id"]: e for e in invalid_calls}
            error_messages = [
                ToolMessage(
                    tool_call_id=call["id"],
                    content=(
                        str(invalid_by_id[call["id"]])
                        if call["id"] in invalid_by_id
                        else "Tool call cancelled because another call in this batch was invalid."
                    ),
                )
                for call in last_message.tool_calls
            ]
            history_dict = history_iokey.to_nested_dict(history + error_messages)
            status_dict = self._status_key.to_nested_dict(
                WorkflowStatusEnum.EXECUTION, state
            )
            return {**history_dict, **status_dict}

        # Filter out pre-approved and session-approved tool calls. Checks may
        # hit the GitLab instance, so run them concurrently across the batch.
        skip_sources = await asyncio.gather(
            *(self._should_skip_approval(call) for call in valid_calls)
        )
        needs_approval = []
        for call, source in zip(valid_calls, skip_sources):
            if source is None:
                needs_approval.append(call)
            else:
                log.info(
                    "Skipping tool call approval",
                    tool_name=call["name"],
                    approval_source=source,
                )

        # If all tools are pre-approved, skip approval entirely
        if not needs_approval:
            # Set status to EXECUTION so router can explicitly route to tools
            status_dict = self._status_key.to_nested_dict(
                WorkflowStatusEnum.EXECUTION, state
            )
            return status_dict

        self._emit_approval_requests(
            needs_approval, resolve_session_id(self._session_id_key, state)
        )

        result = self._status_key.to_nested_dict(
            WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED, state
        )

        # Persist which calls required approval so the fetch node resolves exactly this set (ids/names only).
        if self._approval_requests_key is not None:
            requested_calls = [
                {"id": call.get("id"), "name": call["name"]} for call in needs_approval
            ]
            result = {
                **result,
                **self._approval_requests_key.to_nested_dict(requested_calls, state),
            }

        # Guard on what survived the event filter, not on what was written.
        updates = self._ui_history.pop_state_updates()
        if not updates[FlowStateKeys.UI_CHAT_LOG]:
            raise RuntimeError("No valid tool calls found to display for approval")

        # Emitted after the guard so a failed render cannot orphan request events.
        if self._tracker:
            for call in needs_approval:
                self._tracker.track_tool_governance_event(
                    event_name=EventEnum.WORKFLOW_TOOL_APPROVAL_REQUESTED,
                    tool_name=call["name"],
                )

        return {**result, **updates}

    def _filter_tool_calls(
        self, tool_calls: list
    ) -> tuple[list, list[MalformedToolCallError]]:
        """Filter tool calls into valid and invalid lists.

        Args:
            tool_calls: List of tool calls from AIMessage

        Returns:
            Tuple of (valid_calls, invalid_call_errors)
        """
        valid_calls = []
        invalid_calls = []

        for tool_call in tool_calls:
            try:
                self._toolset.validate_tool_call(tool_call)
                valid_calls.append(tool_call)
            except MalformedToolCallError as e:
                invalid_calls.append(e)

        return valid_calls, invalid_calls

    async def _should_skip_approval(self, tool_call: dict) -> Optional[ApprovalSource]:
        """Check if a tool call should skip approval.

        A tool call skips approval if:
        1. Its tool is in the component's pre_approved_tools list, OR
        2. The toolset reports no approval is required for this exact call
            (privilege-level pre-approval or a session approval persisted
            on the GitLab instance)

        Returns the source of the skip (``PREAPPROVED_CONFIG`` for a
        privilege/component pre-approval, ``SESSION_APPROVAL`` for a session
        approval persisted on the GitLab instance) if the tool call should
        skip approval, or None if human approval is still required.
        """
        tool_name = tool_call["name"]

        # Component-level pre-approval short-circuits before any network check
        if tool_name in self._pre_approved_tools:
            log.debug(
                "Tool call approval skipped: component pre_approved_tools",
                tool_name=tool_name,
            )
            return ApprovalSource.PREAPPROVED_CONFIG

        try:
            return await self._toolset.resolve_approval_source(
                tool_name, tool_call.get("args")
            )
        except UnknownToolError:
            # Defensive: unknown tools are already rejected by
            # _filter_tool_calls before this point. Fail toward requiring
            # approval anyway.
            return None

    def _emit_approval_requests(
        self, tool_calls: list, session_id: Optional[str]
    ) -> None:
        """Write one approval-request entry per renderable tool call.

        All rendering happens before the first write: ``UIHistory`` only clears
        in ``pop_state_updates``, so a mid-batch raise would otherwise strand
        entries for the next run to flush as its own.
        """
        renderable: list[tuple[dict, str]] = []

        for call in tool_calls:
            tool = self._toolset[call["name"]]

            # Get formatted display message for the tool
            msg = format_tool_display_message(tool, call["args"])
            if msg is None:
                continue

            renderable.append((call, msg))

        for call, msg in renderable:
            self._ui_history.log.success(
                msg,
                event=UILogEventsAgent.ON_TOOL_APPROVAL_REQUEST,
                message_id=call["id"],
                tool_info=ToolInfo(name=call["name"], args=call["args"]),
                additional_context=None,
                subsession_id=session_id,
            )
