from enum import StrEnum
from typing import Any, Optional

import structlog
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from duo_workflow_service.agent_platform.v1.components.supervisor.delegate_task import (
    DelegateTask,
)
from duo_workflow_service.agent_platform.v1.components.supervisor.ui_log import (
    UILogEventsSupervisor,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    IOKey,
    merge_nested_dict,
)
from duo_workflow_service.agent_platform.v1.state.base import RuntimeIOKey
from duo_workflow_service.agent_platform.v1.ui_log import UIHistory
from duo_workflow_service.entities import build_tool_info

log = structlog.stdlib.get_logger("subagent_return_node")


class DelegationStatus(StrEnum):
    COMPLETED = "completed"
    ERROR = "error"


class SubagentReturnNode:
    """Handles subagent completion and injects results back into supervisor history.

    When a subagent calls final_response_tool, its FinalResponseNode writes the
    result to a session-scoped context key. This node then:
    1. Reads the subagent's final_answer from the session-scoped context key
    2. Wraps it in XML delegation_result format
    3. Injects it as a ToolMessage into the supervisor's conversation history
        (matching the delegate_task tool_call_id)
    4. Resets the active session context
    5. Emits a ``delegation_returns`` UI log entry
    6. Routes back to the supervisor's agent node

    All state interactions are performed exclusively through ``IOKey`` instances,
    following the Flow Registry guideline of avoiding direct state dictionary access.

    Args:
        name: LangGraph node name.
        delegate_task_cls: The DelegateTask model class (passed by the component
            so the node stays decoupled from the concrete import).
        active_subsession_key: ``IOKey`` pointing to the active subsession ID.
        active_subagent_name_key: ``IOKey`` pointing to the active subagent name.
        final_answer_key: ``RuntimeIOKey`` that resolves the subagent's
            session-scoped final_answer ``IOKey`` at runtime.
        supervisor_history_key: ``RuntimeIOKey`` that resolves the supervisor's
            conversation-history ``IOKey`` at runtime.
        session_id: The subsession ID of the supervisor component that the
            subagent is returning to. ``None`` for top-level supervisors; set to
            a subsession ID for nested subagent architectures.
        ui_history: ``UIHistory`` for emitting ``delegation_returns`` log entries.
    """

    MESSAGE_SUB_TYPE = "delegation_returns"

    def __init__(
        self,
        *,
        name: str,
        delegate_task_cls: type[DelegateTask],
        active_subsession_key: IOKey,
        active_subagent_name_key: IOKey,
        final_answer_key: RuntimeIOKey,
        supervisor_history_key: RuntimeIOKey,
        session_id: Optional[str] = None,
        ui_history: UIHistory,
    ):
        self.name = name
        self._delegate_task_cls = delegate_task_cls
        self._active_subsession_key = active_subsession_key
        self._active_subagent_name_key = active_subagent_name_key
        self._final_answer_key = final_answer_key
        self._supervisor_history_key = supervisor_history_key
        self._session_id = session_id
        self._ui_history = ui_history

    async def run(self, state: FlowState) -> dict:
        """Inject subagent result into supervisor conversation history."""
        active_session = self._active_subsession_key.value_from_state(state)
        active_subagent_name = self._active_subagent_name_key.value_from_state(state)

        supervisor_history_key = self._supervisor_history_key.to_iokey(state)

        if active_session is None or active_subagent_name is None:
            raise ValueError(
                f"No active subsession found for supervisor "
                f"{supervisor_history_key.target}:{supervisor_history_key.subkeys}. "
                f"active_subsession={active_session}, "
                f"active_subagent_name={active_subagent_name}"
            )

        # Read the subagent's final_answer via the session-scoped RuntimeIOKey
        final_answer = self._final_answer_key.to_iokey(state).value_from_state(state)

        delegate_tool_title: str = self._delegate_task_cls.tool_title
        delegate_args = {
            "subagent_name": active_subagent_name,
            "session_id": active_session,
        }

        # Determine status based on whether we got a result
        if final_answer is not None:
            status = DelegationStatus.COMPLETED
            result_content = final_answer
            tool_response = str(result_content)
            self._ui_history.log.success(
                tool_response,
                event=UILogEventsSupervisor.ON_DELEGATION_RETURNS,
                message_sub_type=self.MESSAGE_SUB_TYPE,
                tool_info=build_tool_info(
                    delegate_tool_title, delegate_args, tool_response
                ),
                subsession_id=self._session_id,
            )
        else:
            status = DelegationStatus.ERROR
            result_content = (
                f"Subagent '{active_subagent_name}' subsession {active_session} "
                f"did not produce a final_answer."
            )
            self._ui_history.log.error(
                result_content,
                event=UILogEventsSupervisor.ON_DELEGATION_RETURNS,
                message_sub_type=self.MESSAGE_SUB_TYPE,
                tool_info=build_tool_info(
                    delegate_tool_title, delegate_args, result_content
                ),
                subsession_id=self._session_id,
            )

        # Find the delegate_task tool_call_id from supervisor's conversation history
        supervisor_history = supervisor_history_key.value_from_state(state) or []

        if not supervisor_history:
            raise ValueError(
                f"No conversation history found for supervisor "
                f"{supervisor_history_key.target}:{supervisor_history_key.subkeys}. "
                f"Cannot attach delegation result ToolMessage without a preceding "
                f"delegate_task tool call to respond to."
            )

        delegate_call_id = self._find_delegate_call_id(supervisor_history)

        # Build XML delegation result
        xml_result = self._format_delegation_result(
            subagent_name=active_subagent_name,
            subsession_id=active_session,
            status=status,
            content=result_content,
        )

        log.info(
            "Sub-agent returned",
            supervisor=f"{supervisor_history_key.target}:{supervisor_history_key.subkeys}",
            subagent_name=active_subagent_name,
            subsession_id=active_session,
            status=status,
        )

        # Inject ToolMessage into supervisor's conversation history
        tool_message = ToolMessage(
            content=xml_result,
            tool_call_id=delegate_call_id,
        )

        # Reset active session context using IOKeys
        context_updates: dict[str, Any] = merge_nested_dict(
            self._active_subsession_key.to_nested_dict(None),
            self._active_subagent_name_key.to_nested_dict(None),
        )

        result: dict[str, Any] = {
            **supervisor_history_key.to_nested_dict(
                supervisor_history + [tool_message]
            ),
            **context_updates,
        }

        ui_updates = self._ui_history.pop_state_updates()
        result = merge_nested_dict(result, ui_updates)

        return result

    def _find_delegate_call_id(self, supervisor_history: list[BaseMessage]) -> str:
        """Find the delegate_task tool_call_id from the supervisor's last AIMessage.

        Walks backward through the supervisor's conversation history to find the most
        recent AIMessage containing a delegate_task tool call.

        Raises ``ValueError`` if no such call is found, if the AIMessage mixes
        delegate_task with other tool calls, or if it contains multiple delegate_task
        calls — all of which indicate invalid state that should never have been committed
        to history.
        """
        tool_title: str = self._delegate_task_cls.tool_title
        for message in reversed(supervisor_history):
            if not (isinstance(message, AIMessage) and message.tool_calls):
                continue

            delegate_calls = [
                tc for tc in message.tool_calls if tc["name"] == tool_title
            ]
            if not delegate_calls:
                continue

            if len(message.tool_calls) > len(delegate_calls):
                other_names = sorted(
                    {
                        tc["name"]
                        for tc in message.tool_calls
                        if tc["name"] != tool_title
                    }
                )
                raise ValueError(
                    f"Found {tool_title} mixed with other tool calls "
                    f"({', '.join(other_names)}) in supervisor conversation history. "
                    f"This indicates invalid state — {tool_title} must be the only "
                    f"tool call in a delegation turn."
                )

            if len(delegate_calls) > 1:
                raise ValueError(
                    f"Found {len(delegate_calls)} {tool_title} calls in a single "
                    f"AIMessage in supervisor conversation history. "
                    f"This indicates invalid state — only one delegation per turn is allowed."
                )

            return str(delegate_calls[0]["id"])

        raise ValueError(
            "Could not find delegate_task tool_call_id in supervisor conversation history"
        )

    @staticmethod
    def _format_delegation_result(
        subagent_name: str,
        subsession_id: int,
        status: DelegationStatus,
        content: str,
    ) -> str:
        """Format the delegation result as XML for the supervisor.

        The XML structure clearly separates session metadata from the subagent's response content, making it reliable
        for the supervisor LLM to parse subsession IDs when deciding which subsession to resume.
        """
        if status == DelegationStatus.ERROR:
            return (
                f"<delegation_result>\n"
                f"  <subagent_name>{subagent_name}</subagent_name>\n"
                f"  <subsession_id>{subsession_id}</subsession_id>\n"
                f"  <status>{status}</status>\n"
                f"  <error>\n"
                f"    {content}\n"
                f"  </error>\n"
                f"</delegation_result>"
            )

        return (
            f"<delegation_result>\n"
            f"  <subagent_name>{subagent_name}</subagent_name>\n"
            f"  <subsession_id>{subsession_id}</subsession_id>\n"
            f"  <status>{status}</status>\n"
            f"  <result>\n"
            f"    {content}\n"
            f"  </result>\n"
            f"</delegation_result>"
        )
