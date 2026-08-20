from typing import Any, Optional

import structlog
from langchain_core.messages import AIMessage, BaseMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig

from duo_workflow_service.agent_platform.v1.components.supervisor_v2.delegate_task import (
    DelegateTask,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.nodes.delegation_shared import (
    DelegationFatalError,
    DelegationStatus,
    SubsessionRun,
    SubsessionRunKeyFactory,
    format_delegation_result,
    require_tool_call_id,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.ui_log import (
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

log = structlog.stdlib.get_logger("delegation_collect_node")


class DelegationCollectNode:
    """Answers this turn's dispatched delegate_task calls, once every dispatch completes.

    Runs after every ``DelegationPrepareNode``-dispatched ``Send`` task for
    this turn has completed: a static fan-in edge from each managed subagent
    node (see ``SupervisorAgentComponentV2.attach``) makes LangGraph hold this
    node until the subagent tasks actually scheduled this turn have all
    finished. Managed subagent types that received no ``Send`` this turn simply
    do not run, and so do not hold it back. Its input is the
    supervisor's own conversation history: every ``delegate_task`` call in the
    last turn that has no ``ToolMessage`` answer yet is a call
    ``DelegationPrepareNode`` dispatched, and for each one (in original
    tool-call order) this node reads back the ``SubsessionRun`` record the
    dispatch wrote under that call's ID and turns it into that call's
    ``ToolMessage``.

    Deriving the work from unanswered calls, rather than from a manifest
    persisted by ``DelegationPrepareNode``, is what keeps the two nodes
    independent: a call that failed validation was already answered by the
    prepare node (so it is not unanswered here), and a call that was dispatched
    is answered from a record written by the one node that ran it. Nothing has
    to be kept in sync between the two ends of a turn, and no turn-scoped state
    has to be cleared afterwards.

    Always routes straight back to the supervisor's agent node — there is no
    "active subsession" concept to route around, since every dispatched
    subagent this turn has already completed (or failed) by the time this
    node runs.
    """

    MESSAGE_SUB_TYPE_RETURNS = "delegation_returns"

    def __init__(
        self,
        *,
        name: str,
        delegate_task_cls: type[DelegateTask],
        subsession_run_key_factory: SubsessionRunKeyFactory,
        supervisor_history_key: RuntimeIOKey,
        ui_history: UIHistory,
        logger: Optional[Any] = None,
    ):
        self.name = name
        self._delegate_task_cls = delegate_task_cls
        self._subsession_run_key_factory = subsession_run_key_factory
        self._supervisor_history_key = supervisor_history_key
        self._ui_history = ui_history
        self._logger = logger or log

    async def run(
        self,
        state: FlowState,
        config: RunnableConfig,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        supervisor_history_key = self._supervisor_history_key.to_iokey(state)
        supervisor_history = supervisor_history_key.value_from_state(state) or []
        delegate_tool_title: str = self._delegate_task_cls.tool_title

        new_messages: list[ToolMessage] = []

        for call in self._dispatched_calls(supervisor_history, supervisor_history_key):
            call_id = require_tool_call_id(call)
            run_record = self._run_record(state, call_id)
            subagent_name = str(call["args"]["subagent_name"])
            subsession_id = run_record["subsession_id"]
            status, content = self._resolve_run(run_record, subagent_name)

            self._ui_history.log.success(
                content,
                event=UILogEventsSupervisor.ON_DELEGATION_RETURNS,
                message_sub_type=self.MESSAGE_SUB_TYPE_RETURNS,
                tool_info=build_tool_info(
                    delegate_tool_title,
                    {
                        "subagent_name": subagent_name,
                        "session_id": subsession_id,
                        "description": call["args"]["description"],
                    },
                    content,
                ),
                # Same tool_call_id as this call's ON_DELEGATION entry (see
                # `DelegationPrepareNode._log_prepared`) and as the
                # ToolMessage built below, so the client can pair a result
                # with the delegation that produced it even when several
                # delegations run concurrently in a single turn.
                message_id=call_id,
                subsession_id=str(subsession_id),
            )
            log.info(
                "Sub-agent returned",
                supervisor=f"{supervisor_history_key.target}:{supervisor_history_key.subkeys}",
                subagent_name=subagent_name,
                subsession_id=subsession_id,
                status=status,
            )

            xml_result = format_delegation_result(
                subagent_name=subagent_name,
                status=status,
                content=content,
            )
            new_messages.append(ToolMessage(content=xml_result, tool_call_id=call_id))

        ui_updates = self._ui_history.pop_state_updates()
        return merge_nested_dict(
            supervisor_history_key.to_nested_dict(supervisor_history + new_messages),
            ui_updates,
        )

    def _dispatched_calls(
        self,
        supervisor_history: list[BaseMessage],
        supervisor_history_key: IOKey,
    ) -> list[ToolCall]:
        """Return this turn's still-unanswered delegate_task calls, in original order.

        Raises:
            DelegationFatalError: If the supervisor's last turn holds no
                unanswered ``delegate_task`` call at all — this node is only
                ever reached from a ``DelegationPrepareNode`` dispatch, so
                having nothing to answer means the graph is miswired or the
                history was corrupted, not that the LLM did something odd.
        """
        supervisor = f"{supervisor_history_key.target}:{supervisor_history_key.subkeys}"
        last_turn = next(
            (m for m in reversed(supervisor_history) if isinstance(m, AIMessage)), None
        )
        answered = {
            message.tool_call_id
            for message in supervisor_history
            if isinstance(message, ToolMessage)
        }
        tool_title: str = self._delegate_task_cls.tool_title
        calls = [
            call
            for call in (last_turn.tool_calls or [] if last_turn else [])
            if call["name"] == tool_title and call["id"] not in answered
        ]

        if not calls:
            raise DelegationFatalError(
                f"No unanswered {tool_title} tool call found in {supervisor} -- "
                f"this node only runs on dispatches made by DelegationPrepareNode."
            )

        return calls

    def _run_record(self, state: FlowState, call_id: str) -> SubsessionRun:
        """Read back the run record the dispatch of ``call_id`` wrote.

        Raises:
            DelegationFatalError: If no record exists. ``SubagentDispatchNode``
                writes one for every dispatch it completes, successfully or
                not, so a missing record means this call's ``Send`` never ran
                (or ran against a different call ID) — leaving its
                ``delegate_task`` call unanswerable, which would wedge the
                supervisor's next turn on an incomplete tool-call sequence.
        """
        record: Optional[SubsessionRun] = self._subsession_run_key_factory(
            call_id
        ).value_from_state(state)
        if record is None:
            raise DelegationFatalError(
                f"No subsession run recorded for {self._delegate_task_cls.tool_title} "
                f"call {call_id!r} -- SubagentDispatchNode records every dispatch it "
                f"completes."
            )
        return record

    def _resolve_run(
        self, run_record: SubsessionRun, subagent_name: str
    ) -> tuple[DelegationStatus, str]:
        """Turn one dispatched delegation's run record into ``(status, content)``."""
        subsession_id = run_record["subsession_id"]

        if run_record["status"] == DelegationStatus.ERROR:
            error_message = run_record["error"]
            self._logger.error(
                f"Subagent invocation failed: {error_message}",
                subagent_name=subagent_name,
                subsession_id=subsession_id,
            )
            content = (
                f"Subagent '{subagent_name}' subsession {subsession_id} "
                f"failed: {error_message}"
            )
            return DelegationStatus.ERROR, content

        final_answer = run_record["final_answer"]
        if final_answer is not None:
            return DelegationStatus.COMPLETED, final_answer

        content = (
            f"Subagent '{subagent_name}' subsession {subsession_id} "
            f"did not produce a final_answer."
        )
        return DelegationStatus.ERROR, content
