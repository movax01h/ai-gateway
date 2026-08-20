from typing import Any, NamedTuple, Optional

import structlog
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command, Send
from pydantic import ValidationError

from duo_workflow_service.agent_platform.constants import NODE_ROLE_SEPARATOR
from duo_workflow_service.agent_platform.v1.components.agent.component import (
    SUBSESSION_ID_CONTEXT_KEY,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.delegate_task import (
    DelegateTask,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.nodes.delegation_shared import (
    DELEGATION_CALL_ID_CONTEXT_KEY,
    DelegationFatalError,
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
from duo_workflow_service.entities.state import WorkflowStatusEnum

log = structlog.stdlib.get_logger("delegation_prepare_node")


def format_subagent_title(subagent_name: str, description: str) -> str:
    """Build a short, human-readable UI title for one delegated subagent call.

    Mirrors the ``"{Agent} Task — {description}"`` title convention used by
    other agentic coding tools (e.g. opencode's ``formatSubagentTitle``) to
    label a subagent invocation in the UI: the subagent's own name
    (title-cased, with ``_`` treated as a word separator, e.g.
    ``research_agent`` -> ``Research Agent``) followed by the LLM-authored
    short task description from this call's ``delegate_task`` arguments.

    Args:
        subagent_name: Name of the managed subagent being dispatched to
            (e.g. ``"research_agent"``).
        description: The short (3-5 word) task description from this call's
            ``delegate_task`` arguments.

    Returns:
        A title string, e.g. ``"Research Agent Task — Investigate flaky test"``.
    """
    title_cased_name = subagent_name.replace("_", " ").title()
    return f"{title_cased_name} Task — {description}"


class _Dispatch(NamedTuple):
    """One validated ``delegate_task`` call, resolved to the subsession it will run.

    Lives only for the duration of ``run``: what outlives the dispatch is the
    ``Send`` LangGraph checkpoints, and the run record
    ``SubagentDispatchNode`` writes under ``call_id``. Nothing here is
    persisted in the supervisor's own state.

    Attributes:
        call_id: The tool call ID this dispatch answers, seeded on its initial
            state so its outcome is recorded against the right call.
        subagent_name: Name of the managed subagent to dispatch to.
        subsession_id: The newly minted subsession ID.
        description: The LLM-authored short (3-5 word) task label, used to
            build the UI title (see ``format_subagent_title``).
        prompt: The delegation prompt for this call.
    """

    call_id: str
    subagent_name: str
    subsession_id: int
    description: str
    prompt: str


class DelegationPrepareNode:  # pylint: disable=too-many-instance-attributes
    """Validates this turn's delegate_task calls and dispatches them as native Send tasks.

    Each managed subagent is compiled (once, at graph-build time) and attached
    as a real LangGraph node of the supervisor's own graph (see
    ``SupervisorAgentComponentV2.attach``). When the supervisor's LLM issues one
    or more ``delegate_task`` calls in a single turn, this node:

    1. Validates each call (delegation limits, argument parsing)
        independently — one invalid call doesn't block the others.
    2. Answers every call that failed validation itself, with an error
        ``ToolMessage``: an unvalidatable call never runs anything, so nothing
        downstream has to know it happened.
    3. Dispatches each valid call, building a fully isolated initial
        ``FlowState`` for it (see ``_build_initial_state``): an empty
        transcript, this call's prompt as ``context.goal``, and its
        ``delegate_task`` call ID. A dispatched subagent starts from nothing
        but its prompt — no call can address a subsession an earlier one
        started, so there is no prior state to carry in.

    Unlike a manual ``asyncio.gather``-based dispatch, ``run`` returns a
    ``Command`` whose ``goto`` holds one ``Send(subagent_name, initial_state)``
    per dispatched call — LangGraph's own Pregel scheduler runs every one of
    them concurrently, as real graph tasks with their own distinct checkpoint
    namespaces and, if one pauses for tool approval, their own
    independently-resumable interrupt ID.

    Dispatching from the node's own return value, rather than from a
    conditional edge, keeps it inside LangGraph's documented contract: a node's
    ``Command`` becomes part of that task's persisted writes, so on resume the
    already-dispatched ``Send`` tasks are replayed from the checkpoint and this
    node is not re-run. Nothing about a dispatch is persisted here to be picked
    up later: each call's outcome is recorded by the node that runs it, under
    that call's own ID (see ``SubsessionRun``), and the dispatch payload itself
    is already checkpointed by LangGraph as the pending ``Send``. Because the
    target node names come from the flow config at runtime, the edges this
    implies are declared with ``add_node(..., destinations=...)`` in
    ``SupervisorAgentComponentV2.attach`` instead of a
    ``Command[Literal[...]]`` annotation.

    All state interactions are performed exclusively through ``IOKey`` instances,
    following the Flow Registry guideline of avoiding direct state dictionary access.
    """

    MESSAGE_SUB_TYPE = "delegation"
    MESSAGE_SUB_TYPE_ERROR = "delegation_error"

    def __init__(
        self,
        *,
        name: str,
        supervisor_name: str,
        max_delegations: Optional[int],
        delegate_task_cls: type[DelegateTask],
        delegation_count_key: IOKey,
        max_subsession_id_key: IOKey,
        supervisor_history_key: RuntimeIOKey,
        ui_history: UIHistory,
        logger: Optional[Any] = None,
    ):
        self.name = name
        self._supervisor_name = supervisor_name
        self._max_delegations = max_delegations
        self._delegate_task_cls = delegate_task_cls
        self._delegation_count_key = delegation_count_key
        self._max_subsession_id_key = max_subsession_id_key
        self._supervisor_history_key = supervisor_history_key
        self._ui_history = ui_history
        self._logger = logger or log

    async def run(
        self,
        state: FlowState,
        config: RunnableConfig,  # pylint: disable=unused-argument
    ) -> Command:
        """Validate every delegate_task tool call from the supervisor's last turn, and dispatch it.

        Returns a ``Command`` carrying both halves of the turn: ``update`` with
        this turn's delegation bookkeeping, error answers and UI entries, and
        ``goto`` with one ``Send(subagent_name, initial_state)`` per call ready
        to dispatch (see ``_build_initial_state``), so LangGraph's own Pregel
        scheduler runs every one of them concurrently.

        When there is nothing left to dispatch — every call failed validation,
        or ``delegate_task`` was mixed with other tool calls in one turn (which
        this node rejects wholesale) — ``goto`` names the supervisor's agent
        node instead: this node has already answered every one of those calls,
        so there is nothing for the collect node to do.
        """
        supervisor_history_key = self._supervisor_history_key.to_iokey(state)
        supervisor_history = supervisor_history_key.value_from_state(state) or []
        delegate_tool_title: str = self._delegate_task_cls.tool_title

        delegate_calls, all_tool_calls = self._extract_delegate_calls(
            supervisor_history, supervisor_history_key
        )
        agent_node = f"{self._supervisor_name}{NODE_ROLE_SEPARATOR}agent"

        if len(all_tool_calls) > len(delegate_calls):
            errors = self._mixed_tool_calls_errors(all_tool_calls, delegate_tool_title)
            self._log_prepared(supervisor_history_key, [], errors)
            return Command(
                update=self._build_update(
                    supervisor_history_key, supervisor_history, errors, {}
                ),
                goto=agent_node,
            )

        delegation_count = self._delegation_count_key.value_from_state(state) or 0
        max_subsession_id = self._max_subsession_id_key.value_from_state(state) or 0

        dispatches, errors, running_count, running_max_id = self._prepare_delegations(
            state, delegate_calls, delegation_count, max_subsession_id
        )

        self._log_prepared(supervisor_history_key, dispatches, errors)

        # Built from `state` as it is *now*, before this node's own update is
        # applied: no subagent is handed (or checkpoints) a copy of the answers
        # its concurrently-running siblings are about to record.
        sends: list[Send] = [
            Send(dispatch.subagent_name, self._build_initial_state(state, dispatch))
            for dispatch in dispatches
        ]

        return Command(
            update=self._build_update(
                supervisor_history_key,
                supervisor_history,
                errors,
                self._bookkeeping_updates(running_count, running_max_id),
            ),
            goto=sends or agent_node,
        )

    def _bookkeeping_updates(
        self,
        running_count: int,
        running_max_id: int,
    ) -> dict[str, Any]:
        """Build the context update for this turn's delegation bookkeeping.

        Two counters, and nothing per-subsession: a delegation prompt is handed to the child as its goal and never read
        again, because no later call can address the subsession it started.
        """
        return merge_nested_dict(
            self._delegation_count_key.to_nested_dict(running_count),
            self._max_subsession_id_key.to_nested_dict(running_max_id),
        )

    def _build_update(
        self,
        supervisor_history_key: IOKey,
        supervisor_history: list[BaseMessage],
        errors: dict[str, str],
        context_updates: dict[str, Any],
    ) -> dict[str, Any]:
        """Assemble this node's state update from bookkeeping, error answers and UI entries.

        Every call in ``errors`` is answered right here with an error
        ``ToolMessage``, rather than being handed to ``DelegationCollectNode``:
        a call that failed validation dispatched nothing, so its answer needs
        neither to wait for this turn's dispatches nor to be persisted for
        another node to pick up.
        """
        if errors:
            error_messages = [
                ToolMessage(content=message, tool_call_id=call_id)
                for call_id, message in errors.items()
            ]
            context_updates = merge_nested_dict(
                context_updates,
                supervisor_history_key.to_nested_dict(
                    supervisor_history + error_messages
                ),
            )
        return merge_nested_dict(context_updates, self._ui_history.pop_state_updates())

    def _mixed_tool_calls_errors(
        self, all_tool_calls: list[ToolCall], delegate_tool_title: str
    ) -> dict[str, str]:
        """Build a per-call error map when delegate_task is mixed with other tools in one turn."""
        other_names = sorted(
            {tc["name"] for tc in all_tool_calls if tc["name"] != delegate_tool_title}
        )
        message = (
            f"You mixed {delegate_tool_title} with other tool calls "
            f"({', '.join(other_names)}) in a single turn. When delegating, "
            f"{delegate_tool_title} must be the only tool call. Please retry "
            f"using only {delegate_tool_title}."
        )
        errors: dict[str, str] = {}
        for tc in all_tool_calls:
            errors[require_tool_call_id(tc)] = message
        return errors

    def _prepare_delegations(
        self,
        state: FlowState,
        delegate_calls: list[ToolCall],
        delegation_count: int,
        max_subsession_id: int,
    ) -> tuple[list[_Dispatch], dict[str, str], int, int]:
        """Validate every delegate_task call and mint the subsession it runs as.

        Every call starts a fresh subsession: there is no way to address an
        existing one, so no two dispatches in a turn can ever target the same
        subsession and no validation is needed to keep them apart.

        Returns ``(dispatches, errors, running_count, running_max_id)`` —
        ``dispatches`` holds one entry per call ready to run, in original order;
        ``errors`` maps the call ID of every call that failed validation to its
        error message; ``running_count``/``running_max_id`` are the new
        delegation_count/max_subsession_id after accounting for every
        subsession created here.
        """
        dispatches: list[_Dispatch] = []
        errors: dict[str, str] = {}
        running_count = delegation_count
        running_max_id = max_subsession_id

        for call in delegate_calls:
            call_id = require_tool_call_id(call)

            if (
                self._max_delegations is not None
                and running_count >= self._max_delegations
            ):
                errors[call_id] = (
                    f"Maximum delegation limit ({self._max_delegations}) "
                    f"reached. You must call final_response_tool to "
                    f"complete the workflow."
                )
                continue

            try:
                delegation = self._delegate_task_cls(**call["args"])
            except (ValidationError, TypeError) as e:
                # ValidationError: args fail DelegateTask's pydantic validation (e.g. bad
                # subagent_name, missing/wrong-typed field). TypeError: call["args"] itself
                # isn't a mapping of str keys to unpack as keyword arguments.
                errors[call_id] = f"Invalid delegate_task arguments: {e}"
                continue

            subagent_name = str(delegation.subagent_name)
            running_max_id += 1
            running_count += 1

            dispatches.append(
                _Dispatch(
                    call_id=call_id,
                    subagent_name=subagent_name,
                    subsession_id=running_max_id,
                    description=delegation.description,
                    prompt=delegation.prompt,
                )
            )

        return dispatches, errors, running_count, running_max_id

    def _build_initial_state(self, state: FlowState, dispatch: _Dispatch) -> FlowState:
        """Build the isolated ``FlowState`` one dispatched delegation is invoked with.

        A resume continues from the subsession's persisted transcript and keeps
        its original goal (falling back to this call's prompt for a subsession
        recorded before goals were persisted at dispatch time), and a new
        subsession starts from an empty transcript with this call's prompt as
        its goal.

        Alongside the subsession ID, the payload carries this call's
        ``delegate_task`` ID: it is what tells ``SubagentDispatchNode`` which
        call the run it is about to execute answers, and it is deliberately
        passed through the dispatch rather than persisted in shared state,
        since LangGraph already checkpoints this payload as the pending
        ``Send``.
        """
        initial_history: list[BaseMessage] = []
        goal_text = dispatch.prompt

        # The supervisor's own sub-dict is deliberately left out of the payload.
        # It holds this component's delegation bookkeeping — including one
        # `SubsessionRun` record per call, each carrying that subagent's full
        # final answer — which no subagent reads. Copying it would put every
        # earlier delegation's answer inside every later dispatch, and each
        # dispatch is checkpointed (as a pending `Send`, and again in the nested
        # run's own lineage), so a long session's checkpoints would grow with the
        # square of its delegation count.
        inherited_context = {
            key: value
            for key, value in (state.get("context") or {}).items()
            if key != self._supervisor_name
        }

        return {
            "status": WorkflowStatusEnum.EXECUTION,
            "conversation_history": {dispatch.subagent_name: initial_history},
            "ui_chat_log": [],
            "context": {
                **inherited_context,
                "goal": goal_text,
                SUBSESSION_ID_CONTEXT_KEY: dispatch.subsession_id,
                DELEGATION_CALL_ID_CONTEXT_KEY: dispatch.call_id,
            },
            "agent_context_limits": {},
        }

    def _log_prepared(
        self,
        supervisor_history_key: IOKey,
        dispatches: list[_Dispatch],
        errors: dict[str, str],
    ) -> None:
        """Log (and add UI entries for) every call this node rejected, then every one it dispatches."""
        delegate_tool_title: str = self._delegate_task_cls.tool_title
        supervisor = f"{supervisor_history_key.target}:{supervisor_history_key.subkeys}"

        for call_id, message in errors.items():
            self._logger.warning(message, supervisor=supervisor)
            self._ui_history.log.error(
                message,
                event=UILogEventsSupervisor.ON_DELEGATION_ERROR,
                message_sub_type=self.MESSAGE_SUB_TYPE_ERROR,
                tool_info=build_tool_info(delegate_tool_title, {}, message),
                message_id=call_id,
                subsession_id=None,
            )

        for dispatch in dispatches:
            log.info(
                "Delegating task",
                supervisor=supervisor,
                subagent_name=dispatch.subagent_name,
                subsession_id=dispatch.subsession_id,
                description=dispatch.description,
            )
            delegate_args = {
                "subagent_name": dispatch.subagent_name,
                "session_id": dispatch.subsession_id,
                "description": dispatch.description,
                "prompt": dispatch.prompt,
            }
            self._ui_history.log.success(
                format_subagent_title(dispatch.subagent_name, dispatch.description),
                event=UILogEventsSupervisor.ON_DELEGATION,
                message_sub_type=self.MESSAGE_SUB_TYPE,
                tool_info=build_tool_info(delegate_tool_title, delegate_args),
                # Tag with this call's tool_call_id so the client can pair this
                # "delegation started" entry with the matching
                # ON_DELEGATION_RETURNS entry (and the ToolMessage answering
                # this call) -- with several delegations dispatched
                # concurrently in a single turn and resolving in arbitrary
                # order, the tool_call_id is the only reliable correlator.
                message_id=dispatch.call_id,
                subsession_id=None,
            )

    def _extract_delegate_calls(
        self,
        supervisor_history: list[BaseMessage],
        supervisor_history_key: IOKey,
    ) -> tuple[list[ToolCall], list[ToolCall]]:
        """Extract every delegate_task tool call from the supervisor's last turn.

        Returns a tuple of ``(delegate_calls, all_tool_calls)``.

        Raises:
            DelegationFatalError: For state/wiring bugs (missing history, last
                message not an AIMessage, or no delegate_task call present at
                all) — these should never occur given the routing that leads
                to this node and indicate a bug rather than a recoverable LLM
                mistake.
        """
        if not supervisor_history:
            raise DelegationFatalError(
                f"No conversation history found for supervisor "
                f"{supervisor_history_key.target}:{supervisor_history_key.subkeys}"
            )

        last_message = supervisor_history[-1]
        if not isinstance(last_message, AIMessage):
            raise DelegationFatalError(
                f"Last message for supervisor "
                f"{supervisor_history_key.target}:{supervisor_history_key.subkeys} "
                f"is not AIMessage"
            )

        tool_title: str = self._delegate_task_cls.tool_title
        all_tool_calls = last_message.tool_calls or []
        delegate_calls = [tc for tc in all_tool_calls if tc["name"] == tool_title]

        if not delegate_calls:
            raise DelegationFatalError(
                f"No {tool_title} tool call found in "
                f"{supervisor_history_key.target}:{supervisor_history_key.subkeys}"
            )

        return delegate_calls, all_tool_calls
