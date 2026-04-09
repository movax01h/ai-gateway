from typing import Any, Callable, NamedTuple, Optional

import structlog
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from duo_workflow_service.agent_platform.experimental.components.supervisor.delegate_task import (
    DelegateTask,
)
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    IOKey,
    merge_nested_dict,
)
from duo_workflow_service.agent_platform.experimental.state.base import RuntimeIOKey

# Factory that builds a subsession-scoped conversation-history IOKey given the
# subagent type name and subsession ID.  Defined as a named type so callers
# can annotate constructor parameters without importing Callable directly.
SubsessionHistoryKeyFactory = Callable[[str, int], IOKey]

SUBSESSION_KEY_SEPARATOR = "__"

log = structlog.stdlib.get_logger("delegation_node")


class ExtractedDelegateCall(NamedTuple):
    """A single delegate_task tool call extracted and parsed from supervisor history.

    Attributes:
        call_id: The tool call ID, used to match the corresponding ToolMessage response.
        delegation: The parsed DelegateTask instance containing subagent_name,
            subsession_id, and prompt.
    """

    call_id: str
    delegation: DelegateTask


class SubsessionResult(NamedTuple):
    """Result of starting or resuming a subagent subsession.

    Attributes:
        subsession_id: The ID of the subsession (new or resumed).
        new_max_id: The updated maximum subsession ID.  For new subsessions
            this equals ``subsession_id``; for resumed ones it is unchanged.
        state_updates: All state writes produced by the subsession processing,
            ready to be merged into the graph state by ``run()``.  For new
            subsessions this includes both the empty conversation-history write
            and the subsession-scoped goal IOKey write; for resumed subsessions
            it contains the updated conversation history with the new
            HumanMessage appended.
    """

    subsession_id: int
    new_max_id: int
    state_updates: dict[str, Any]


class DelegationFatalError(Exception):
    """Unrecoverable delegation error indicating a graph wiring or state corruption bug.

    These errors propagate up and stop execution — they should never occur during normal operation and cannot be
    meaningfully handled by the LLM.
    """


class DelegationError(Exception):
    """Recoverable delegation error that the supervisor LLM can react to.

    Carries ``call_ids`` — the IDs of all delegate_task tool calls that need a
    ToolMessage response.  Normally this is a single-element list, but when the
    LLM issues multiple delegate_task calls in one turn ``run()`` must respond
    with one ToolMessage per call to satisfy the AIMessage/ToolMessage pairing
    requirement.
    """

    def __init__(self, message: str, call_ids: list[str]) -> None:
        super().__init__(message)
        self.call_ids = call_ids


class DelegationNode:
    """Handles delegate_task tool calls from the supervisor.

    Responsibilities:
    - Assigns or validates subsession IDs
    - For new subsessions: writes the delegation prompt to the subsession-scoped goal IOKey;
        conversation history starts empty (subagent reads goal via inputs)
    - For resumed subsessions: appends a HumanMessage to the existing conversation history
    - Sets the active subsession context
    - Tracks delegation count for safety limits

    Routing after this node is handled by
    ``SupervisorAgentComponent._delegation_router``, which owns all routing
    decisions for the supervisor graph.

    All state interactions are performed exclusively through ``IOKey`` instances,
    following the Flow Registry guideline of avoiding direct state dictionary access.
    """

    def __init__(
        self,
        *,
        name: str,
        max_delegations: Optional[int],
        delegate_task_cls: type[DelegateTask],
        delegation_count_key: IOKey,
        active_subsession_key: IOKey,
        active_subagent_name_key: IOKey,
        max_subsession_id_key: IOKey,
        supervisor_history_key: RuntimeIOKey,
        # The subsession history includes subsession id
        # within IOKey. This node is responsible for resolving new
        # active subsession id (either assigns new id value
        # or selects one to resume), therefore at the execution time
        # active subsession id is not present within graph state
        # which prevents RuntimeIOKey abstraction from being applicable
        # in this case
        subsession_history_key_factory: SubsessionHistoryKeyFactory,
        subsession_goal_key_factory: Callable[[str, int], IOKey],
    ):
        self.name = name
        self._max_delegations = max_delegations
        self._delegate_task_cls = delegate_task_cls
        self._delegation_count_key = delegation_count_key
        self._active_subsession_key = active_subsession_key
        self._active_subagent_name_key = active_subagent_name_key
        self._max_subsession_id_key = max_subsession_id_key
        self._supervisor_history_key = supervisor_history_key
        self._subsession_history_key_factory = subsession_history_key_factory
        self._subsession_goal_key_factory = subsession_goal_key_factory

    async def run(self, state: FlowState) -> dict[str, Any]:
        """Process a delegate_task tool call from the supervisor."""
        supervisor_history_key = self._supervisor_history_key.to_iokey(state)
        supervisor_history = supervisor_history_key.value_from_state(state) or []

        try:
            call_id, delegation = self._extract_delegate_call(
                supervisor_history, supervisor_history_key
            )

            delegation_count = self._delegation_count_key.value_from_state(state) or 0
            max_subsession_id = self._max_subsession_id_key.value_from_state(state) or 0

            if (
                self._max_delegations is not None
                and delegation_count >= self._max_delegations
            ):
                raise DelegationError(
                    f"Maximum delegation limit ({self._max_delegations}) reached. "
                    f"You must call final_response_tool to complete the workflow.",
                    call_ids=[call_id],
                )

            subagent_name = str(delegation.subagent_name)

            if delegation.subsession_id is None:
                subsession_result = self._start_new_subsession(
                    max_subsession_id=max_subsession_id,
                    subagent_name=subagent_name,
                    prompt=delegation.prompt,
                )
            else:
                subsession_result = self._resume_subsession(
                    state=state,
                    subsession_id=delegation.subsession_id,
                    subagent_name=subagent_name,
                    max_subsession_id=max_subsession_id,
                    prompt=delegation.prompt,
                    call_id=call_id,
                )

        except DelegationError as e:
            supervisor_key_id = (
                f"{supervisor_history_key.target}:{supervisor_history_key.subkeys}"
            )
            log.warning(str(e), supervisor=supervisor_key_id)
            # One ToolMessage per call ID — required because every tool call in an
            # AIMessage must be matched by a corresponding ToolMessage.
            error_messages = [
                ToolMessage(content=str(e), tool_call_id=call_id)
                for call_id in e.call_ids
            ]
            return supervisor_history_key.to_nested_dict(
                supervisor_history + error_messages
            )

        log.info(
            "Delegating task",
            supervisor=f"{supervisor_history_key.target}:{supervisor_history_key.subkeys}",
            subagent_name=subagent_name,
            subsession_id=subsession_result.subsession_id,
            delegation_count=delegation_count + 1,
            is_resume=delegation.subsession_id is not None,
        )

        context_updates: dict[str, Any] = {}
        for iokey, value in (
            (self._max_subsession_id_key, subsession_result.new_max_id),
            (self._active_subsession_key, subsession_result.subsession_id),
            (self._active_subagent_name_key, subagent_name),
            (self._delegation_count_key, delegation_count + 1),
        ):
            context_updates = merge_nested_dict(
                context_updates,
                iokey.to_nested_dict(value),
            )

        return merge_nested_dict(context_updates, subsession_result.state_updates)

    def _extract_delegate_call(
        self,
        supervisor_history: list[BaseMessage],
        supervisor_history_key: IOKey,
    ) -> ExtractedDelegateCall:
        """Extract and parse the delegate_task tool call from supervisor history.

        Covers the full pipeline: validate history → validate last message →
        find tool call → parse arguments.

        Returns an ``ExtractedDelegateCall`` on success.
        Raises ``DelegationFatalError`` for state/wiring bugs.
        Raises ``DelegationError`` for recoverable parse failures.
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
        tool_calls = last_message.tool_calls or []
        all_tool_call_ids: list[str] = []
        for tc in tool_calls:
            assert isinstance(tc["id"], str), "Tool call id must be a string"
            all_tool_call_ids.append(tc["id"])

        delegate_calls = [tc for tc in tool_calls if tc["name"] == tool_title]

        if not delegate_calls:
            raise DelegationFatalError(
                f"No {tool_title} tool call found in "
                f"{supervisor_history_key.target}:{supervisor_history_key.subkeys}"
            )

        if len(tool_calls) > len(delegate_calls):
            # Other tool calls are mixed with delegate_task in the same turn.
            # This is not allowed — delegate_task must be the only tool call.
            # Every call in the AIMessage must receive a corresponding ToolMessage.
            other_names = sorted(
                {tc["name"] for tc in tool_calls if tc["name"] != tool_title}
            )
            raise DelegationError(
                f"You mixed {tool_title} with other tool calls ({', '.join(other_names)}) "
                f"in a single turn. When delegating, {tool_title} must be the only tool "
                f"call. Please retry using only {tool_title}.",
                call_ids=all_tool_call_ids,
            )

        if len(delegate_calls) > 1:
            # Parallel delegation is not supported — subagents execute sequentially.
            # Every call in the AIMessage must receive a corresponding ToolMessage,
            # so we collect all call IDs and let run() fan out one ToolMessage each.
            raise DelegationError(
                f"You called {tool_title} {len(delegate_calls)} times in a single turn. "
                f"Parallel delegation is not supported — "
                f"you must delegate to one subagent at a time. "
                f"Please call {tool_title} again with a single delegation.",
                call_ids=all_tool_call_ids,
            )

        raw_call = delegate_calls[0]
        assert isinstance(raw_call["id"], str), "Tool call id must be a string"
        call_id: str = raw_call["id"]
        try:
            delegation = self._delegate_task_cls(**raw_call["args"])
        except Exception as e:
            raise DelegationError(
                f"Invalid delegate_task arguments: {e}",
                call_ids=[call_id],
            ) from e

        return ExtractedDelegateCall(call_id=call_id, delegation=delegation)

    def _start_new_subsession(
        self,
        *,
        max_subsession_id: int,
        subagent_name: str,
        prompt: str,
    ) -> SubsessionResult:
        """Start a new subsession with an empty conversation history.

        Writes the delegation prompt to the subsession-scoped goal IOKey
        (``context:<supervisor>__<subagent>__<id>/goal``) so the subagent reads
        it as its ``{{goal}}`` template variable.  No ``HumanMessage`` is seeded
        into the conversation history — the subagent starts fresh and receives
        the goal through its prompt inputs.

        Returns a ``SubsessionResult`` with ``subsession_id == new_max_id``
        (both equal ``max_subsession_id + 1``) and ``state_updates`` containing
        only the subsession-scoped goal IOKey write.  The conversation history
        for the new subsession is implicitly empty (no key exists in state yet),
        so no explicit empty-list write is needed.

        Note: for a new subsession ``subsession_id == new_max_id``, whereas for
        a resumed subsession they differ (``new_max_id`` stays unchanged).  The
        uniform return type keeps the call-site consistent for both branches.
        """
        subsession_id = max_subsession_id + 1
        goal_iokey = self._subsession_goal_key_factory(subagent_name, subsession_id)
        return SubsessionResult(
            subsession_id=subsession_id,
            new_max_id=subsession_id,
            state_updates=goal_iokey.to_nested_dict(prompt),
        )

    def _resume_subsession(
        self,
        *,
        state: FlowState,
        subsession_id: int,
        subagent_name: str,
        max_subsession_id: int,
        prompt: str,
        call_id: str,
    ) -> SubsessionResult:
        """Resume an existing subsession, appending the new prompt as a HumanMessage.

        Returns a ``SubsessionResult`` with ``new_max_id`` unchanged and
        ``state_updates`` containing the updated conversation history.
        Raises ``DelegationError`` for recoverable failures (invalid ID, missing history).
        """
        if subsession_id < 1 or subsession_id > max_subsession_id:
            raise DelegationError(
                f"Invalid subsession ID {subsession_id}. "
                f"Valid range: 1 to {max_subsession_id}",
                call_ids=[call_id],
            )

        subsession_history_iokey = self._subsession_history_key_factory(
            subagent_name, subsession_id
        )
        existing_history: list[BaseMessage] = (
            subsession_history_iokey.value_from_state(state) or []
        )

        if not existing_history:
            raise DelegationError(
                f"No conversation history found for subsession {subsession_id} "
                f"(subagent_name: {subagent_name}). The subsession may belong to "
                f"a different subagent.",
                call_ids=[call_id],
            )

        return SubsessionResult(
            subsession_id=subsession_id,
            new_max_id=max_subsession_id,
            state_updates=subsession_history_iokey.to_nested_dict(
                existing_history + [HumanMessage(content=prompt)]
            ),
        )
