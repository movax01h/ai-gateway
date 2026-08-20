from typing import Any, cast

import structlog
from langchain_core.runnables import RunnableConfig
from langgraph.errors import GraphBubbleUp, GraphRecursionError
from langgraph.graph.state import CompiledStateGraph

from duo_workflow_service.agent_platform.utils.exceptions import (
    NotifiableAgentException,
)
from duo_workflow_service.agent_platform.v1.components.agent.component import (
    SUBSESSION_ID_CONTEXT_KEY,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.nodes.delegation_shared import (
    DELEGATION_CALL_ID_CONTEXT_KEY,
    DelegationFatalError,
    DelegationStatus,
    SubsessionRun,
    SubsessionRunKeyFactory,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    IOKey,
    merge_nested_dict,
)
from duo_workflow_service.errors.error_handler import ModelError
from duo_workflow_service.errors.typing import (
    InvalidRequestException,
    NotifiableException,
)
from duo_workflow_service.security.exceptions import SecurityException
from lib.usage_quota.errors import UsageQuotaError
from lib.usage_quota.service import InsufficientCredits

log = structlog.stdlib.get_logger("subagent_dispatch_node")

# Both seeded on every dispatched initial state by
# `DelegationPrepareNode._build_initial_state`, and read back here to scope this
# dispatch's transcript (by subsession) and its run record (by call).
_SUBSESSION_ID_KEY = IOKey(
    target="context", subkeys=[SUBSESSION_ID_CONTEXT_KEY], optional=True
)
_CALL_ID_KEY = IOKey(
    target="context", subkeys=[DELEGATION_CALL_ID_CONTEXT_KEY], optional=True
)

# The dispatched subagent's own UiChatLog entries, forwarded to the supervisor's
# shared log as-is (they are already attributed to this subsession, since
# `_build_initial_state` seeds the subsession ID the subagent's nodes tag their
# entries with). Not subsession-scoped: `ui_chat_log` is a flat, append-reduced
# channel shared by every component in the flow.
_UI_CHAT_LOG_KEY = IOKey(target="ui_chat_log")

# Failures that are *not* attributable to the subagent's own work, and so must
# terminate the whole flow the way they did before subagents ran behind this
# node (v1 dispatched subagents inline, with no guard at all, so every one of
# these propagated). Containing them as a per-call delegation error would both
# lose their distinct downstream handling and invite the supervisor to
# re-delegate against a failure no amount of retrying can fix -- with
# `max_delegations` unset by default and the agent's soft cycle limit at 280,
# that is a long, expensive burn instead of a fast failure.
#
# The counterpart -- what stays contained -- is anything the subagent itself is
# responsible for: `AgentStuckError` (it looped or ignored its wrap-up
# instruction), a tool blowing up, malformed model output. Those are exactly
# the cases where letting the supervisor see the failure and try a different
# delegation is better than killing the flow.
_FATAL_EXCEPTIONS: tuple[type[Exception], ...] = (
    # Reaching the recursion limit inside a subagent is turned into a specific,
    # user-facing "maximum step limit" message by `Flow._handle_compile_and_run_exception`.
    GraphRecursionError,
    # `HumanInputComponent`'s fetch node raises this for a malformed resume
    # request; the flow deliberately does *not* transition to FAILED for it and
    # the server maps it to gRPC INVALID_ARGUMENT.
    InvalidRequestException,
    # Provider failure that already exhausted `ModelErrorHandler`'s retries, or
    # a non-retryable one (auth, permission, invalid request, context too large)
    # that it re-raises on the first occurrence. Terminal by design.
    ModelError,
    # The only exceptions allowed to put their own message in front of the user.
    NotifiableAgentException,
    NotifiableException,
    # Prompt injection and friends: never feed the result back to a model.
    SecurityException,
    # Out of credits/entitlements, or the quota check itself is unavailable.
    InsufficientCredits,
    UsageQuotaError,
)


class SubagentDispatchNode:
    """Runs one managed subagent's isolated subgraph as a native LangGraph node.

    Registered once per managed subagent type (``graph.add_node(subagent_name,
    node.run)``) by ``SupervisorAgentComponentV2.attach``. ``DelegationPrepareNode``
    dispatches to it via ``Send(subagent_name, initial_state)`` — once per
    ``delegate_task`` call targeting this subagent type, run concurrently
    (including multiple concurrent dispatches to the *same* subagent type,
    e.g. two parallel subsessions) by LangGraph's own Pregel scheduler.

    Because this node is a real part of the supervisor's own graph, whatever
    it returns is merged into the supervisor's *shared* ``FlowState`` channels
    using the supervisor's own reducers — unlike a manually ``ainvoke``d
    isolated subgraph, there is no automatic isolation. This node is therefore
    responsible for translating the subagent's own (component-scoped) result
    into an explicitly scoped update before returning it, so that concurrent
    dispatches never collide when they merge back in the same superstep. It
    owns that translation whole, since it owns a synchronous subagent's whole
    runtime, and it writes:

    - one ``SubsessionRun`` record, holding this dispatch's complete outcome,
        under the ``delegate_task`` call ID that dispatched it — the single slot
        ``DelegationCollectNode`` reads to answer that call;
    - the subagent's own ``ui_chat_log`` entries, appended to the flow's shared
        log (already subsession-tagged by the subagent itself, see
        ``_UI_CHAT_LOG_KEY``).

    The subagent's transcript is deliberately *not* among them. It was once
    copied into a subsession-scoped slot of the supervisor's state so that a
    later ``delegate_task`` call could resume that subsession from it; with
    resume gone, nothing ever reads it back, and keeping it would grow the
    parent's checkpoint by a full subagent transcript per delegation for no
    reader. What the supervisor learns from a delegation is its result, not the
    steps that produced it.

    Both sides of the translation go through ``IOKey``s: the subagent's own
    slots are read with the component-scoped keys built in ``__init__``, and the
    run record is written through the key this dispatch's factory returns, so
    its naming convention stays owned by ``SupervisorAgentComponentV2`` alone.

    A subagent's own ``interrupt()`` (e.g. from ``require_tool_approval=True``)
    or a ``Command(graph=Command.PARENT, ...)`` bubbling up from within it is
    *not* caught here — both are subclasses of LangGraph's ``GraphBubbleUp``
    and must propagate unchanged so LangGraph pauses the whole workflow (or
    routes the parent-targeted command), assigning this dispatch its own
    distinct, independently resumable interrupt ID (even when several
    subagent instances are dispatched concurrently in one turn). Neither does
    this node catch any of the ``_FATAL_EXCEPTIONS``, which stay terminal for
    the whole flow. Every *other* exception is caught and recorded as this
    call's ``ERROR`` outcome, so one failing subagent can't take down its
    siblings or the whole supervisor turn — ``DelegationCollectNode`` surfaces
    it as a recoverable per-call error ToolMessage instead.
    """

    def __init__(
        self,
        *,
        subagent_name: str,
        subsession_run_key_factory: SubsessionRunKeyFactory,
        compiled_graph: CompiledStateGraph,
        logger: Any = None,
    ):
        self.name = subagent_name
        self._subsession_run_key_factory = subsession_run_key_factory
        self._compiled_graph = compiled_graph
        self._logger = logger or log

        # The subagent's *own* (component-scoped) answer slot, which is what
        # `AgentComponent.compile_as_subagent` leaves its answer in: every
        # invocation gets a fully isolated FlowState, so it is never
        # subsession-scoped. Read from the returned state and rewritten under
        # this dispatch's call-scoped run record below.
        self._subagent_answer_key = IOKey(
            target="context", subkeys=[subagent_name, "final_answer"], optional=True
        )

    async def run(self, state: FlowState, config: RunnableConfig) -> dict[str, Any]:
        subsession_id = self._require_seeded(state, _SUBSESSION_ID_KEY)
        call_id = self._require_seeded(state, _CALL_ID_KEY)
        run_key = self._subsession_run_key_factory(call_id)

        try:
            # The compiled subagent graph's own state schema is `FlowState`, so
            # its result is one too -- `ainvoke` is just untyped at this seam.
            result_state = cast(
                FlowState, await self._compiled_graph.ainvoke(state, config=config)
            )
        except GraphBubbleUp:
            # Normal control flow: an interrupt, or a parent-targeted command.
            raise
        except _FATAL_EXCEPTIONS:
            # Not this subagent's failure to answer for -- see `_FATAL_EXCEPTIONS`.
            raise
        except Exception as e:  # pylint: disable=broad-except
            self._logger.error(
                f"Subagent invocation failed: {e!r}",
                subagent_name=self.name,
                subsession_id=subsession_id,
            )
            # `ui_chat_log` is absent because `ainvoke` raised,
            # so the subagent's state -- and with it every UI entry the run
            # produced -- never came back. The user therefore sees this
            # delegation's "started" entry and its error result, with nothing in
            # between. Recovering the intermediate entries would mean draining
            # the subagent's `UIHistory` directly, which is shared by concurrent
            # dispatches of the same subagent type (see "Known limitations"), so
            # it is left for the follow-up that scopes that accumulation.
            return run_key.to_nested_dict(
                SubsessionRun(
                    subsession_id=subsession_id,
                    status=DelegationStatus.ERROR,
                    error=str(e),
                    final_answer=None,
                )
            )

        final_answer = self._subagent_answer_key.value_from_state(result_state)
        ui_chat_log = _UI_CHAT_LOG_KEY.value_from_state(result_state) or []

        # Reaching here means the subagent's graph ran to its own terminal node,
        # so this run is COMPLETED whether or not it produced an answer -- an
        # answerless completion is `DelegationCollectNode`'s to report, since
        # only it knows how to phrase that back to the supervisor.
        state_update = run_key.to_nested_dict(
            SubsessionRun(
                subsession_id=subsession_id,
                status=DelegationStatus.COMPLETED,
                error=None,
                final_answer=None if final_answer is None else str(final_answer),
            )
        )
        return merge_nested_dict(
            state_update, _UI_CHAT_LOG_KEY.to_nested_dict(ui_chat_log)
        )

    def _require_seeded(self, state: FlowState, key: IOKey) -> Any:
        """Read one of the values ``DelegationPrepareNode`` seeds on every dispatch.

        A missing value is a wiring/state bug, not a recoverable per-call error:
        without the subsession ID this dispatch's outcome and every UI entry it
        produced would be attributed to no subsession, and without the call ID
        its outcome would be recorded where nothing reads it, hanging the
        supervisor's turn on a ``delegate_task`` call that is never answered.

        Raises:
            DelegationFatalError: If the value is missing.
        """
        value = key.value_from_state(state)
        if value is None:
            raise DelegationFatalError(
                f"Missing {key.template_variable_name!r} in context for subagent "
                f"'{self.name}' dispatch -- DelegationPrepareNode must seed it on "
                f"every dispatched initial state."
            )
        return value
