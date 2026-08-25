"""Product-analytics events for supervisor-to-subagent delegation.

Privacy: payloads carry only the supervisor and subagent component names, flow
type, workflow id, and subsession bookkeeping - never the delegation prompt or
the subagent's returned content.
"""

from enum import StrEnum
from typing import Optional

from duo_workflow_service.tracking.errors import log_exception
from duo_workflow_service.tracking.monitoring_context import current_monitoring_context
from lib.events import GLReportingEventContext
from lib.internal_events import InternalEventAdditionalProperties, InternalEventsClient
from lib.internal_events.event_enum import EventEnum

__all__ = [
    "DelegationRejectionReason",
    "SubagentDelegationTracker",
    "track_subagent_delegated",
    "track_subagent_delegation_rejected",
    "track_subagent_returned",
]


class DelegationRejectionReason(StrEnum):
    """Why a supervisor refused a ``delegate_task`` call before any subagent ran.

    Coarse by design: the operand is the LLM's mistake, not the wording used to
    correct it, so these stay stable while the messages are free to change.
    """

    LIMIT_REACHED = "limit_reached"
    MIXED_TOOL_CALLS = "mixed_tool_calls"
    PARALLEL_CALL = "parallel_call"
    INVALID_ARGS = "invalid_args"
    INVALID_SUBSESSION = "invalid_subsession"


def _flow_revision_extras() -> dict[str, str]:
    fields = current_monitoring_context.get().flow_versioning_fields()
    return {k: v for k, v in fields.items() if k != "flow_id"}


def track_subagent_delegated(
    internal_event_client: InternalEventsClient,
    *,
    flow_type: str,
    flow_id: str,
    supervisor_name: Optional[str],
    subagent_name: str,
    subsession_id: int,
    is_resume: bool,
    delegation_count: int,
    parallel: bool,
) -> None:
    """Track a supervisor delegating a task to one of its subagents.

    Tracking is strictly best-effort: any exception raised while building or
    emitting the event is logged and swallowed, so tracking must never break the
    delegation path that calls it.

    Args:
        internal_event_client: Client the event is emitted through.
        flow_type: Flow/workflow type (e.g. ``chat``), also used as category.
        flow_id: Workflow id used to correlate with other workflow events.
        supervisor_name: Component name of the delegating supervisor.
        subagent_name: Component name of the subagent being delegated to.
        subsession_id: Subsession that was started or resumed.
        is_resume: Whether an existing subsession was resumed.
        delegation_count: Delegations made by this supervisor so far, including
            this one. Counts one per turn under the sequential supervisor and
            one per concurrently dispatched call under the parallel one.
        parallel: Whether the parallel supervisor produced this delegation.
            ``delegation_count`` and ``is_resume`` are not comparable across
            the two, so slice on this before aggregating either.
    """
    try:
        additional_properties = InternalEventAdditionalProperties(
            label=supervisor_name,
            property=subagent_name,
            value=flow_id,
            subsession_id=subsession_id,
            is_resume=is_resume,
            delegation_count=delegation_count,
            parallel=parallel,
            **_flow_revision_extras(),
        )
        internal_event_client.track_event(
            event_name=EventEnum.WORKFLOW_SUBAGENT_DELEGATED.value,
            additional_properties=additional_properties,
            category=flow_type,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        log_exception(
            e, extra={"event_name": str(EventEnum.WORKFLOW_SUBAGENT_DELEGATED)}
        )


def track_subagent_returned(
    internal_event_client: InternalEventsClient,
    *,
    flow_type: str,
    flow_id: str,
    supervisor_name: Optional[str],
    subagent_name: str,
    subsession_id: int,
    status: str,
    parallel: bool,
) -> None:
    """Track a subagent finishing and returning control to its supervisor.

    Best-effort in the same way as :func:`track_subagent_delegated`.

    Args:
        internal_event_client: Client the event is emitted through.
        flow_type: Flow/workflow type (e.g. ``chat``), also used as category.
        flow_id: Workflow id used to correlate with other workflow events.
        supervisor_name: Component name of the supervisor being returned to.
        subagent_name: Component name of the subagent that returned.
        subsession_id: Subsession that returned.
        status: ``completed`` when the subagent produced a final answer,
            ``error`` when it did not.
        parallel: Whether the parallel supervisor produced this delegation.
    """
    try:
        additional_properties = InternalEventAdditionalProperties(
            label=supervisor_name,
            property=subagent_name,
            value=flow_id,
            subsession_id=subsession_id,
            status=status,
            parallel=parallel,
            **_flow_revision_extras(),
        )
        internal_event_client.track_event(
            event_name=EventEnum.WORKFLOW_SUBAGENT_RETURNED.value,
            additional_properties=additional_properties,
            category=flow_type,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        log_exception(
            e, extra={"event_name": str(EventEnum.WORKFLOW_SUBAGENT_RETURNED)}
        )


def track_subagent_delegation_rejected(
    internal_event_client: InternalEventsClient,
    *,
    flow_type: str,
    flow_id: str,
    supervisor_name: Optional[str],
    reason: DelegationRejectionReason,
    subagent_name: Optional[str],
    parallel: bool,
) -> None:
    """Track a ``delegate_task`` call the supervisor refused before dispatching it.

    No subagent ran, so this is deliberately *not* a delegation: it is counted
    separately so that ``duo_workflow_subagent_delegated`` keeps meaning "a
    delegation happened" and needs no filtering to be summed.

    Best-effort in the same way as :func:`track_subagent_delegated`.

    Args:
        internal_event_client: Client the event is emitted through.
        flow_type: Flow/workflow type (e.g. ``chat``), also used as category.
        flow_id: Workflow id used to correlate with other workflow events.
        supervisor_name: Component name of the refusing supervisor.
        reason: Coarse cause, carried as ``property`` because it is the
            dimension worth slicing on -- unlike the other two events, the
            subagent is frequently unknown here.
        subagent_name: Target subagent when the call named a valid one;
            ``None`` when the arguments never parsed or the rejection covers a
            whole turn rather than one call.
        parallel: Whether the parallel supervisor produced this rejection. The
            sequential supervisor rejects at most once per turn; the parallel
            one rejects per call.
    """
    try:
        additional_properties = InternalEventAdditionalProperties(
            label=supervisor_name,
            property=reason.value,
            value=flow_id,
            subagent_name=subagent_name,
            parallel=parallel,
            **_flow_revision_extras(),
        )
        internal_event_client.track_event(
            event_name=EventEnum.WORKFLOW_SUBAGENT_DELEGATION_REJECTED.value,
            additional_properties=additional_properties,
            category=flow_type,
        )
    except Exception as e:  # pylint: disable=broad-exception-caught
        log_exception(
            e,
            extra={"event_name": str(EventEnum.WORKFLOW_SUBAGENT_DELEGATION_REJECTED)},
        )


class SubagentDelegationTracker:
    """Binds the fields that are constant for one supervisor.

    Fixing ``parallel`` at construction is the point: it separates the
    sequential and parallel regimes in the data, so no emit site should be free
    to restate it.
    """

    def __init__(
        self,
        *,
        flow_id: str,
        flow_type: GLReportingEventContext,
        internal_event_client: InternalEventsClient,
        supervisor_name: str,
        parallel: bool,
    ):
        self._flow_id = flow_id
        self._flow_type = flow_type
        self._internal_event_client = internal_event_client
        self._supervisor_name = supervisor_name
        self._parallel = parallel

    def delegated(
        self,
        *,
        subagent_name: str,
        subsession_id: int,
        is_resume: bool,
        delegation_count: int,
    ) -> None:
        """Record that a task was delegated to a subagent."""
        track_subagent_delegated(
            self._internal_event_client,
            flow_type=self._flow_type.value,
            flow_id=self._flow_id,
            supervisor_name=self._supervisor_name,
            subagent_name=subagent_name,
            subsession_id=subsession_id,
            is_resume=is_resume,
            delegation_count=delegation_count,
            parallel=self._parallel,
        )

    def returned(
        self,
        *,
        subagent_name: str,
        subsession_id: int,
        status: str,
    ) -> None:
        """Record that a subagent finished and returned control."""
        track_subagent_returned(
            self._internal_event_client,
            flow_type=self._flow_type.value,
            flow_id=self._flow_id,
            supervisor_name=self._supervisor_name,
            subagent_name=subagent_name,
            subsession_id=subsession_id,
            status=status,
            parallel=self._parallel,
        )

    def rejected(
        self,
        *,
        reason: DelegationRejectionReason,
        subagent_name: Optional[str] = None,
    ) -> None:
        """Record a delegate_task call refused before any subagent ran."""
        track_subagent_delegation_rejected(
            self._internal_event_client,
            flow_type=self._flow_type.value,
            flow_id=self._flow_id,
            supervisor_name=self._supervisor_name,
            reason=reason,
            subagent_name=subagent_name,
            parallel=self._parallel,
        )
