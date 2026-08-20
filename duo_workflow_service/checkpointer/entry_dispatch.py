from dataclasses import dataclass
from typing import Awaitable, Callable

import structlog

from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    STATUS_TO_EVENT_PROPERTY,
    WorkflowStatusEventEnum,
)
from duo_workflow_service.entities import WorkflowStatusEnum
from duo_workflow_service.errors.typing import WorkflowAlreadyFinishedException
from duo_workflow_service.gitlab.gitlab_api import WorkflowConfig
from duo_workflow_service.status_updater.gitlab_status_updater import (
    UnsupportedStatusEvent,
)
from lib.internal_events.event_enum import EventPropertyEnum

__all__ = [
    "EntryDispatchContext",
    "InitialEntryDispatch",
    "rails_status_dispatch",
]


@dataclass(frozen=True)
class EntryDispatchContext:
    """What an entry-dispatch strategy may read to classify a session entry.

    Session viability (the archived guard) is not a strategy concern:
    ``GitLabWorkflow._get_initial_status_event`` applies it to every entry
    before delegating to the strategy.
    """

    workflow_config: WorkflowConfig
    logger: structlog.stdlib.BoundLogger


# The entry-dispatch strategy contract: classify a viable session entry into
# the initial status event plus its tracking property. May raise to reject
# the entry before any graph invocation.
InitialEntryDispatch = Callable[
    [EntryDispatchContext],
    Awaitable[tuple[WorkflowStatusEventEnum, EventPropertyEnum]],
]


async def rails_status_dispatch(
    ctx: EntryDispatchContext,
) -> tuple[WorkflowStatusEventEnum, EventPropertyEnum]:
    """Classify the entry from the Rails workflow status string.

    The default strategy on ``GitLabWorkflow``: analyzes the Rails-side state
    of the workflow to determine whether it's a new workflow (START), a
    resumption of an interrupted workflow (RESUME), or a retry of an existing
    workflow (RETRY).

    Args:
        ctx: The dispatch context of the session entry.

    Returns:
        A tuple containing:
            - WorkflowStatusEventEnum: The status event (START, RESUME, or RETRY)
            - EventPropertyEnum: The associated event property for tracking

    Raises:
        UnsupportedStatusEvent: If a `created` workflow already has checkpoints.
        WorkflowAlreadyFinishedException: If the workflow already finished.
    """
    checkpoint_tuple = (
        ctx.workflow_config.get("latest_checkpoint", None)
        or ctx.workflow_config["first_checkpoint"]
    )
    status = ctx.workflow_config["workflow_status"]

    if status in [
        WorkflowStatusEnum.INPUT_REQUIRED,
        WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED,
        WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED,
    ]:
        if not checkpoint_tuple:
            ctx.logger.error(
                "The workflow record of the GitLab database is a continuous state "
                "but there are no associtated checkopints."
                "This data integrity issue could be caused by the expiring mechanism of checkpoint records."
                "We try to execute this workflow, however, there might be an unexpected behavior.",
                **ctx.workflow_config,
            )

        return WorkflowStatusEventEnum.RESUME, STATUS_TO_EVENT_PROPERTY.get(
            status, EventPropertyEnum.WORKFLOW_RESUME_BY_PLAN
        )

    if status == WorkflowStatusEnum.CREATED:
        if checkpoint_tuple:
            raise UnsupportedStatusEvent(
                f"Workflow with status 'created' should not have existing checkpoints. "
                f"Found checkpoint: {checkpoint_tuple}"
            )
        return WorkflowStatusEventEnum.START, EventPropertyEnum.WORKFLOW_ID

    if status == WorkflowStatusEnum.FINISHED:
        # `finished` is terminal in the Rails state machine: the `retry` the
        # fallback below would send is rejected with a 400 and surfaces as an
        # INTERNAL gRPC error. Nothing is left to execute, so end the session
        # cleanly instead — no status event, no graph run, Rails untouched.
        raise WorkflowAlreadyFinishedException(
            "Session has already finished and cannot be resumed."
        )

    if status == WorkflowStatusEnum.STOPPED:
        # Pure detection — no checkpoint access here. The resolution of this
        # DWS-internal signal into RESUME/START (or a legacy plain RETRY)
        # happens in AbstractWorkflow._resolve_stop_recovery; the wire event
        # sent to Rails is translated to `retry` in _update_workflow_status.
        return (
            WorkflowStatusEventEnum.STOP_RECOVERY,
            EventPropertyEnum.WORKFLOW_RESUME_BY_USER,
        )

    return (
        WorkflowStatusEventEnum.RETRY,
        EventPropertyEnum.WORKFLOW_RESUME_BY_USER,
    )
