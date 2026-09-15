from typing import Any, Optional, cast
from unittest.mock import MagicMock

import pytest

from duo_workflow_service.checkpointer.entry_dispatch import (
    EntryDispatchContext,
    rails_status_dispatch,
)
from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    WorkflowStatusEventEnum,
)
from duo_workflow_service.entities import WorkflowStatusEnum
from duo_workflow_service.errors.typing import WorkflowAlreadyFinishedException
from duo_workflow_service.gitlab.gitlab_api import WorkflowConfig
from duo_workflow_service.status_updater.gitlab_status_updater import (
    UnsupportedStatusEvent,
)
from lib.internal_events.event_enum import EventPropertyEnum

_CHECKPOINT = {"checkpoint": "checkpoint_data"}


def _context(
    workflow_status: WorkflowStatusEnum,
    latest_checkpoint: Optional[dict[str, Any]] = None,
    first_checkpoint: Optional[dict[str, Any]] = None,
    logger: Optional[MagicMock] = None,
) -> EntryDispatchContext:
    # Only the keys rails_status_dispatch reads; the full WorkflowConfig
    # TypedDict is irrelevant to the strategy contract.
    workflow_config = cast(
        WorkflowConfig,
        {
            "workflow_status": workflow_status,
            "latest_checkpoint": latest_checkpoint,
            "first_checkpoint": first_checkpoint,
        },
    )
    return EntryDispatchContext(
        workflow_config=workflow_config,
        logger=logger if logger is not None else MagicMock(),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "ctx,expected_event,expected_property",
    [
        (
            _context(WorkflowStatusEnum.CREATED),
            WorkflowStatusEventEnum.START,
            EventPropertyEnum.WORKFLOW_ID,
        ),
        (
            _context(WorkflowStatusEnum.INPUT_REQUIRED, latest_checkpoint=_CHECKPOINT),
            WorkflowStatusEventEnum.RESUME,
            EventPropertyEnum.WORKFLOW_RESUME_BY_PLAN_AFTER_INPUT,
        ),
        (
            _context(
                WorkflowStatusEnum.PLAN_APPROVAL_REQUIRED,
                latest_checkpoint=_CHECKPOINT,
            ),
            WorkflowStatusEventEnum.RESUME,
            EventPropertyEnum.WORKFLOW_RESUME_BY_PLAN_AFTER_APPROVAL,
        ),
        (
            _context(
                WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED,
                latest_checkpoint=_CHECKPOINT,
            ),
            WorkflowStatusEventEnum.RESUME,
            EventPropertyEnum.WORKFLOW_RESUME_BY_PLAN,
        ),
        # first_checkpoint backfills a missing latest_checkpoint.
        (
            _context(WorkflowStatusEnum.INPUT_REQUIRED, first_checkpoint=_CHECKPOINT),
            WorkflowStatusEventEnum.RESUME,
            EventPropertyEnum.WORKFLOW_RESUME_BY_PLAN_AFTER_INPUT,
        ),
        (
            _context(WorkflowStatusEnum.STOPPED, latest_checkpoint=_CHECKPOINT),
            WorkflowStatusEventEnum.STOP_RECOVERY,
            EventPropertyEnum.WORKFLOW_RESUME_BY_USER,
        ),
        (
            _context(WorkflowStatusEnum.EXECUTION, latest_checkpoint=_CHECKPOINT),
            WorkflowStatusEventEnum.RETRY,
            EventPropertyEnum.WORKFLOW_RESUME_BY_USER,
        ),
    ],
    ids=[
        "created-starts",
        "input-required-resumes",
        "plan-approval-resumes",
        "tool-call-approval-resumes-with-fallback-property",
        "first-checkpoint-backfill",
        "stopped-detects-stop-recovery",
        "execution-retries",
    ],
)
async def test_rails_status_dispatch(ctx, expected_event, expected_property):
    assert await rails_status_dispatch(ctx) == (expected_event, expected_property)


@pytest.mark.asyncio
async def test_created_with_existing_checkpoint_is_rejected():
    ctx = _context(WorkflowStatusEnum.CREATED, first_checkpoint=_CHECKPOINT)

    with pytest.raises(UnsupportedStatusEvent) as exc_info:
        await rails_status_dispatch(ctx)

    assert "should not have existing checkpoints" in str(exc_info.value)


@pytest.mark.asyncio
async def test_finished_session_is_rejected_without_a_status_event():
    ctx = _context(WorkflowStatusEnum.FINISHED, latest_checkpoint=_CHECKPOINT)

    with pytest.raises(WorkflowAlreadyFinishedException):
        await rails_status_dispatch(ctx)


@pytest.mark.asyncio
async def test_continuous_status_without_checkpoints_logs_integrity_error():
    logger = MagicMock()
    ctx = _context(WorkflowStatusEnum.INPUT_REQUIRED, logger=logger)

    event, _ = await rails_status_dispatch(ctx)

    assert event == WorkflowStatusEventEnum.RESUME
    logger.error.assert_called_once()
