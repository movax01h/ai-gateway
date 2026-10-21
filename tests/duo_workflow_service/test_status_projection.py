from unittest.mock import patch

import pytest

from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    WorkflowStatusEventEnum,
)
from duo_workflow_service.entities.state import WorkflowStatusEnum
from duo_workflow_service.status_projection import (
    IDLE_STATUS_CAPABILITY,
    IDLE_STATUS_EVENT_CAPABILITY,
    client_checkpoint_status,
    rails_status_event,
)


@pytest.fixture(name="declare_capabilities")
def declare_capabilities_fixture():
    """Patch ``is_client_capable`` so only the given capability names are declared."""
    with patch(
        "duo_workflow_service.status_projection.is_client_capable"
    ) as is_client_capable:

        def _declare(*names: str):
            is_client_capable.side_effect = lambda capability: capability in names

        _declare()
        yield _declare


def test_capability_names():
    assert IDLE_STATUS_CAPABILITY == "idle_status"
    assert IDLE_STATUS_EVENT_CAPABILITY == "idle_status_event"


@pytest.mark.parametrize(
    ("capabilities", "event", "expected"),
    [
        # IDLE reaches only an instance that accepts the `idle` status event.
        (
            {IDLE_STATUS_EVENT_CAPABILITY},
            WorkflowStatusEventEnum.IDLE,
            WorkflowStatusEventEnum.IDLE,
        ),
        (set(), WorkflowStatusEventEnum.IDLE, WorkflowStatusEventEnum.REQUIRE_INPUT),
        # The client capability alone does not reach Rails.
        (
            {IDLE_STATUS_CAPABILITY},
            WorkflowStatusEventEnum.IDLE,
            WorkflowStatusEventEnum.REQUIRE_INPUT,
        ),
        # STOP_RECOVERY is still sent as RETRY, with or without the capability.
        (
            set(),
            WorkflowStatusEventEnum.STOP_RECOVERY,
            WorkflowStatusEventEnum.RETRY,
        ),
        (
            {IDLE_STATUS_EVENT_CAPABILITY},
            WorkflowStatusEventEnum.STOP_RECOVERY,
            WorkflowStatusEventEnum.RETRY,
        ),
        # Other events pass through unchanged.
        (
            set(),
            WorkflowStatusEventEnum.REQUIRE_INPUT,
            WorkflowStatusEventEnum.REQUIRE_INPUT,
        ),
        (
            {IDLE_STATUS_EVENT_CAPABILITY},
            WorkflowStatusEventEnum.REQUIRE_INPUT,
            WorkflowStatusEventEnum.REQUIRE_INPUT,
        ),
        (set(), WorkflowStatusEventEnum.FINISH, WorkflowStatusEventEnum.FINISH),
        (
            {IDLE_STATUS_EVENT_CAPABILITY},
            WorkflowStatusEventEnum.RESUME,
            WorkflowStatusEventEnum.RESUME,
        ),
    ],
)
def test_rails_status_event(declare_capabilities, capabilities, event, expected):
    declare_capabilities(*capabilities)

    assert rails_status_event(event) is expected


@pytest.mark.parametrize(
    ("capabilities", "status", "expected"),
    [
        # IDLE reaches only a client that renders it.
        ({IDLE_STATUS_CAPABILITY}, WorkflowStatusEnum.IDLE, "IDLE"),
        (set(), WorkflowStatusEnum.IDLE, "INPUT_REQUIRED"),
        # The server capability alone does not reach the client.
        ({IDLE_STATUS_EVENT_CAPABILITY}, WorkflowStatusEnum.IDLE, "INPUT_REQUIRED"),
        # Other statuses map as before, with or without the capability.
        (set(), WorkflowStatusEnum.INPUT_REQUIRED, "INPUT_REQUIRED"),
        ({IDLE_STATUS_CAPABILITY}, WorkflowStatusEnum.INPUT_REQUIRED, "INPUT_REQUIRED"),
        (set(), WorkflowStatusEnum.COMPLETED, "FINISHED"),
        ({IDLE_STATUS_CAPABILITY}, WorkflowStatusEnum.EXECUTION, "RUNNING"),
    ],
)
def test_client_checkpoint_status(declare_capabilities, capabilities, status, expected):
    declare_capabilities(*capabilities)

    assert client_checkpoint_status(status) == expected
