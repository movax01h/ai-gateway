"""How an engine status is shown to each consumer.

The engine keeps one vocabulary (``WorkflowStatusEnum``). A consumer that predates a status sees the nearest status
it knows instead, decided here and nowhere else. This covers the two places a status goes on the wire: the Rails
status event and the streamed checkpoint's ``status``. The saved checkpoint keeps the engine's own status.

``IDLE`` is the first such status: a session at rest after a turn, done for now and taking a next message. The
nearest status older consumers know is ``INPUT_REQUIRED``, which is exactly what they saw before ``IDLE`` existed.

Two consumers, two capabilities, both negotiated through the existing capability list Workhorse forwards.

GitLab advertises ``IDLE_STATUS_EVENT_CAPABILITY`` as a server capability when it accepts the ``idle`` status event.
An instance that does not, or a path that skips Workhorse, gets ``require_input``: an older instance rejects an unknown
status event with a 400 and the session would fail.

The client streaming the session declares ``IDLE_STATUS_CAPABILITY`` when it renders ``IDLE``. Older clients
(gitlab-lsp before 9.31.0) drop a checkpoint whose status they don't know, so the Duo CLI and the IDE would lose the
end of the turn.
"""

from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    INTERNAL_TO_RAILS_STATUS_EVENT,
    WORKFLOW_STATUS_TO_CHECKPOINT_STATUS,
    WorkflowStatusEventEnum,
)
from duo_workflow_service.client_capabilities import is_client_capable
from duo_workflow_service.entities.state import WorkflowStatusEnum

__all__ = [
    "IDLE_STATUS_CAPABILITY",
    "IDLE_STATUS_EVENT_CAPABILITY",
    "client_checkpoint_status",
    "rails_status_event",
]

# Declared by a client that renders the IDLE checkpoint status.
IDLE_STATUS_CAPABILITY = "idle_status"

# Advertised by a GitLab instance that accepts the `idle` status event.
IDLE_STATUS_EVENT_CAPABILITY = "idle_status_event"


def rails_status_event(event: WorkflowStatusEventEnum) -> WorkflowStatusEventEnum:
    """The status event this GitLab instance accepts for ``event``."""
    if event == WorkflowStatusEventEnum.IDLE and not is_client_capable(
        IDLE_STATUS_EVENT_CAPABILITY
    ):
        return WorkflowStatusEventEnum.REQUIRE_INPUT

    return INTERNAL_TO_RAILS_STATUS_EVENT.get(event, event)


def client_checkpoint_status(status: WorkflowStatusEnum) -> str:
    """The checkpoint status the streaming client is sent for the engine ``status``."""
    if status == WorkflowStatusEnum.IDLE and not is_client_capable(
        IDLE_STATUS_CAPABILITY
    ):
        status = WorkflowStatusEnum.INPUT_REQUIRED

    return WORKFLOW_STATUS_TO_CHECKPOINT_STATUS[status]
