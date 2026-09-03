from typing import Optional

from pydantic import BaseModel

__all__ = [
    "AIO_CANCEL_INFRA_STOP_WORKFLOW_REQUEST",
    "AIO_CANCEL_STOP_WORKFLOW_REQUEST",
    "INFRA_STOP_REASONS",
    "MAX_MESSAGE_SIZE",
    "OUTGOING_MESSAGE_TOO_LARGE",
    "AdditionalContext",
    "OsInformationContext",
    "ShellInformationContext",
]

# Mirrored into the server's grpc.max_send_message_length / max_receive_message_length
# options and used by payload producers to size themselves against the real limit.
# Lives here rather than server.py so checkpointer.notifier can import it without a cycle.
MAX_MESSAGE_SIZE = 4 * 1024 * 1024

AIO_CANCEL_STOP_WORKFLOW_REQUEST = "AIO_CANCEL_STOP_WORKFLOW_REQUEST"
# Distinct cancellation message for infrastructure-initiated stops (e.g. Workhorse pod
# rotation, WebSocket ping failures).  Using a separate constant — rather than reading
# ``MonitoringContext.workflow_stop_reason`` in ``__aexit__`` — keeps the stop-type
# discrimination self-contained in the cancellation signal and avoids overloading
# ``MonitoringContext`` with a new responsibility.
AIO_CANCEL_INFRA_STOP_WORKFLOW_REQUEST = "AIO_CANCEL_INFRA_STOP_WORKFLOW_REQUEST"
OUTGOING_MESSAGE_TOO_LARGE = "OUTGOING_MESSAGE_TOO_LARGE"

# Stop reasons that originate from infrastructure events (e.g. Workhorse pod rotation,
# WebSocket ping failures) rather than an explicit user action.  When DWS receives a
# stopWorkflow request carrying one of these reasons it must NOT persist a `stopped`
# status to Rails — the session should remain `running` so that the client's automatic
# reconnect is classified as a plain RETRY (LangGraph replay from the last checkpoint)
# rather than a STOP_RECOVERY (which would attempt a rollback or restart with an empty
# goal).  Sessions that never reconnect are eventually reaped by Rails'
# FailStuckWorkflowsWorker and land as `failed`, which is the accurate terminal state
# for an infra interruption that was never recovered.
INFRA_STOP_REASONS = frozenset(
    {
        "WORKHORSE_SERVER_SHUTDOWN",
        "WORKHORSE_WEBSOCKET_PING_FAILED",
    }
)


# Note: additional_context is an alias for injected_context
class AdditionalContext(BaseModel):
    # One of "file", "snippet", "merge_request", "issue", "dependency", "local_git", "terminal", "repository",
    # "directory". The corresponding unit primitives must be registered with `include_{category}_context` format.
    # https://gitlab.com/gitlab-org/cloud-connector/gitlab-cloud-connector/-/tree/main/config/unit_primitives
    category: str
    id: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[dict] = None


class OsInformationContext(BaseModel):
    platform: str
    architecture: str


class ShellInformationContext(BaseModel):
    shell_name: str
    shell_type: str  # 'unix' | 'windows' | 'hybrid'
    shell_variant: Optional[str] = None
    shell_environment: Optional[str] = (
        None  # 'native' | 'wsl' | 'git-bash' | 'cygwin' | 'mingw' | 'ssh' | 'docker'
    )
    ssh_session: Optional[bool] = None
    cwd: Optional[str] = None
