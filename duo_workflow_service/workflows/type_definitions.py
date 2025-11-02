from typing import Optional

from pydantic import BaseModel

__all__ = [
    "AdditionalContext",
    "OsInformationContext",
    "ShellInformationContext",
    "AIO_CANCEL_STOP_WORKFLOW_REQUEST",
    "OUTGOING_MESSAGE_TOO_LARGE",
]

AIO_CANCEL_STOP_WORKFLOW_REQUEST = "AIO_CANCEL_STOP_WORKFLOW_REQUEST"
OUTGOING_MESSAGE_TOO_LARGE = "OUTGOING_MESSAGE_TOO_LARGE"


# Note: additionaL_context is an alias for injected_context
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
