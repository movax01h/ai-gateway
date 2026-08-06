# Re-export from v1 to prevent code duplication, as the sibling nodes do.
from duo_workflow_service.agent_platform.v1.components.agent.nodes._session import (
    DEFAULT_SESSION_ID_KEY,
    resolve_session_id,
)

__all__ = ["DEFAULT_SESSION_ID_KEY", "resolve_session_id"]
