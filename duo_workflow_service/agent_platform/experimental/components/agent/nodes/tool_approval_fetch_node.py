# Re-export ToolApprovalFetchNode from v1 to prevent code duplication.
# The experimental implementation has been promoted to v1.
from duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_approval_fetch_node import (
    ToolApprovalFetchNode,
)

__all__ = ["ToolApprovalFetchNode"]
