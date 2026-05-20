# Re-export v1 ToolNode to prevent code duplication.
from duo_workflow_service.agent_platform.v1.components.agent.nodes.tool_node import (  # noqa: F401
    ToolNode,
)

__all__ = ["ToolNode"]
