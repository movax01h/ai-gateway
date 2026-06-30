# Re-export AgentNode and AgentFinalOutput from v1
# to prevent code duplication. The experimental implementation has been promoted to v1.
from duo_workflow_service.agent_platform.v1.components.agent.nodes.agent_node import (
    AgentFinalOutput,
    AgentNode,
)

__all__ = ["AgentFinalOutput", "AgentNode"]
