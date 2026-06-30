# Re-export FinalResponseNode from v1
# to prevent code duplication. The experimental implementation has been promoted to v1.
from duo_workflow_service.agent_platform.v1.components.agent.nodes.final_response_node import (
    FinalResponseNode,
)

__all__ = ["FinalResponseNode"]
