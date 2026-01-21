from typing import Dict, List

from langchain_core.messages import BaseMessage, HumanMessage

from duo_workflow_service.entities.state import WorkflowState
from duo_workflow_service.tools.handover import HandoverTool


class PlanSupervisorAgent:
    _supervised_agent_name: str

    def __init__(self, supervised_agent_name: str):
        self._supervised_agent_name = supervised_agent_name

    async def run(
        self, state: WorkflowState  # pylint: disable=unused-argument
    ) -> Dict[str, Dict[str, List[BaseMessage]]]:
        return {
            "conversation_history": {
                self._supervised_agent_name: [
                    HumanMessage(
                        content=f"What is the next task? Call the `{HandoverTool.tool_title}` tool if your task is "
                        "complete"
                    )
                ]
            }
        }
