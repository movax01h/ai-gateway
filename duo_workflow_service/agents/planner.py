from typing import Dict, List

from langchain_core.messages import BaseMessage, HumanMessage

from duo_workflow_service.agents.prompts import NEXT_STEP_PROMPT
from duo_workflow_service.entities.state import WorkflowState


class PlanSupervisorAgent:
    _supervised_agent_name: str

    def __init__(self, supervised_agent_name: str):
        self._supervised_agent_name = supervised_agent_name

    async def run(
        self, _state: WorkflowState
    ) -> Dict[str, Dict[str, List[BaseMessage]]]:
        return {
            "conversation_history": {
                self._supervised_agent_name: [HumanMessage(content=NEXT_STEP_PROMPT)]
            }
        }
