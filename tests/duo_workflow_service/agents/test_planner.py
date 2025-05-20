# pylint: disable=file-naming-for-tests

import pytest
from langchain_core.messages import HumanMessage

from duo_workflow_service.agents import PlanSupervisorAgent
from duo_workflow_service.agents.prompts import NEXT_STEP_PROMPT
from duo_workflow_service.entities.state import Plan, WorkflowState, WorkflowStatusEnum


class TestPlanSupervisorAgent:
    @pytest.fixture
    def workflow_state(self) -> WorkflowState:
        return WorkflowState(
            plan=Plan(steps=[]),
            status=WorkflowStatusEnum.NOT_STARTED,
            conversation_history={},
            handover=[],
            last_human_input=None,
            ui_chat_log=[],
        )

    @pytest.mark.asyncio
    async def test_run(self, workflow_state: WorkflowState):
        supervisor_agent = PlanSupervisorAgent(supervised_agent_name="executor")

        result = await supervisor_agent.run(workflow_state)

        assert result == {
            "conversation_history": {
                "executor": [HumanMessage(content=NEXT_STEP_PROMPT)]
            }
        }
