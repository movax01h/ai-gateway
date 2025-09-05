import pytest
from langchain_core.messages import HumanMessage

from duo_workflow_service.agents import PlanSupervisorAgent
from duo_workflow_service.entities.state import Plan, WorkflowState, WorkflowStatusEnum
from duo_workflow_service.tools.handover import HandoverTool


class TestPlanSupervisorAgent:
    @pytest.fixture(name="workflow_state")
    def workflow_state_fixture(self) -> WorkflowState:
        return WorkflowState(
            plan=Plan(steps=[]),
            status=WorkflowStatusEnum.NOT_STARTED,
            conversation_history={},
            handover=[],
            last_human_input=None,
            ui_chat_log=[],
            project=None,
            goal=None,
            additional_context=None,
        )

    @pytest.mark.asyncio
    async def test_run(self, workflow_state: WorkflowState):
        supervisor_agent = PlanSupervisorAgent(supervised_agent_name="executor")

        result = await supervisor_agent.run(workflow_state)

        assert result == {
            "conversation_history": {
                "executor": [
                    HumanMessage(
                        content=f"What is the next task? Call the `{HandoverTool.tool_title}` tool if your task is "
                        "complete"
                    )
                ]
            }
        }
