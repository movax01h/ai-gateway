import pytest

from duo_workflow_service.entities.state import Plan, WorkflowState, WorkflowStatusEnum


@pytest.fixture(scope="function")
def graph_input() -> WorkflowState:
    return WorkflowState(
        status=WorkflowStatusEnum.NOT_STARTED,
        conversation_history={},
        last_human_input=None,
        handover=[],
        ui_chat_log=[],
        plan=Plan(steps=[]),
    )


@pytest.fixture
def graph_config():
    return {"configurable": {"thread_id": "test-workflow"}}
