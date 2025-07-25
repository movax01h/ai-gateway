import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from langchain_core.messages import AIMessage

from duo_workflow_service.entities.state import Plan, WorkflowState, WorkflowStatusEnum
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture
def config_values():
    return {"mock_model_responses": True}


@pytest.fixture(scope="function")
def graph_input() -> WorkflowState:
    return WorkflowState(
        status=WorkflowStatusEnum.NOT_STARTED,
        conversation_history={},
        last_human_input=None,
        handover=[],
        ui_chat_log=[],
        plan=Plan(steps=[]),
        project=None,
        goal=None,
        additional_context=None,
    )


@pytest.fixture
def graph_config():
    return {"configurable": {"thread_id": "test-workflow"}}


@pytest.fixture
def end_message():
    return AIMessage(
        content="Done with the execution, over to handover agent",
        tool_calls=[
            {
                "id": "1",
                "name": "handover_tool",
                "args": {"summary": "done"},
            }
        ],
    )


@pytest.fixture
def workflow_type() -> CategoryEnum:
    return CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT


@pytest.fixture
def user():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            scopes=["duo_workflow_execute_workflow"],
            issuer="gitlab-duo-workflow-service",
        ),
    )
