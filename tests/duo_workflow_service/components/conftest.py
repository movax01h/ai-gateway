import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from langchain_core.messages import AIMessage


@pytest.fixture(name="graph_config")
def graph_config_fixture():
    return {"configurable": {"thread_id": "test-workflow"}}


@pytest.fixture(name="end_message")
def end_message_fixture():
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


@pytest.fixture(name="user")
def user_fixture():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            scopes=["duo_workflow_execute_workflow"],
            issuer="gitlab-duo-workflow-service",
        ),
    )
