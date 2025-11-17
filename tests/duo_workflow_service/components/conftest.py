import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims


@pytest.fixture(name="graph_config")
def graph_config_fixture():
    return {"configurable": {"thread_id": "test-workflow"}}


@pytest.fixture(name="user")
def user_fixture():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            scopes=["duo_workflow_execute_workflow"],
            issuer="gitlab-duo-workflow-service",
        ),
    )
