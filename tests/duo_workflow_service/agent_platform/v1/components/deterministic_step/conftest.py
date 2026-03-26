"""Shared fixtures for deterministic step component tests."""

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from duo_workflow_service.entities.state import WorkflowStatusEnum


@pytest.fixture(name="user")
def user_fixture():
    """Fixture for CloudConnectorUser."""
    return CloudConnectorUser(
        authenticated=True,
        is_debug=False,
        claims=UserClaims(scopes=[], gitlab_instance_uid="unique-instance-uid"),
    )


@pytest.fixture(name="workflow_state")
def workflow_state_fixture():
    """Fixture for workflow state."""
    return {
        "status": WorkflowStatusEnum.EXECUTION,
        "conversation_history": {},
        "ui_chat_log": [],
        "context": {"input1": "value1", "input2": "value2"},
    }
