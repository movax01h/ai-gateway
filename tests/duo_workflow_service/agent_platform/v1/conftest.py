from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from langgraph.graph import StateGraph

from duo_workflow_service.agent_platform.v1.components import ComponentRegistry
from duo_workflow_service.checkpointer.gitlab_workflow import GitLabWorkflow
from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    WorkflowStatusEventEnum,
)
from duo_workflow_service.checkpointer.notifier import UserInterface


@pytest.fixture
def component_registry_instance_type():
    # Mock component registry 'instance' to always return a fresh object instead of the singleton
    with patch(
        "duo_workflow_service.agent_platform.v1.components.ComponentRegistry.instance"
    ) as mock:
        mock.return_value = ComponentRegistry(force_new=True)
        yield mock


@pytest.fixture(name="mock_state_graph")
def mock_state_graph_fixture(
    mock_fetch_workflow_and_container_data,  # pylint: disable=unused-argument  # fixture-on-fixture ordering dep
):
    """A `StateGraph` whose compiled graph records what `AbstractWorkflow.run` streams from."""
    mock_state_graph = Mock(spec=StateGraph)
    mock_state_graph.nodes = {}
    mock_compiled_graph = Mock()

    # Mock the compiled graph's astream method
    async def mock_astream(*_args, **_kwargs):
        yield ("values", {"status": "running"})
        yield ("updates", [{"step": "agent_processing"}])

    # mock_compiled_graph.astream = AsyncMock(side_effect=mock_astream)
    mock_compiled_graph.astream = Mock(return_value=mock_astream())
    mock_state_graph.compile.return_value = mock_compiled_graph
    ui_notifier = MagicMock(spec=UserInterface)
    ui_notifier.ui_chat_log = []

    with (
        patch(
            "duo_workflow_service.workflows.abstract_workflow.empty_workflow_config",
            return_value={
                "agent_privileges_names": [],
                "pre_approved_agent_privileges_names": [],
                "allow_agent_to_request_user": False,
                "mcp_enabled": False,
                "first_checkpoint": None,
                "workflow_status": "",
                "gitlab_host": "",
            },
        ),
        patch(
            "duo_workflow_service.workflows.abstract_workflow.UserInterface",
            return_value=ui_notifier,
        ),
        patch("duo_workflow_service.gitlab.gitlab_api.GitLabUrlParser") as mock_parser,
        patch(
            "duo_workflow_service.agent_platform.v1.flows.graph_builder.StateGraph",
            return_value=mock_state_graph,
        ),
    ):
        mock_parser.extract_host_from_url.return_value = "gitlab.com"

        yield mock_state_graph


@pytest.fixture(name="mock_checkpointer")
def mock_checkpointer_fixture():
    """The `GitLabWorkflow` a run enters, standing in for the session's checkpoints."""
    mock_checkpointer = Mock(spec=GitLabWorkflow)
    mock_checkpointer.initial_status_event = WorkflowStatusEventEnum.START
    mock_checkpointer.aget_tuple = AsyncMock(return_value=None)
    mock_gitlab_workflow = AsyncMock()
    mock_gitlab_workflow.__aenter__ = AsyncMock(return_value=mock_checkpointer)
    mock_gitlab_workflow.__aexit__ = AsyncMock(return_value=None)
    with patch(
        "duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow",
        return_value=mock_gitlab_workflow,
    ):
        yield mock_checkpointer
