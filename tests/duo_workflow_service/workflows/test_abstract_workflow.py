# pylint: disable=direct-environment-variable-reference,too-many-lines
import asyncio
import os
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from contract import contract_pb2
from duo_workflow_service.agent_platform.utils.exceptions import (
    NotifiableAgentException,
)
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    ToolStatus,
    WorkflowStatusEnum,
)
from duo_workflow_service.errors.typing import NotifiableException
from duo_workflow_service.tools import UNTRUSTED_MCP_WARNING
from duo_workflow_service.tracking import (
    MonitoringContext,
    current_monitoring_context,
)
from duo_workflow_service.workflows.abstract_workflow import (
    AbstractWorkflow,
    TraceableException,
)
from duo_workflow_service.workflows.type_definitions import (
    AIO_CANCEL_STOP_WORKFLOW_REQUEST,
)
from lib.internal_events import InternalEventAdditionalProperties
from lib.internal_events.event_enum import CategoryEnum, EventEnum
from lib.langsmith_tracing import set_langsmith_trace_headers


# Concrete implementation for testing
class MockGraph:
    async def astream(  # pylint: disable=unused-argument  # astream() signature
        self, input, config, stream_mode
    ):
        yield "updates", {"step1": {"key": "value"}}


class MockGraphWithUiChatLog:
    async def astream(  # pylint: disable=unused-argument  # astream() signature
        self, input, config, stream_mode
    ):
        yield (
            "values",
            {
                "status": WorkflowStatusEnum.COMPLETED,
                "ui_chat_log": [
                    {"content": "First message"},
                    {"content": "Second message"},
                    {"content": "Final response"},
                ],
            },
        )


class MockWorkflow(AbstractWorkflow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.handle_failure_calls: list[tuple[BaseException, Any, Any]] = []

    def _compile(self, goal, tools_registry, checkpointer):
        return MockGraph()

    def get_workflow_state(self, goal):
        return {"goal": goal, "state": "initial"}

    async def _handle_workflow_failure(self, error, compiled_graph, graph_config):
        self.handle_failure_calls.append((error, compiled_graph, graph_config))

    def log_workflow_elements(self, element):
        print(element)


@pytest.fixture(autouse=True)
def prepare_container(  # pylint: disable=unused-argument  # fixture-on-fixture ordering dep
    mock_duo_workflow_service_container,
):
    # ``current_monitoring_context`` defaults to a single shared, mutable ``MonitoringContext``
    # instance. Any test (or production code path exercised by a test) that mutates it via
    # ``set_flow_identity`` would otherwise leak flow versioning fields (flow_id / flow_version /
    # schema_version) into later tests on the same worker, breaking order-dependent assertions
    # such as "legacy flows carry no versioning metadata". Give each test a fresh context.
    token = current_monitoring_context.set(MonitoringContext())
    try:
        yield
    finally:
        current_monitoring_context.reset(token)


def test_extract_trace_output_with_valid_state(user):
    workflow = MockWorkflow(
        "test-id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
    )
    state = {
        "ui_chat_log": [
            {"content": "First message"},
            {"content": "Second message"},
            {"content": "Final response"},
        ]
    }
    result = workflow._extract_trace_output(state)
    assert result == "Final response"


def test_extract_trace_output_with_empty_ui_chat_log(user):
    workflow = MockWorkflow(
        "test-id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
    )
    state = {"ui_chat_log": []}
    result = workflow._extract_trace_output(state)
    assert result is None


def test_extract_trace_output_with_none_state(user):
    workflow = MockWorkflow(
        "test-id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
    )
    result = workflow._extract_trace_output(None)
    assert result is None


@pytest.fixture(name="workflow")
def workflow_fixture(user):
    workflow_id = "test-workflow-id"
    metadata = {
        "extended_logging": True,
        "git_url": "https://example.com",
        "git_sha": "abc123",
    }
    workflow_type = CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT
    return MockWorkflow(
        workflow_id,
        metadata,
        workflow_type,
        user,
    )


@pytest.fixture(name="langsmith_trace_headers")
def langsmith_trace_headers_fixture():
    """Realistic LangSmith trace headers as stored by the interceptor."""
    return {
        "langsmith-trace": "20260311T180532123456Z-abcd1234-5678-90ef-ghij-klmnopqrstuv",
        "baggage": {
            "langsmith-metadata": {
                "revision_id": "v1.0.0",
                "__ls_runner": "py_sdk_evaluate",
                "num_repetitions": 3,
                "example_version": "2026-03-11T18:05:32.123456+00:00",
                "ls_method": "traceable",
            },
            "langsmith-project": "workflow-tests-42",
        },
    }


@pytest.fixture(name="mock_namespace")
def mock_namespace_fixture():
    return {
        "id": MagicMock(),
        "description": MagicMock(),
        "name": MagicMock(),
        "web_url": "https://example.com/group",
    }


@pytest.fixture(name="mock_action")
def mock_action_fixture():
    return contract_pb2.Action()


@pytest.mark.asyncio
async def test_init(user):
    # Test initialization
    workflow_id = "test-workflow-id"
    metadata = {"key": "value"}
    mcp_tools = [
        contract_pb2.McpTool(name="get_issue", description="Tool to get issue")
    ]
    user = MagicMock()
    workflow = MockWorkflow(
        workflow_id,
        metadata,
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
        mcp_tools,
    )

    assert workflow._workflow_id == workflow_id
    assert workflow._workflow_metadata == metadata
    assert workflow.is_done is False
    assert len(workflow._mcp_tools) == 1
    # Tools are now configs (dicts), not classes
    tool_config = workflow._mcp_tools[0]
    assert tool_config["llm_name"] == "get_issue"
    assert tool_config["description"] == f"{UNTRUSTED_MCP_WARNING}\n\nTool to get issue"
    assert tool_config["original_name"] == "get_issue"
    assert workflow._user == user


@pytest.mark.asyncio
async def test_get_from_outbox(workflow, mock_action):
    # Put an item in the outbox
    workflow._outbox.put_action(mock_action)

    # Get the item
    item = await workflow.get_from_outbox()

    assert item == mock_action
    assert workflow._outbox._queue.empty()


@pytest.mark.asyncio
async def test_fail_outbox_action(workflow, mock_action):
    workflow._outbox.put_action(mock_action)

    item = await workflow.get_from_outbox()
    workflow.fail_outbox_action(item.requestID, "Something went wrong")

    assert item.requestID not in workflow._outbox._action_response


@pytest.mark.asyncio
async def test_set_action_response(workflow):
    request_id = workflow._outbox.put_action(contract_pb2.Action())

    # Set action response that has the corresponding request ID
    workflow.set_action_response(
        contract_pb2.ClientEvent(
            actionResponse=contract_pb2.ActionResponse(
                plainTextResponse=contract_pb2.PlainTextResponse(
                    response="the response"
                ),
                requestID=request_id,
            )
        )
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.convert_mcp_tools_to_configs")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
@pytest.mark.parametrize("mcp_enabled", [True, False])
async def test_compile_and_run_graph(
    mock_tools_registry,
    mock_gitlab_workflow,
    mock_convert_mcp_tools,
    mock_fetch_workflow_and_container_data,
    project,
    mcp_enabled,
    user,
):
    # Setup mocks
    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.aget_tuple = AsyncMock(return_value=None)
    mock_checkpointer.initial_status_event = "START"
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer

    mcp_tool = MagicMock()
    mock_convert_mcp_tools.return_value = [mcp_tool]

    workflow = MockWorkflow(
        "id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
        [mcp_tool],
    )

    # Run the method
    await workflow._compile_and_run_graph("Test goal")

    # Assertions
    assert workflow.is_done
    assert (
        workflow._session_url
        == "https://gitlab.com/test/repo/-/automate/agent-sessions/id"
    )
    mock_fetch_workflow_and_container_data.assert_called_once()
    mock_tools_registry.assert_called_once_with(
        outbox=workflow._outbox,
        workflow_config=workflow._workflow_config,
        gl_http_client=workflow._http_client,
        project=project,
        workflow_id="id",
        mcp_tools=[mcp_tool] if mcp_enabled else [],
        language_server_version=None,
        denied_tools=[],
    )


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
@patch("duo_workflow_service.workflows.abstract_workflow.convert_mcp_tools_to_configs")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
async def test_compile_and_run_graph_merges_jwt_pre_approved_tools(
    mock_tools_registry,
    mock_gitlab_workflow,
    _mock_convert_mcp_tools,
):
    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.initial_status_event = "START"
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer

    user = CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            gitlab_realm="saas",
            extra={
                "tool_access_policies": '{"allow": ["read_file", "create_issue"], "deny": []}'
            },
        ),
    )
    workflow = MockWorkflow(
        "id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
    )
    await workflow._compile_and_run_graph("Test goal")

    assert "read_file" in workflow._preapproved_tools
    assert "create_issue" in workflow._preapproved_tools


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
@patch("duo_workflow_service.workflows.abstract_workflow.convert_mcp_tools_to_configs")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
async def test_compile_and_run_graph_skips_merge_when_no_jwt_pre_approved_tools(
    mock_tools_registry,
    mock_gitlab_workflow,
    _mock_convert_mcp_tools,
):
    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.initial_status_event = "START"
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer

    user = CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(gitlab_realm="saas", extra={}),  # no pre_approved_tools claim
    )

    workflow = MockWorkflow(
        "id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
    )

    await workflow._compile_and_run_graph("Test goal")

    assert not workflow._preapproved_tools


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
@patch("duo_workflow_service.workflows.abstract_workflow.convert_mcp_tools_to_configs")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
async def test_compile_and_run_graph_skips_merge_when_pre_approved_tools_invalid_json(
    mock_tools_registry,
    mock_gitlab_workflow,
    _mock_convert_mcp_tools,
):
    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.initial_status_event = "START"
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer

    user = CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            gitlab_realm="saas", extra={"tool_access_policies": "not_valid_json"}
        ),
    )
    workflow = MockWorkflow(
        "id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
    )
    await workflow._compile_and_run_graph("Test goal")

    assert not workflow._preapproved_tools


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
@patch("duo_workflow_service.workflows.abstract_workflow.convert_mcp_tools_to_configs")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
async def test_compile_and_run_graph_merges_with_existing_preapproved_tools(
    mock_tools_registry,
    mock_gitlab_workflow,
    _mock_convert_mcp_tools,
):
    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.initial_status_event = "START"
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer

    user = CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            gitlab_realm="saas",
            extra={"tool_access_policies": '{"allow": ["create_issue"], "deny": []}'},
        ),
    )
    workflow = MockWorkflow(
        "id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
    )
    workflow._preapproved_tools = ["read_file"]
    await workflow._compile_and_run_graph("Test goal")

    assert "read_file" in workflow._preapproved_tools
    assert "create_issue" in workflow._preapproved_tools


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
@patch("duo_workflow_service.workflows.abstract_workflow.convert_mcp_tools_to_configs")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
async def test_compile_and_run_graph_skips_merge_when_pre_approved_tools_empty_json_array(
    mock_tools_registry,
    mock_gitlab_workflow,
    _mock_convert_mcp_tools,
):
    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.initial_status_event = "START"
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer

    user = CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            gitlab_realm="saas",
            extra={"tool_access_policies": '{"allow": [], "deny": []}'},
        ),
    )
    workflow = MockWorkflow(
        "id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
    )
    await workflow._compile_and_run_graph("Test goal")

    assert not workflow._preapproved_tools


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
@patch("duo_workflow_service.workflows.abstract_workflow.convert_mcp_tools_to_configs")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
async def test_compile_and_run_graph_skips_merge_when_claims_is_none(
    mock_tools_registry,
    mock_gitlab_workflow,
    _mock_convert_mcp_tools,
):
    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.initial_status_event = "START"
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer

    user = CloudConnectorUser(authenticated=True, claims=None)

    workflow = MockWorkflow(
        "id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
    )

    await workflow._compile_and_run_graph("Test goal")

    assert not workflow._preapproved_tools


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.log_exception")
async def test_cleanup(mock_log_exception, workflow):
    # Add items to queues
    workflow._outbox.put_action(contract_pb2.Action())

    # Run cleanup
    await workflow.cleanup(workflow._workflow_id)

    # Check queues are empty
    assert workflow._outbox._queue.empty()
    assert workflow.is_done is True
    mock_log_exception.assert_not_called()


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.log_exception")
async def test_cleanup_with_exception(mock_log_exception, workflow):
    workflow._outbox.put_action(contract_pb2.Action())
    # Make drain_queue raise an exception
    with patch.object(
        workflow._outbox._queue, "get_nowait", side_effect=RuntimeError("Test error")
    ):
        # Catch exception raised during cleanup
        try:
            await workflow.cleanup(workflow._workflow_id)
        except RuntimeError:
            pass

    # Two log_exception calls: one from _drain_queue and one from cleanup
    assert mock_log_exception.call_count == 1

    # Check the first call (from cleanup)
    first_call = mock_log_exception.call_args_list[0]
    args, kwargs = first_call
    assert isinstance(args[0], RuntimeError)
    assert kwargs["extra"]["workflow_id"] == workflow._workflow_id
    assert kwargs["extra"]["context"] == "Workflow cleanup failed"


def test_track_internal_event(workflow, internal_event_client: Mock):
    # Test tracking an internal event
    event_name = EventEnum.WORKFLOW_START
    workflow_type = CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT
    additional_properties = InternalEventAdditionalProperties()
    workflow._internal_event_client = internal_event_client

    workflow._track_internal_event(
        event_name=event_name,
        additional_properties=additional_properties,
        category=workflow_type,
    )

    internal_event_client.track_event.assert_called_once_with(
        event_name=event_name.value,
        additional_properties=additional_properties,
        category=workflow_type,
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
async def test_compile_and_run_graph_with_exception(
    mock_tools_registry,
    _mock_gitlab_workflow,
    mock_fetch_workflow_and_container_data,
    workflow,
):
    # Setup mocks to raise an exception
    mock_tools_registry.side_effect = Exception("Test exception")

    with pytest.raises(TraceableException) as exc_info:
        await workflow._compile_and_run_graph("Test goal")

    mock_tools_registry.assert_called_once()
    mock_fetch_workflow_and_container_data.assert_called_once()
    assert workflow.is_done
    assert isinstance(exc_info.value.original_exception, Exception)
    assert str(exc_info.value.original_exception) == "Test exception"


@pytest.mark.asyncio
@patch("duo_workflow_service.workflows.abstract_workflow.UserInterface")
async def test_compile_and_run_graph_with_cancellation_during_fetch(
    mock_user_interface,
    mock_fetch_workflow_and_container_data,
    workflow,
):
    mock_fetch_workflow_and_container_data.side_effect = asyncio.CancelledError(
        AIO_CANCEL_STOP_WORKFLOW_REQUEST
    )

    mock_notifier = AsyncMock()
    mock_user_interface.return_value = mock_notifier

    with pytest.raises(TraceableException) as exc_info:
        await workflow._compile_and_run_graph("Test goal")

    mock_fetch_workflow_and_container_data.assert_called_once()
    assert workflow.is_done
    assert isinstance(exc_info.value.original_exception, asyncio.CancelledError)
    assert str(exc_info.value.original_exception) == AIO_CANCEL_STOP_WORKFLOW_REQUEST

    mock_notifier.send_event.assert_called_once_with(
        type="values",
        state={"status": WorkflowStatusEnum.CANCELLED, "ui_chat_log": []},
        stream=workflow._stream,
    )


@pytest.mark.asyncio
# pylint: disable=direct-environment-variable-reference
@patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"})
# pylint: enable=direct-environment-variable-reference
@patch.object(MockWorkflow, "_compile_and_run_graph")
async def test_run_passes_correct_metadata_to_langsmith_extra(
    mock_compile_and_run_graph, workflow
):
    await workflow.run("Test goal")

    call_args = mock_compile_and_run_graph.call_args
    _args, kwargs = call_args

    metadata = kwargs["langsmith_extra"]["metadata"]
    assert metadata["git_url"] == "https://example.com"
    assert metadata["git_sha"] == "abc123"
    assert metadata["workflow_type"] == CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT.value
    assert metadata["thread_id"] == workflow._workflow_id
    # Flow versioning identifiers are unset for legacy flows and must not leak in.
    assert "flow_id" not in metadata
    assert "flow_version" not in metadata
    assert "schema_version" not in metadata


@pytest.mark.asyncio
# pylint: disable=direct-environment-variable-reference
@patch.dict(os.environ, {"LANGCHAIN_TRACING_V2": "true"})
# pylint: enable=direct-environment-variable-reference
@patch.object(MockWorkflow, "_compile_and_run_graph")
async def test_run_passes_flow_versioning_metadata_to_langsmith_extra(
    mock_compile_and_run_graph, workflow
):
    token = current_monitoring_context.set(
        MonitoringContext(
            flow_id="developer",
            flow_version="1.2.3",
            schema_version="v1",
        )
    )
    try:
        await workflow.run("Test goal")
    finally:
        current_monitoring_context.reset(token)

    _, kwargs = mock_compile_and_run_graph.call_args
    metadata = kwargs["langsmith_extra"]["metadata"]
    assert metadata["flow_id"] == "developer"
    assert metadata["flow_version"] == "1.2.3"
    assert metadata["schema_version"] == "v1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "project,namespace",
    [
        (
            None,
            {
                "id": 123,
                "name": "test-namespace",
                "description": "This is a test namespace",
                "web_url": "https://gitlab.com/test",
            },
        )
    ],
)
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
async def test_namespace_level_workflow(
    mock_tools_registry,
    mock_gitlab_workflow,
    workflow,
):
    # Setup mocks
    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.aget_tuple = AsyncMock(return_value=None)
    mock_checkpointer.initial_status_event = "START"
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer

    # Run the method
    with pytest.raises(TraceableException) as exc_info:
        await workflow._compile_and_run_graph("Test goal")

    # Assertions
    assert workflow.is_done
    assert workflow._session_url is None
    assert isinstance(exc_info.value.original_exception, Exception)
    assert (
        str(exc_info.value.original_exception)
        == "This feature is only available at the project level. Please try again from within a specific project."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("env_var_value", "extended_logging", "expected_tracing_enabled"),
    [
        ("false", True, False),  # Explicitly disabled
        ("false", False, False),  # Explicitly disabled regardless of extended_logging
        ("true", True, True),  # Explicitly enabled with extended_logging
        ("true", False, False),  # Explicitly enabled but extended_logging is False
        ("", True, True),  # Not set, follows extended_logging
        ("", False, False),  # Not set, follows extended_logging
    ],
)
@patch("duo_workflow_service.workflows.abstract_workflow.tracing_context")
@patch.object(MockWorkflow, "_compile_and_run_graph")
async def test_tracing_enabled_based_on_env_and_extended_logging(
    _mock_compile_and_run_graph,
    mock_tracing_context,
    env_var_value,
    extended_logging,
    expected_tracing_enabled,
    user,
):
    """Test that tracing is enabled/disabled based on LANGSMITH_TRACING_V2 and extended_logging."""
    workflow_id = "test-workflow-id"
    metadata = {
        "extended_logging": extended_logging,
        "git_url": "https://example.com",
        "git_sha": "abc123",
    }
    workflow_type = CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT
    workflow = MockWorkflow(workflow_id, metadata, workflow_type, user)

    with patch.dict(os.environ, {"LANGSMITH_TRACING_V2": env_var_value}, clear=False):
        await workflow.run("Test goal")

    # Verify tracing_context was called with the expected enabled value
    mock_tracing_context.assert_called_once()
    call_kwargs = mock_tracing_context.call_args[1]
    assert call_kwargs["enabled"] == expected_tracing_enabled


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "extended_logging,langsmith_tracing_v2,has_parent_trace,expected_enabled",
    [
        (False, "", True, False),
        (True, "", True, True),
        (True, "", False, True),
        (False, "false", True, False),
    ],
    ids=[
        "parent_trace_present_extended_logging_false",
        "parent_trace_present_extended_logging_true",
        "no_parent_trace_extended_logging_true",
        "langsmith_tracing_v2_false_with_parent_trace",
    ],
)
@patch("duo_workflow_service.workflows.abstract_workflow.tracing_context")
@patch.object(MockWorkflow, "_compile_and_run_graph")
async def test_tracing_context_with_parent_trace_headers(
    _mock_compile_and_run_graph,
    mock_tracing_context,
    user,
    langsmith_trace_headers,
    extended_logging,
    langsmith_tracing_v2,
    has_parent_trace,
    expected_enabled,
):
    """Test tracing_context is called with correct enabled and parent values."""
    parent_trace = langsmith_trace_headers if has_parent_trace else None
    set_langsmith_trace_headers(parent_trace)

    workflow_id = "test-workflow-id"
    metadata = {
        "extended_logging": extended_logging,
        "git_url": "https://example.com",
        "git_sha": "abc123",
    }
    workflow_type = CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT
    workflow = MockWorkflow(workflow_id, metadata, workflow_type, user)

    with patch.dict(
        os.environ, {"LANGSMITH_TRACING_V2": langsmith_tracing_v2}, clear=False
    ):
        await workflow.run("Test goal")

    mock_tracing_context.assert_called_once()
    call_kwargs = mock_tracing_context.call_args[1]
    assert call_kwargs["enabled"] is expected_enabled
    assert call_kwargs["parent"] == parent_trace

    set_langsmith_trace_headers(None)


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
@patch("duo_workflow_service.workflows.abstract_workflow.duo_workflow_metrics")
async def test_compile_and_run_graph_records_first_response_on_first_graph_update(
    mock_metrics,
    mock_tools_registry,
    mock_gitlab_workflow,
    user,
):
    """Test that _compile_and_run_graph records time to first response when graph is updated."""

    class MockGraphWithMessages:
        async def astream(  # pylint: disable=unused-argument  # astream() signature
            self, input, config, stream_mode
        ):
            yield "updates", {"step1": {"key": "value"}}
            yield "values", {"status": "running", "ui_chat_log": []}
            yield "messages", {"message": "second message"}

    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.aget_tuple = AsyncMock(return_value=None)
    mock_checkpointer.initial_status_event = "START"
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer

    workflow = MockWorkflow(
        "id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
    )

    workflow._compile = MagicMock(return_value=MockGraphWithMessages())

    await workflow._compile_and_run_graph("Test goal")

    assert mock_metrics.record_time_to_first_response.call_count == 1
    assert workflow._first_response_metric_recorded is True


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
async def test_compile_and_run_graph_returns_final_response_content(
    mock_tools_registry,
    mock_gitlab_workflow,
    user,
):
    """Test that _compile_and_run_graph returns the final response content from ui_chat_log."""
    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.aget_tuple = AsyncMock(return_value=None)
    mock_checkpointer.initial_status_event = "START"
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer

    workflow = MockWorkflow(
        "id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
    )

    workflow._compile = MagicMock(return_value=MockGraphWithUiChatLog())

    result = await workflow._compile_and_run_graph("Test goal")

    assert result == "Final response"


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
async def test_compile_and_run_graph_returns_none_when_no_ui_chat_log(
    mock_tools_registry,
    mock_gitlab_workflow,
    user,
):
    """Test that _compile_and_run_graph returns None when ui_chat_log is empty."""

    class MockGraphWithEmptyUiChatLog:
        async def astream(  # pylint: disable=unused-argument  # astream() signature
            self, input, config, stream_mode
        ):
            yield "values", {"status": WorkflowStatusEnum.COMPLETED, "ui_chat_log": []}

    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.aget_tuple = AsyncMock(return_value=None)
    mock_checkpointer.initial_status_event = "START"
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer

    workflow = MockWorkflow(
        "id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
    )

    workflow._compile = MagicMock(return_value=MockGraphWithEmptyUiChatLog())

    result = await workflow._compile_and_run_graph("Test goal")

    assert result is None


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
@patch("duo_workflow_service.workflows.abstract_workflow.convert_mcp_tools_to_configs")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
@patch("duo_workflow_service.workflows.abstract_workflow.UserInterface")
async def test_compile_and_run_graph_notifiable_exception_handling(
    mock_user_interface,
    mock_tools_registry,
    mock_gitlab_workflow,
    mock_convert_mcp_tools,
    user,
):
    """Test that NotifiableException is caught and properly converted to error state with UI notification."""

    original_error = RuntimeError("Original error details")

    class MockGraphWithNotifiableException:
        async def astream(  # pylint: disable=unused-argument  # astream() signature
            self, input, config, stream_mode
        ):
            yield "updates", {"step1": {"key": "value"}}
            raise NotifiableException(
                "Custom error message for user"
            ) from original_error

    mock_notifier = AsyncMock()
    mock_notifier.send_event = AsyncMock()
    mock_user_interface.return_value = mock_notifier

    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.aget_tuple = AsyncMock(return_value=None)
    mock_checkpointer.initial_status_event = "START"
    mock_checkpointer.send_event = AsyncMock()
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer

    mcp_tool = MagicMock()
    mock_convert_mcp_tools.return_value = [mcp_tool]

    workflow = MockWorkflow(
        "test-workflow-id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
        [mcp_tool],
    )

    workflow._compile = MagicMock(return_value=MockGraphWithNotifiableException())

    with pytest.raises(TraceableException) as exc_info:
        await workflow._compile_and_run_graph("Test goal")

    exc = exc_info.value
    assert isinstance(exc.original_exception, NotifiableException)
    assert str(exc.original_exception) == "Custom error message for user"
    assert workflow.last_error is original_error

    # NotifiableException keeps its existing semantics: _handle_workflow_failure
    # must NOT be called for it because the abstract layer already surfaces
    # str(e) directly in the UI.
    assert not workflow.handle_failure_calls

    mock_notifier.send_event.assert_called_once()
    call_args = mock_notifier.send_event.call_args
    assert call_args.kwargs["type"] == "values"
    state = call_args.kwargs["state"]
    assert state["status"] == WorkflowStatusEnum.ERROR
    assert len(state["ui_chat_log"]) == 1
    assert state["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
    assert state["ui_chat_log"][0]["content"] == "Custom error message for user"
    assert state["ui_chat_log"][0]["status"] == ToolStatus.FAILURE


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
@patch("duo_workflow_service.workflows.abstract_workflow.convert_mcp_tools_to_configs")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
async def test_compile_and_run_graph_parses_tool_access_policies_object_format(
    mock_tools_registry,
    mock_gitlab_workflow,
    _mock_convert_mcp_tools,
):
    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.initial_status_event = "START"
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer

    user = CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            gitlab_realm="saas",
            extra={
                "tool_access_policies": '{"allow": ["create_issue"], "deny": ["create_merge_request"]}'
            },
        ),
    )
    workflow = MockWorkflow("id", {}, CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT, user)
    await workflow._compile_and_run_graph("Test goal")

    assert "create_issue" in workflow._preapproved_tools
    assert "create_merge_request" in workflow._denied_tools
    assert "create_merge_request" not in workflow._preapproved_tools


@pytest.mark.asyncio
@pytest.mark.usefixtures("mock_fetch_workflow_and_container_data")
@patch("duo_workflow_service.workflows.abstract_workflow.convert_mcp_tools_to_configs")
@patch("duo_workflow_service.workflows.abstract_workflow.GitLabWorkflow")
@patch("duo_workflow_service.workflows.abstract_workflow.ToolsRegistry.configure")
@patch("duo_workflow_service.workflows.abstract_workflow.UserInterface")
async def test_compile_and_run_graph_notifiable_agent_exception_handling(
    mock_user_interface,
    mock_tools_registry,
    mock_gitlab_workflow,
    mock_convert_mcp_tools,
    user,
):
    """Test that NotifiableAgentException surfaces ui_message and logs internal_detail server-side."""

    original_error = RuntimeError("Original error details")
    secret = "secret-token-xyz"

    class MockGraphWithNotifiableAgentException:
        async def astream(  # pylint: disable=unused-argument  # astream() signature
            self, input, config, stream_mode
        ):
            yield "updates", {"step1": {"key": "value"}}
            raise NotifiableAgentException(
                "Safe message for user", internal_detail=secret
            ) from original_error

    mock_notifier = AsyncMock()
    mock_notifier.send_event = AsyncMock()
    mock_user_interface.return_value = mock_notifier

    mock_tools_registry.return_value = MagicMock()
    mock_checkpointer = AsyncMock()
    mock_checkpointer.aget_tuple = AsyncMock(return_value=None)
    mock_checkpointer.initial_status_event = "START"
    mock_checkpointer.send_event = AsyncMock()
    mock_gitlab_workflow.return_value.__aenter__.return_value = mock_checkpointer

    mcp_tool = MagicMock()
    mock_convert_mcp_tools.return_value = [mcp_tool]

    workflow = MockWorkflow(
        "test-workflow-id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
        [mcp_tool],
    )

    workflow._compile = MagicMock(return_value=MockGraphWithNotifiableAgentException())

    with pytest.raises(TraceableException) as exc_info:
        await workflow._compile_and_run_graph("Test goal")

    exc = exc_info.value
    assert isinstance(exc.original_exception, NotifiableAgentException)
    assert workflow.last_error is original_error

    # _handle_workflow_failure must be called so subclasses can log the
    # internal_detail server-side and persist the safe ui_message to the
    # UI chat log via their own UI logic.
    assert len(workflow.handle_failure_calls) == 1
    failure_error, _, _ = workflow.handle_failure_calls[0]
    assert isinstance(failure_error, NotifiableAgentException)
    assert failure_error.internal_detail == secret

    mock_notifier.send_event.assert_called_once()
    call_args = mock_notifier.send_event.call_args
    assert call_args.kwargs["type"] == "values"
    state = call_args.kwargs["state"]
    assert state["status"] == WorkflowStatusEnum.ERROR
    assert len(state["ui_chat_log"]) == 1
    assert state["ui_chat_log"][0]["message_type"] == MessageTypeEnum.AGENT
    # Only the safe ui_message is surfaced; internal_detail must not leak
    assert state["ui_chat_log"][0]["content"] == "Safe message for user"
    assert secret not in state["ui_chat_log"][0]["content"]
    assert state["ui_chat_log"][0]["status"] == ToolStatus.FAILURE


@pytest.mark.asyncio
async def test_handle_compile_and_run_exception_logs_warning_when_checkpoint_notifier_is_none(
    user,
):
    """Test that a warning is logged when checkpoint_notifier is None during error handling.

    checkpoint_notifier is assigned unconditionally before the try block in _compile_and_run_graph, so this branch is
    theoretically unreachable in normal execution. The defensive guard is tested by calling
    _handle_compile_and_run_exception directly with checkpoint_notifier explicitly set to None.
    """
    workflow = MockWorkflow(
        "test-workflow-id",
        {},
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user,
    )
    # Simulate the theoretically-unreachable state where checkpoint_notifier is None
    workflow.checkpoint_notifier = None

    error = RuntimeError("graph error")

    with patch.object(workflow, "log") as mock_log:
        with pytest.raises(TraceableException):
            await workflow._handle_compile_and_run_exception(
                error, compiled_graph=None, graph_config={}
            )

    mock_log.warning.assert_called_once_with(
        "checkpoint_notifier is None; error status event not sent to client",
        workflow_id=workflow._workflow_id,
    )
