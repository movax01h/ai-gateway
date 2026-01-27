# pylint: disable=direct-environment-variable-reference,too-many-lines
import asyncio
import json
import os
import signal
from datetime import datetime, timedelta, timezone
from typing import AsyncIterable, List, Optional, cast
from unittest.mock import AsyncMock, MagicMock, call, patch

import grpc
import litellm
import pytest
from dependency_injector import providers
from gitlab_cloud_connector import (
    CloudConnectorConfig,
    CloudConnectorUser,
    GitLabUnitPrimitive,
    UserClaims,
)
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict

from ai_gateway.config import Config, ConfigCustomModels, ConfigGoogleCloudPlatform
from ai_gateway.container import ContainerApplication
from ai_gateway.prompts import BasePromptRegistry
from contract import contract_pb2
from duo_workflow_service.client_capabilities import client_capabilities
from duo_workflow_service.executor.outbox import OutboxSignal
from duo_workflow_service.interceptors.authentication_interceptor import current_user
from duo_workflow_service.server import (
    DuoWorkflowService,
    clean_start_request,
    next_client_event,
    run,
    serve,
    setup_signal_handlers,
    validate_llm_access,
)
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.workflows.type_definitions import (
    AIO_CANCEL_STOP_WORKFLOW_REQUEST,
    OUTGOING_MESSAGE_TOO_LARGE,
    AdditionalContext,
)
from lib.events import GLReportingEventContext
from lib.internal_events.context import InternalEventAdditionalProperties
from lib.internal_events.event_enum import (
    CategoryEnum,
    EventEnum,
    EventLabelEnum,
    EventPropertyEnum,
)


@pytest.fixture(autouse=True)
def setup_event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)


@pytest.fixture
def simple_flow_config():
    return {
        "version": "v1",
        "environment": "test",
        "components": [{"name": "test_agent", "type": "AgentComponent"}],
        "flow": {"entry_point": "test_agent"},
    }


def create_mock_internal_event_client():
    """Helper function to create a mock internal event client for tests."""
    mock_client = MagicMock()
    mock_client.track_event = MagicMock()
    return mock_client


@pytest.mark.parametrize(
    "custom_models_enabled,should_validate_llm",
    [
        ("true", False),
        ("false", True),
    ],
)
def test_run(custom_models_enabled, vertex_project, should_validate_llm):
    with (
        patch("duo_workflow_service.server.setup_profiling") as mock_setup_profiling,
        patch(
            "duo_workflow_service.server.setup_error_tracking"
        ) as mock_setup_error_tracking,
        patch("duo_workflow_service.server.setup_monitoring") as mock_setup_monitoring,
        patch("duo_workflow_service.server.setup_logging") as mock_setup_logging,
        patch(
            "duo_workflow_service.server.validate_llm_access"
        ) as mock_validate_llm_access,
        patch("asyncio.get_event_loop") as mock_get_loop,
    ):
        mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        mock_loop.run_until_complete = MagicMock()
        mock_get_loop.return_value = mock_loop

        config = Config(
            google_cloud_platform=ConfigGoogleCloudPlatform(project=vertex_project),
            custom_models=ConfigCustomModels(enabled=custom_models_enabled),
        )

        run(config)

        mock_setup_profiling.assert_called_once()
        mock_setup_error_tracking.assert_called_once()
        mock_setup_monitoring.assert_called_once()
        mock_setup_logging.assert_called_once()

        if should_validate_llm:
            mock_validate_llm_access.assert_called_once()
        else:
            mock_validate_llm_access.assert_not_called()

        assert mock_loop.run_until_complete.call_count == 1
        actual_arg = mock_loop.run_until_complete.call_args[0][0]
        assert asyncio.iscoroutine(actual_arg)

        assert litellm.vertex_project == vertex_project

        # Clean up the coroutine to prevent the warning
        actual_arg.close()


def test_validate_ll_access(mock_duo_workflow_service_container: ContainerApplication):
    pkg_prompts = cast(
        providers.Container, mock_duo_workflow_service_container.pkg_prompts
    )
    prompt_registry = cast(BasePromptRegistry, pkg_prompts.prompt_registry())

    with patch.object(prompt_registry, "validate_default_models") as mock_validate:
        validate_llm_access()

    mock_validate.assert_awaited_once_with(GitLabUnitPrimitive.DUO_AGENT_PLATFORM)


@pytest.mark.asyncio
@patch("duo_workflow_service.server.tools_registry._DEFAULT_TOOLS")
@patch("duo_workflow_service.server.tools_registry._READ_ONLY_GITLAB_TOOLS")
@patch("duo_workflow_service.server.tools_registry._AGENT_PRIVILEGES")
@patch("duo_workflow_service.server.convert_to_openai_tool")
async def test_list_tools(
    mock_convert_to_openai_tool,
    mock_agent_privileges,
    mock_readonly_tools,
    mock_default_tools,
):
    # avoid duplicated mock tool with the same tool name
    _tool_class_cache = {}

    def create_mock_tool(name: str, eval_prompts: Optional[List[str]] = None):
        cache_key = name
        if cache_key in _tool_class_cache:
            return _tool_class_cache[cache_key]

        class MockTool(DuoBaseTool):

            def __init__(self):
                super().__init__(
                    name=name,
                    description=f"{name} description",
                    eval_prompts=eval_prompts,
                )

            async def _execute(self, *args, **kwargs):
                pass

        _tool_class_cache[cache_key] = MockTool
        return MockTool

    def mock_convert_side_effect(tool):
        tool_name = tool.name
        return {
            "type": "function",
            "function": {
                "name": tool_name,
                "description": f"{tool_name} description",
            },
        }

    mock_default_tools.__add__.return_value = [create_mock_tool(name="tool1")]
    mock_readonly_tools.__add__.return_value = [
        create_mock_tool(name="tool2", eval_prompts=["prompt2"])
    ]
    mock_agent_privileges.values.return_value = [
        [
            create_mock_tool(name="tool1"),
            create_mock_tool(name="tool2", eval_prompts=["prompt2"]),
        ],
        [create_mock_tool(name="tool3")],
    ]
    mock_convert_to_openai_tool.side_effect = mock_convert_side_effect
    mock_context = MagicMock(spec=grpc.ServicerContext)
    service = DuoWorkflowService()
    response = await service.ListTools(contract_pb2.ListToolsRequest(), mock_context)
    assert isinstance(response, contract_pb2.ListToolsResponse)
    assert len(response.tools) == 3
    assert mock_convert_to_openai_tool.called

    actual_sorted = sorted(
        [MessageToDict(tool) for tool in response.tools],
        key=lambda x: x.get("function", {}).get("name", ""),
    )
    expected_sorted = sorted(
        [
            {
                "type": "function",
                "function": {"description": "tool1 description", "name": "tool1"},
            },
            {
                "type": "function",
                "function": {"description": "tool2 description", "name": "tool2"},
            },
            {
                "type": "function",
                "function": {"description": "tool3 description", "name": "tool3"},
            },
        ],
        key=lambda x: x.get("function", {}).get("name", ""),
    )
    assert actual_sorted == expected_sorted


@pytest.mark.asyncio
@patch("duo_workflow_service.server.flow_registry.list_configs")
async def test_list_flows(mock_list_configs):
    mock_list_configs.return_value = [
        {"name": "flow1", "description": "First flow config"},
        {"name": "flow2", "description": "Second flow config"},
    ]

    mock_context = MagicMock(spec=grpc.ServicerContext)
    service = DuoWorkflowService()
    response = await service.ListFlows(contract_pb2.ListFlowsRequest(), mock_context)

    assert isinstance(response, contract_pb2.ListFlowsResponse)
    assert len(response.configs) == 2
    mock_list_configs.assert_called_once()

    configs_dict = [MessageToDict(config) for config in response.configs]
    expected_configs = [
        {"name": "flow1", "description": "First flow config"},
        {"name": "flow2", "description": "Second flow config"},
    ]
    assert configs_dict == expected_configs


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "filter_params,expected_count,validation_func",
    [
        # Explicit None filters
        (None, 3, lambda configs: len(configs) == 3),  # Should return all configs
        # Filter by flow_identifier
        (
            {"flow_identifier": ["flow1", "flow3"]},
            2,
            lambda configs: all(
                config["flow_identifier"] in ["flow1", "flow3"] for config in configs
            ),
        ),
        # Filter by environment
        (
            {"environment": ["prod"]},
            1,
            lambda configs: all(config["environment"] == "prod" for config in configs),
        ),
        # Filter by version
        (
            {"version": ["v1"]},
            2,
            lambda configs: all(config["version"] == "v1" for config in configs),
        ),
        # Multiple filters (flow_identifier and environment)
        (
            {"flow_identifier": ["flow1"], "environment": ["prod"]},
            1,
            lambda configs: len(configs) == 1
            and configs[0]["flow_identifier"] == "flow1"
            and configs[0]["environment"] == "prod",
        ),
        # Filter that matches no flows
        (
            {"environment": ["staging"]},
            0,
            lambda configs: True,  # No validation needed for empty result
        ),
        # Multiple values for same filter
        (
            {"environment": ["prod", "test"]},
            2,
            lambda configs: all(
                config["environment"] in ["prod", "test"] for config in configs
            ),
        ),
        # Complex multi-filter scenario
        (
            {"version": ["v1"], "environment": ["prod", "dev"]},
            2,
            lambda configs: all(
                config["version"] == "v1" and config["environment"] in ["prod", "dev"]
                for config in configs
            ),
        ),
    ],
)
@patch("duo_workflow_service.server.flow_registry.list_configs")
async def test_list_flows_with_filters(
    mock_list_configs, filter_params, expected_count, validation_func
):
    mock_list_configs.return_value = [
        {
            "flow_identifier": "flow1",
            "version": "v1",
            "environment": "prod",
            "description": "First flow config",
        },
        {
            "flow_identifier": "flow2",
            "version": "v2",
            "environment": "test",
            "description": "Second flow config",
        },
        {
            "flow_identifier": "flow3",
            "version": "v1",
            "environment": "dev",
            "description": "Third flow config",
        },
    ]

    mock_context = MagicMock(spec=grpc.ServicerContext)
    service = DuoWorkflowService()

    if filter_params is None:
        request = contract_pb2.ListFlowsRequest(filters=None)
    else:
        filters = contract_pb2.ListFlowsRequestFilter(**filter_params)
        request = contract_pb2.ListFlowsRequest(filters=filters)

    response = await service.ListFlows(request, mock_context)

    assert len(response.configs) == expected_count

    configs_dict = [MessageToDict(config) for config in response.configs]
    assert validation_func(configs_dict)


@pytest.mark.asyncio
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.resolve_workflow_class")
async def test_execute_workflow_when_no_events_ends(
    mock_resolve_workflow,
    mock_abstract_workflow_class,
):
    mock_resolve_workflow.return_value = mock_abstract_workflow_class
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = True
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
    mock_workflow.get_from_outbox = AsyncMock(
        return_value=OutboxSignal.NO_MORE_OUTBOUND_REQUESTS
    )

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(goal="test")
        )

    current_user.set(CloudConnectorUser(authenticated=True, is_debug=True))
    mock_context = MagicMock(spec=grpc.ServicerContext)
    servicer = DuoWorkflowService()
    result = servicer.ExecuteWorkflow(
        mock_request_iterator(),
        mock_context,
        internal_event_client=create_mock_internal_event_client(),
    )
    assert isinstance(result, AsyncIterable)
    with pytest.raises(StopAsyncIteration):
        await anext(result)


@pytest.mark.asyncio
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.resolve_workflow_class")
async def test_execute_workflow_when_message_too_large_cancels_workflow(
    mock_resolve_workflow,
    mock_abstract_workflow_class,
):
    mock_resolve_workflow.return_value = mock_abstract_workflow_class
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = True
    mock_workflow.run = AsyncMock()
    mock_workflow.cancel = AsyncMock()
    mock_workflow.cleanup = AsyncMock()

    request_id = "test-request-id"
    mock_workflow.get_from_outbox = AsyncMock(
        side_effect=[
            contract_pb2.Action(
                requestID=request_id,
                runCommand=contract_pb2.RunCommandAction(program="a" * 5 * 1024 * 1024),
            ),
            # Calling cancel will always trigger an outbox close which queues
            # this
            OutboxSignal.NO_MORE_OUTBOUND_REQUESTS,
        ]
    )

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(goal="test")
        )

    current_user.set(CloudConnectorUser(authenticated=True, is_debug=True))
    mock_context = MagicMock(spec=grpc.ServicerContext)
    servicer = DuoWorkflowService()
    result = servicer.ExecuteWorkflow(
        mock_request_iterator(),
        mock_context,
        internal_event_client=create_mock_internal_event_client(),
    )
    assert isinstance(result, AsyncIterable)
    with pytest.raises(asyncio.CancelledError):
        await anext(result)

    mock_workflow.fail_outbox_action.assert_called_once_with(
        request_id, OUTGOING_MESSAGE_TOO_LARGE
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.resolve_workflow_class")
async def test_execute_workflow_when_nothing_in_outbox(
    mock_resolve_workflow, mock_abstract_workflow_class
):
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = False
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
    mock_resolve_workflow.return_value = mock_abstract_workflow_class

    def side_effect():
        mock_workflow.is_done = True
        return OutboxSignal.NO_MORE_OUTBOUND_REQUESTS

    mock_workflow.get_from_outbox = AsyncMock(side_effect=side_effect)

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(goal="test")
        )

    current_user.set(CloudConnectorUser(authenticated=True, is_debug=True))
    mock_context = MagicMock(spec=grpc.ServicerContext)
    servicer = DuoWorkflowService()
    result = servicer.ExecuteWorkflow(
        mock_request_iterator(),
        mock_context,
        internal_event_client=create_mock_internal_event_client(),
    )
    assert isinstance(result, AsyncIterable)
    with pytest.raises(StopAsyncIteration):
        await anext(result)


@pytest.mark.asyncio
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.resolve_workflow_class")
async def test_workflow_is_cancelled_on_parent_task_cancellation(
    mock_resolve_workflow, mock_abstract_workflow_class
):
    """Test that workflow task is properly cancelled when parent task is cancelled."""
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = False
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
    mock_workflow.last_gitlab_status = "running"
    mock_resolve_workflow.return_value = mock_abstract_workflow_class

    mock_workflow.get_from_outbox = AsyncMock(
        side_effect=asyncio.CancelledError("Task cancelled")
    )

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(goal="test")
        )

    current_user.set(CloudConnectorUser(authenticated=True, is_debug=True))
    mock_context = MagicMock(spec=grpc.ServicerContext)

    real_workflow_task: asyncio.Task = None
    original_create_task = asyncio.create_task

    def mock_create_task(coro, **kwargs):
        nonlocal real_workflow_task
        if real_workflow_task:
            return original_create_task(coro, **kwargs)

        real_workflow_task = original_create_task(coro, **kwargs)
        return real_workflow_task

    with patch("asyncio.create_task", side_effect=mock_create_task):
        servicer = DuoWorkflowService()
        result = servicer.ExecuteWorkflow(
            mock_request_iterator(),
            mock_context,
            internal_event_client=create_mock_internal_event_client(),
        )

        with pytest.raises(asyncio.CancelledError):
            await anext(result)

        assert real_workflow_task is not None
        assert real_workflow_task.cancelled()

        mock_context.set_code.assert_called_once_with(grpc.StatusCode.CANCELLED)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workflow_error,successful_execution,stop_reason,expected_status,expected_detail_prefix",
    [
        (None, True, None, grpc.StatusCode.OK, "workflow execution success"),
        (
            AIO_CANCEL_STOP_WORKFLOW_REQUEST,
            False,
            None,
            grpc.StatusCode.OK,
            "workflow execution stopped:",
        ),
        (
            AIO_CANCEL_STOP_WORKFLOW_REQUEST,
            False,
            "WORKHORSE_SERVER_SHUTDOWN",
            grpc.StatusCode.UNAVAILABLE,
            "workflow execution interrupted:",
        ),
        (
            AIO_CANCEL_STOP_WORKFLOW_REQUEST,
            False,
            "USER_CANCELLED",
            grpc.StatusCode.OK,
            "workflow execution stopped:",
        ),
        (
            ValueError("Some error"),
            False,
            None,
            grpc.StatusCode.INTERNAL,
            "workflow execution failure: ValueError: Some error",
        ),
        (
            None,
            False,
            None,
            grpc.StatusCode.UNKNOWN,
            "RPC ended with unknown workflow state:",
        ),
        (
            OUTGOING_MESSAGE_TOO_LARGE,
            False,
            None,
            grpc.StatusCode.RESOURCE_EXHAUSTED,
            "Outgoing message too large",
        ),
    ],
)
@patch("duo_workflow_service.server.current_monitoring_context")
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.resolve_workflow_class")
async def test_execute_workflow_status_codes(
    mock_resolve_workflow,
    mock_abstract_workflow_class,
    mock_current_monitoring_context,
    workflow_error,
    successful_execution,
    stop_reason,
    expected_status,
    expected_detail_prefix,
):
    """Test that ExecuteWorkflow sets correct status codes for different workflow states."""
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = True
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
    mock_workflow.successful_execution = MagicMock(return_value=successful_execution)
    mock_workflow.last_error = workflow_error
    mock_workflow.last_gitlab_status = "test_status"
    mock_workflow.get_from_outbox = AsyncMock(
        return_value=OutboxSignal.NO_MORE_OUTBOUND_REQUESTS
    )
    mock_resolve_workflow.return_value = mock_abstract_workflow_class

    mock_monitoring_context = MagicMock()
    mock_monitoring_context.workflow_stop_reason = stop_reason
    mock_current_monitoring_context.get.return_value = mock_monitoring_context

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(goal="test")
        )

    current_user.set(CloudConnectorUser(authenticated=True, is_debug=True))
    mock_context = MagicMock(spec=grpc.ServicerContext)
    servicer = DuoWorkflowService()
    result = servicer.ExecuteWorkflow(
        mock_request_iterator(),
        mock_context,
        internal_event_client=create_mock_internal_event_client(),
    )

    with pytest.raises(StopAsyncIteration):
        await anext(result)

    mock_context.set_code.assert_called_once_with(expected_status)
    actual_detail = mock_context.set_details.call_args[0][0]
    assert actual_detail.startswith(expected_detail_prefix)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "cancel_error_message,expected_status,expected_detail_prefix,expected_log_count",
    [
        (
            AIO_CANCEL_STOP_WORKFLOW_REQUEST,
            grpc.StatusCode.OK,
            "workflow execution stopped:",
            1,  # Only called from abort_workflow
        ),
        (
            # Tuple-style error from LangGraph cleanup
            (AIO_CANCEL_STOP_WORKFLOW_REQUEST, "<Task cancelled>"),
            grpc.StatusCode.OK,
            "workflow execution stopped:",
            1,  # Only called from abort_workflow
        ),
        (
            "Some other cancellation",
            grpc.StatusCode.CANCELLED,
            "RPC cancelled by client",
            2,  # Called from main handler AND abort_workflow
        ),
    ],
)
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.resolve_workflow_class")
@patch("duo_workflow_service.server.log_exception")
async def test_execute_workflow_cancellation_handling(
    mock_log_exception,
    mock_resolve_workflow,
    mock_abstract_workflow_class,
    cancel_error_message,
    expected_status,
    expected_detail_prefix,
    expected_log_count,
):
    """Test that ExecuteWorkflow handles different CancelledError scenarios correctly.

    This test verifies that:
    1. Simple string error messages are handled correctly
    2. Tuple-style error messages (from LangGraph cleanup) are handled correctly
    3. Unexpected cancellations are logged appropriately
    """
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = False
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
    mock_workflow.last_gitlab_status = "retry"
    mock_workflow.get_from_outbox = AsyncMock(
        side_effect=asyncio.CancelledError(cancel_error_message)
    )
    mock_resolve_workflow.return_value = mock_abstract_workflow_class

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(goal="test")
        )

    current_user.set(CloudConnectorUser(authenticated=True, is_debug=True))
    mock_context = MagicMock(spec=grpc.ServicerContext)
    servicer = DuoWorkflowService()
    result = servicer.ExecuteWorkflow(
        mock_request_iterator(),
        mock_context,
        internal_event_client=create_mock_internal_event_client(),
    )

    with pytest.raises(asyncio.CancelledError):
        await anext(result)

    # Verify status code and details
    mock_context.set_code.assert_called_once_with(expected_status)
    actual_detail = mock_context.set_details.call_args[0][0]
    assert actual_detail.startswith(expected_detail_prefix)

    # Verify logging behavior
    assert mock_log_exception.call_count == expected_log_count

    # For the main handler call (when expected_log_count == 2), verify the first call
    if expected_log_count == 2:
        # First call should be from the main exception handler
        first_call_exception = mock_log_exception.call_args_list[0][0][0]
        assert isinstance(first_call_exception, asyncio.CancelledError)
        assert str(first_call_exception) == cancel_error_message


@pytest.mark.asyncio
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.resolve_workflow_class")
@pytest.mark.parametrize(
    ("unidirectional_streaming_enabled", "request_iterator_count"),
    [
        ("", 6),
        ("enabled", 3),
    ],
)
async def test_execute_workflow(
    mock_resolve_workflow,
    mock_abstract_workflow_class,
    unidirectional_streaming_enabled,
    request_iterator_count,
):
    mock_workflow_instance = mock_abstract_workflow_class.return_value
    mock_workflow_instance.is_done = False
    mock_workflow_instance.run = AsyncMock()
    mock_workflow_instance.cleanup = AsyncMock()

    checkpoint = contract_pb2.NewCheckpoint(checkpoint='{"checkpoint":1}')
    mock_notifier = MagicMock()
    mock_notifier.most_recent_checkpoint_number = MagicMock(side_effect=[1, 2, 3, 4])
    mock_notifier.most_recent_new_checkpoint = MagicMock(return_value=checkpoint)
    mock_workflow_instance.checkpoint_notifier = mock_notifier

    mock_workflow_instance.get_from_outbox = AsyncMock(
        side_effect=[
            contract_pb2.Action(
                newCheckpoint=contract_pb2.NewCheckpoint(), requestID="1"
            ),
            contract_pb2.Action(requestID="2"),
            contract_pb2.Action(
                newCheckpoint=contract_pb2.NewCheckpoint(), requestID="3"
            ),
            contract_pb2.Action(requestID="4"),
            contract_pb2.Action(
                newCheckpoint=contract_pb2.NewCheckpoint(), requestID="5"
            ),
            contract_pb2.Action(
                newCheckpoint=contract_pb2.NewCheckpoint(), requestID="6"
            ),
            OutboxSignal.NO_MORE_OUTBOUND_REQUESTS,
        ]
    )
    mock_resolve_workflow.return_value = mock_abstract_workflow_class

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(
                goal="test",
                clientCapabilities=["capability_a", "capability_b"],
            )
        )

        for _ in range(request_iterator_count):
            yield contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(
                    plainTextResponse=contract_pb2.PlainTextResponse(
                        response="the response"
                    )
                )
            )

    current_user.set(CloudConnectorUser(authenticated=True, is_debug=True))
    mock_context = MagicMock(spec=grpc.ServicerContext)
    mock_context.invocation_metadata.return_value = [
        ("x-gitlab-unidirectional-streaming", unidirectional_streaming_enabled),
    ]
    servicer = DuoWorkflowService()
    result = servicer.ExecuteWorkflow(
        mock_request_iterator(),
        mock_context,
        internal_event_client=create_mock_internal_event_client(),
    )
    assert isinstance(result, AsyncIterable)
    assert (await anext(result)).WhichOneof("action") == "newCheckpoint"
    assert (await anext(result)).WhichOneof("action") != "newCheckpoint"
    assert (await anext(result)).WhichOneof("action") == "newCheckpoint"
    assert (await anext(result)).WhichOneof("action") != "newCheckpoint"
    assert (await anext(result)).WhichOneof("action") == "newCheckpoint"
    assert (await anext(result)).WhichOneof("action") == "newCheckpoint"
    with pytest.raises(StopAsyncIteration):
        await anext(result)

    assert (
        mock_workflow_instance.set_action_response.call_count == request_iterator_count
    )

    assert client_capabilities.get() == {"capability_a", "capability_b"}


@pytest.mark.asyncio
@patch("duo_workflow_service.server.TokenAuthority")
@patch("contract.contract_pb2.GenerateTokenResponse")
@patch.dict(os.environ, {"CLOUD_CONNECTOR_SERVICE_NAME": "gitlab-duo-workflow-service"})
async def test_generate_token(mock_generate_token_response, mock_token_authority):
    one_hour_later = datetime.now(tz=timezone.utc) + timedelta(hours=1)
    mock_token_authority.return_value.encode = MagicMock(
        return_value=("token", one_hour_later)
    )
    mock_context = MagicMock(spec=grpc.ServicerContext)

    user = CloudConnectorUser(
        authenticated=True,
        is_debug=False,
        claims=UserClaims(
            issuer="gitlab.com",
            scopes=[
                "duo_agent_platform",
                "duo_chat",
                "include_file_context",
                "unknown_scope",
            ],
        ),
    )
    current_user.set(user)

    servicer = DuoWorkflowService()
    await servicer.GenerateToken(contract_pb2.GenerateTokenRequest(), mock_context)

    args = mock_token_authority.return_value.encode.call_args.args
    passed_scopes = args[-1]
    assert set(passed_scopes) == {
        "duo_agent_platform",
        "duo_chat",
        "include_file_context",
    }
    mock_generate_token_response.assert_called_once_with(
        token="token", expiresAt=one_hour_later
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.server.TokenAuthority")
@patch("contract.contract_pb2.GenerateTokenResponse")
@patch.dict(os.environ, {"CLOUD_CONNECTOR_SERVICE_NAME": "gitlab-duo-workflow-service"})
async def test_generate_token_with_legacy_duo_workflow_execute_workflow_up(
    mock_generate_token_response, mock_token_authority
):
    one_hour_later = datetime.now(tz=timezone.utc) + timedelta(hours=1)
    mock_token_authority.return_value.encode = MagicMock(
        return_value=("token", one_hour_later)
    )
    mock_context = MagicMock(spec=grpc.ServicerContext)

    user = CloudConnectorUser(
        authenticated=True,
        is_debug=False,
        claims=UserClaims(
            issuer="gitlab.com",
            scopes=[
                "duo_workflow_execute_workflow",
                "duo_chat",
                "include_file_context",
                "unknown_scope",
            ],
        ),
    )
    current_user.set(user)

    servicer = DuoWorkflowService()
    await servicer.GenerateToken(contract_pb2.GenerateTokenRequest(), mock_context)

    args = mock_token_authority.return_value.encode.call_args.args
    passed_scopes = args[-1]
    assert set(passed_scopes) == {
        "duo_workflow_execute_workflow",
        "duo_chat",
        "include_file_context",
    }
    mock_generate_token_response.assert_called_once_with(
        token="token", expiresAt=one_hour_later
    )


@pytest.mark.asyncio
@patch.dict(os.environ, {"CLOUD_CONNECTOR_SERVICE_NAME": "gitlab-duo-workflow-service"})
async def test_generate_token_with_self_signed_token_issuer():
    user = CloudConnectorUser(
        authenticated=True,
        is_debug=False,
        claims=UserClaims(
            issuer=CloudConnectorConfig().service_name,
            scopes=["duo_agent_platform"],
        ),
    )
    current_user.set(user)
    mock_context = MagicMock(spec=grpc.ServicerContext)
    mock_context.abort.side_effect = grpc.RpcError("Aborted")

    servicer = DuoWorkflowService()
    with pytest.raises(grpc.RpcError):
        await servicer.GenerateToken(contract_pb2.GenerateTokenRequest(), mock_context)

    mock_context.abort.assert_called_once_with(
        grpc.StatusCode.PERMISSION_DENIED, "Unauthorized to generate token"
    )


@pytest.mark.asyncio
@patch.dict(os.environ, {"CLOUD_CONNECTOR_SERVICE_NAME": "gitlab-duo-workflow-service"})
async def test_generate_token_unauthorized_for_chat_workflow():
    user = CloudConnectorUser(
        authenticated=True,
        is_debug=False,
        claims=UserClaims(
            issuer="gitlab.com",
            scopes=["duo_agent_platform"],  # Missing duo_chat scope
        ),
    )

    # Mock the can method to return False for chat primitive
    user.can = MagicMock(return_value=False)
    current_user.set(user)

    mock_context = MagicMock(spec=grpc.ServicerContext)
    mock_context.abort.side_effect = grpc.RpcError("Aborted")

    servicer = DuoWorkflowService()
    request = contract_pb2.GenerateTokenRequest(workflowDefinition="chat")

    with pytest.raises(grpc.RpcError):
        await servicer.GenerateToken(request, mock_context)

    # Verify user.can was called with DUO_CHAT primitive
    user.can.assert_called_once_with(
        unit_primitive=GitLabUnitPrimitive.DUO_CHAT,
        disallowed_issuers=[CloudConnectorConfig().service_name],
    )

    mock_context.abort.assert_called_once_with(
        grpc.StatusCode.PERMISSION_DENIED, "Unauthorized to generate token"
    )


@pytest.mark.asyncio
@patch.dict(os.environ, {"CLOUD_CONNECTOR_SERVICE_NAME": "gitlab-duo-workflow-service"})
async def test_generate_token_unauthorized_for_any_flow():
    user = CloudConnectorUser(
        authenticated=True,
        is_debug=False,
        claims=UserClaims(
            issuer="gitlab.com",
            scopes=["duo_chat"],  # Missing duo_agent_platform scope
        ),
    )

    # Mock the can method to return False for chat primitive
    user.can = MagicMock(return_value=False)
    current_user.set(user)

    mock_context = MagicMock(spec=grpc.ServicerContext)
    mock_context.abort.side_effect = grpc.RpcError("Aborted")

    servicer = DuoWorkflowService()
    request = contract_pb2.GenerateTokenRequest(workflowDefinition="agent")

    with pytest.raises(grpc.RpcError):
        await servicer.GenerateToken(request, mock_context)

    user.can.assert_has_calls(
        [
            call(
                unit_primitive=GitLabUnitPrimitive.DUO_AGENT_PLATFORM,
                disallowed_issuers=[CloudConnectorConfig().service_name],
            ),
            call(
                unit_primitive=GitLabUnitPrimitive.DUO_WORKFLOW_EXECUTE_WORKFLOW,
                disallowed_issuers=[CloudConnectorConfig().service_name],
            ),
        ]
    )

    mock_context.abort.assert_called_once_with(
        grpc.StatusCode.PERMISSION_DENIED, "Unauthorized to generate token"
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.server.setup_signal_handlers")
async def test_grpc_server(mock_setup_signal_handlers):
    """Test that the gRPC server starts correctly and sets up signal handlers."""
    mock_server = AsyncMock()
    mock_server.add_insecure_port.return_value = None
    mock_server.start.return_value = None
    mock_server.wait_for_termination.return_value = None

    with (
        patch(
            "duo_workflow_service.server.grpc.aio.server",
            return_value=mock_server,
        ),
        patch(
            "duo_workflow_service.server.contract_pb2_grpc.add_DuoWorkflowServicer_to_server"
        ) as mock_add_servicer,
        patch(
            "duo_workflow_service.server.reflection.enable_server_reflection"
        ) as mock_enable_reflection,
        patch("duo_workflow_service.server.connection_pool") as mock_connection_pool,
    ):
        mock_connection_pool.__aenter__ = AsyncMock(return_value=mock_connection_pool)
        mock_connection_pool.__aexit__ = AsyncMock(return_value=None)

        await serve(50052)

    mock_server.add_insecure_port.assert_called_once_with("[::]:50052")
    mock_server.start.assert_called_once()
    mock_server.wait_for_termination.assert_called_once()
    mock_add_servicer.assert_called_once()
    mock_enable_reflection.assert_called_once()
    mock_setup_signal_handlers.assert_called_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "signal_type,grace_period_env,expected_grace_period",
    [
        (signal.SIGTERM, "15", 15),
        (signal.SIGTERM, None, None),
        (signal.SIGINT, "20", 20),
        (signal.SIGINT, None, None),
    ],
)
async def test_signal_handler_calls_server_stop(
    signal_type, grace_period_env, expected_grace_period
):
    """Test that signal handlers call server.stop() with the correct grace period."""
    mock_server = AsyncMock()
    mock_server.stop = AsyncMock()
    loop = asyncio.get_running_loop()

    env_dict = {}
    if grace_period_env is not None:
        env_dict["DUO_WORKFLOW_SHUTDOWN_GRACE_PERIOD_S"] = grace_period_env

    with (
        patch.dict(os.environ, env_dict, clear=True),
        patch("duo_workflow_service.server.log") as mock_log,
    ):
        setup_signal_handlers(mock_server, loop)

        os.kill(os.getpid(), signal_type)

        # Give the event loop time to process the signal
        await asyncio.sleep(0.1)

        mock_server.stop.assert_called()
        call_args = mock_server.stop.call_args
        assert call_args[1]["grace"] == expected_grace_period

        log_calls = [str(call) for call in mock_log.info.call_args_list]
        assert any("graceful shutdown" in log_call.lower() for log_call in log_calls)


@pytest.mark.asyncio
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.resolve_workflow_class")
async def test_execute_workflow_missing_workflow_metadata(
    mock_resolve_workflow, mock_abstract_workflow_class
):
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = True
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
    mock_workflow.last_error = ValueError("validation error")
    mock_workflow.successful_execution = MagicMock(return_value=False)
    mock_workflow.get_from_outbox = AsyncMock(
        return_value=OutboxSignal.NO_MORE_OUTBOUND_REQUESTS
    )
    mock_resolve_workflow.return_value = mock_abstract_workflow_class

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(workflowID="123")
        )

    user = CloudConnectorUser(authenticated=True, is_debug=True)
    current_user.set(user)
    mock_context = MagicMock(spec=grpc.ServicerContext)
    servicer = DuoWorkflowService()
    result = servicer.ExecuteWorkflow(
        mock_request_iterator(),
        mock_context,
        internal_event_client=create_mock_internal_event_client(),
    )
    with pytest.raises(StopAsyncIteration):
        await anext(result)

    mock_abstract_workflow_class.assert_called_once_with(
        workflow_id="123",
        workflow_metadata={},
        workflow_type=GLReportingEventContext.from_workflow_definition(
            "software_development"
        ),  # backward compatibility when workflow_definition is empty
        user=user,
        additional_context=None,
        invocation_metadata={"base_url": "", "gitlab_token": ""},
        mcp_tools=[],
        approval=contract_pb2.Approval(),
        language_server_version=None,
        preapproved_tools=[],
    )

    mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
    mock_context.set_details.assert_called_once_with(
        "workflow execution failure: ValueError: validation error"
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.resolve_workflow_class")
async def test_execute_workflow_valid_workflow_metadata(
    mock_resolve_workflow, mock_abstract_workflow_class
):
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = True
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
    mock_workflow.last_error = None
    mock_workflow.get_from_outbox = AsyncMock(
        return_value=OutboxSignal.NO_MORE_OUTBOUND_REQUESTS
    )
    mock_resolve_workflow.return_value = mock_abstract_workflow_class
    mcp_tools = [
        contract_pb2.McpTool(name="get_issue", description="Tool to get issue")
    ]
    approval = contract_pb2.Approval(approval=contract_pb2.Approval.Approved())
    preapproved_tools = ["get_issue"]

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(
                workflowID="123",
                workflowMetadata=json.dumps({"key": "value"}),
                mcpTools=mcp_tools,
                approval=approval,
                preapproved_tools=preapproved_tools,
                additional_context=[
                    contract_pb2.AdditionalContext(
                        category="merge_request",
                        id="1",
                        content="merge request data",
                        metadata="{}",
                    ),
                ],
            )
        )

    user = CloudConnectorUser(authenticated=True, is_debug=True)
    current_user.set(user)
    mock_context = MagicMock(spec=grpc.ServicerContext)
    mock_context.invocation_metadata.return_value = [
        ("x-gitlab-base-url", "http://test.url"),
        ("x-gitlab-oauth-token", "123"),
    ]
    servicer = DuoWorkflowService()
    result = servicer.ExecuteWorkflow(
        mock_request_iterator(),
        mock_context,
        internal_event_client=create_mock_internal_event_client(),
    )
    assert isinstance(result, AsyncIterable)
    with pytest.raises(StopAsyncIteration):
        await anext(result)

    mock_abstract_workflow_class.assert_called_once_with(
        workflow_id="123",
        workflow_metadata={"key": "value"},
        workflow_type=GLReportingEventContext.from_workflow_definition(
            "software_development"
        ),  # backward compatibility when workflow_definition is empty,
        user=user,
        additional_context=[
            AdditionalContext(
                category="merge_request",
                id="1",
                content="merge request data",
                metadata={},
            ),
        ],
        invocation_metadata={"base_url": "http://test.url", "gitlab_token": "123"},
        mcp_tools=mcp_tools,
        approval=approval,
        language_server_version=None,
        preapproved_tools=preapproved_tools,
    )

    mock_context.set_code.assert_called_once_with(grpc.StatusCode.OK)


def test_clean_start_request():
    # Create a test request with workflow metadata
    start_request = contract_pb2.StartWorkflowRequest(
        workflowID="test-id",
        goal="test-goal",
        workflowMetadata=json.dumps({"key": "value"}),
    )
    client_event = contract_pb2.ClientEvent(startRequest=start_request)

    # Call the clean_start_request function
    cleaned_request = clean_start_request(client_event)

    # Verify that the cleaned request is a new object (not the same instance)
    assert cleaned_request is not client_event

    # Verify that the original request still has its metadata
    assert client_event.startRequest.workflowMetadata == json.dumps({"key": "value"})

    # Verify that the cleaned request has no metadata but retains other fields
    assert cleaned_request.startRequest.workflowMetadata == ""
    assert cleaned_request.startRequest.workflowID == "test-id"
    assert cleaned_request.startRequest.goal == ""


@pytest.mark.asyncio
async def test_next_client_event():
    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            heartbeat=contract_pb2.HeartbeatRequest(timestamp=123)
        )
        yield contract_pb2.ClientEvent(
            heartbeat=contract_pb2.HeartbeatRequest(timestamp=456)
        )
        yield contract_pb2.ClientEvent(
            actionResponse=contract_pb2.ActionResponse(
                plainTextResponse=contract_pb2.PlainTextResponse(
                    response="the response"
                )
            )
        )

    iterator = mock_request_iterator()

    with patch("duo_workflow_service.server.log") as mock_log:
        result = await next_client_event(iterator)
        assert result.HasField("heartbeat")
        result = await next_client_event(iterator)
        assert result.HasField("heartbeat")
        result = await next_client_event(iterator)
        assert result.actionResponse.plainTextResponse.response == "the response"
        assert not result.HasField("heartbeat")
        mock_log.info.assert_called()


@pytest.mark.asyncio
async def test_next_client_event_client_streaming_closed():
    mock_iterator = AsyncMock()
    mock_iterator.__next__.side_effect = StopAsyncIteration
    result = await next_client_event(mock_iterator)
    assert result is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_ids,expected_action_ids",
    [
        (
            ["request-1", "request-2", "request-3"],
            ["request-1", "request-2", "request-3"],
        ),
        (["single-request"], ["single-request"]),
        ([], []),
        (
            ["req-a", "req-b", "req-c", "req-d", "req-e"],
            ["req-a", "req-b", "req-c", "req-d", "req-e"],
        ),
    ],
)
async def test_self_hosted_execute_workflow(request_ids, expected_action_ids):
    """Test that TrackSelfHostedExecuteWorkflow echoes back client events with matching requestID."""
    user = CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            issuer="gitlab.com",
            scopes=["duo_agent_platform"],
        ),
    )
    user.can = MagicMock(return_value=True)
    current_user.set(user)

    async def mock_request_iterator() -> (
        AsyncIterable[contract_pb2.TrackSelfHostedClientEvent]
    ):
        for request_id in request_ids:
            yield contract_pb2.TrackSelfHostedClientEvent(
                requestID=request_id,
                workflowID="test-workflow",
                featureQualifiedName="test_feature",
                featureAiCatalogItem=True,
            )

    mock_context = MagicMock(spec=grpc.ServicerContext)
    servicer = DuoWorkflowService()

    result = servicer.TrackSelfHostedExecuteWorkflow(
        mock_request_iterator(),
        mock_context,
        billing_event_client=MagicMock(),
    )

    assert isinstance(result, AsyncIterable)

    received_action_ids = []
    async for action in result:
        assert isinstance(action, contract_pb2.TrackSelfHostedAction)
        received_action_ids.append(action.requestID)

    assert received_action_ids == expected_action_ids


@pytest.mark.asyncio
@patch.dict(os.environ, {"CLOUD_CONNECTOR_SERVICE_NAME": "gitlab-duo-workflow-service"})
async def test_track_self_hosted_execute_workflow_unauthorized():
    user = CloudConnectorUser(
        authenticated=True,
        is_debug=False,
        claims=UserClaims(
            issuer="gitlab.com",
            scopes=["duo_chat"],
        ),
    )
    user.can = MagicMock(return_value=False)
    current_user.set(user)

    mock_context = MagicMock(spec=grpc.ServicerContext)
    mock_context.abort = AsyncMock(side_effect=grpc.RpcError("Aborted"))

    async def mock_request_iterator() -> (
        AsyncIterable[contract_pb2.TrackSelfHostedClientEvent]
    ):
        yield contract_pb2.TrackSelfHostedClientEvent(
            requestID="test-req",
            workflowID="test-workflow",
            featureQualifiedName="test_feature",
            featureAiCatalogItem=True,
        )

    servicer = DuoWorkflowService()
    result_generator = servicer.TrackSelfHostedExecuteWorkflow(
        mock_request_iterator(),
        mock_context,
        billing_event_client=MagicMock(),
    )

    with pytest.raises(grpc.RpcError):
        _ = [action async for action in result_generator]

    user.can.assert_called_once_with(
        unit_primitive=GitLabUnitPrimitive.DUO_AGENT_PLATFORM,
        disallowed_issuers=[CloudConnectorConfig().service_name],
    )

    mock_context.abort.assert_called_once_with(
        grpc.StatusCode.PERMISSION_DENIED,
        "Unauthorized to track self-hosted workflow execution",
    )


@pytest.mark.asyncio
async def test_track_self_hosted_execute_workflow_billing_event():
    user = CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(
            issuer="gitlab.com",
            scopes=["duo_agent_platform"],
        ),
    )
    user.can = MagicMock(return_value=True)
    current_user.set(user)

    mock_context = MagicMock(spec=grpc.ServicerContext)
    mock_billing_client = MagicMock()

    async def mock_request_iterator() -> (
        AsyncIterable[contract_pb2.TrackSelfHostedClientEvent]
    ):
        yield contract_pb2.TrackSelfHostedClientEvent(
            requestID="test-req-id",
            workflowID="test-workflow-id",
            featureQualifiedName="test_feature",
            featureAiCatalogItem=True,
        )

    servicer = DuoWorkflowService()
    result = servicer.TrackSelfHostedExecuteWorkflow(
        mock_request_iterator(),
        mock_context,
        billing_event_client=mock_billing_client,
    )

    actions = [action async for action in result]
    assert len(actions) == 1
    assert actions[0].requestID == "test-req-id"

    mock_billing_client.track_billing_event.assert_called_once()
    call_args = mock_billing_client.track_billing_event.call_args
    assert call_args.kwargs["user"] == user
    assert call_args.kwargs["category"] == "DuoWorkflowService"
    assert call_args.kwargs["unit_of_measure"] == "request"
    assert call_args.kwargs["quantity"] == 1

    metadata = call_args.kwargs["metadata"]
    assert metadata["workflow_id"] == "test-workflow-id"
    assert metadata["feature_qualified_name"] == "test_feature"
    assert metadata["feature_ai_catalog_item"] is True
    assert metadata["execution_environment"] == "duo_agent_platform"
    assert "llm_operations" in metadata
    assert len(metadata["llm_operations"]) == 1
    assert metadata["llm_operations"][0]["model_id"] == "self-hosted-model"
    assert metadata["llm_operations"][0]["token_count"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "flow_config_name,flow_config_schema_version,ignore_schema_version,expected_version",
    [
        ("simple_flow_config", "experimental", True, "v1"),
        ("simple_flow_config", "experimental", False, "experimental"),
    ],
)
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.resolve_workflow_class")
@patch("duo_workflow_service.server.language_server_version")
async def test_execute_workflow_with_flow_config_schema_version_parameterized(
    mock_language_server_version,
    mock_resolve_workflow,
    mock_abstract_workflow_class,
    request,
    flow_config_name,
    flow_config_schema_version,
    ignore_schema_version,
    expected_version,
):
    # Setup mocks
    mock_language_server_version.get.return_value.ignore_broken_flow_schema_version.return_value = (
        ignore_schema_version
    )
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = True
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
    mock_workflow.get_from_outbox = AsyncMock(
        return_value=OutboxSignal.NO_MORE_OUTBOUND_REQUESTS
    )
    mock_resolve_workflow.return_value = mock_abstract_workflow_class

    flow_config = request.getfixturevalue(flow_config_name)
    flow_config_struct = struct_pb2.Struct()
    flow_config_struct.update(flow_config)

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(
                workflowID="test-workflow-123",
                workflowDefinition="test",
                flowConfig=flow_config,
                flowConfigSchemaVersion=flow_config_schema_version,
            )
        )

    # Setup user and context
    user = CloudConnectorUser(authenticated=True, is_debug=True)
    current_user.set(user)
    mock_context = MagicMock(spec=grpc.ServicerContext)
    mock_context.invocation_metadata.return_value = []

    servicer = DuoWorkflowService()
    result = servicer.ExecuteWorkflow(
        mock_request_iterator(),
        mock_context,
        internal_event_client=create_mock_internal_event_client(),
    )

    with pytest.raises(StopAsyncIteration):
        await anext(result)

    mock_resolve_workflow.assert_called_once_with("test", flow_config, expected_version)


@pytest.mark.asyncio
@patch("duo_workflow_service.server.duo_workflow_metrics")
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.resolve_workflow_class")
async def test_execute_workflow_tracks_receive_start_request_internal_event(
    mock_resolve_workflow, mock_abstract_workflow_class, mock_duo_workflow_metrics
):
    """Test that both the receive_start_request internal event and Prometheus metric are tracked when ExecuteWorkflow is
    called."""
    # Setup mocks
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = True
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
    mock_workflow.get_from_outbox = AsyncMock(
        return_value=OutboxSignal.NO_MORE_OUTBOUND_REQUESTS
    )
    mock_resolve_workflow.return_value = mock_abstract_workflow_class

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(
                workflowID="test-workflow-123",
                workflowDefinition="software_development",
            )
        )

    # Setup user and context
    user = CloudConnectorUser(authenticated=True, is_debug=True)
    current_user.set(user)
    mock_context = MagicMock(spec=grpc.ServicerContext)
    mock_context.invocation_metadata.return_value = []

    # Setup servicer
    servicer = DuoWorkflowService()

    # Execute the test with mocked internal_event_client parameter
    mock_internal_event_client = create_mock_internal_event_client()
    result = servicer.ExecuteWorkflow(
        mock_request_iterator(),
        mock_context,
        internal_event_client=mock_internal_event_client,
    )

    # Consume the async iterator to trigger the internal event tracking
    with pytest.raises(StopAsyncIteration):
        await anext(result)

    # Verify the Prometheus metric was tracked
    mock_duo_workflow_metrics.count_agent_platform_receive_start_counter.assert_called_once_with(
        flow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT
    )

    # Verify the internal event was tracked
    mock_internal_event_client.track_event.assert_called_with(
        event_name=EventEnum.RECEIVE_START_REQUEST.value,
        additional_properties=InternalEventAdditionalProperties(
            label=EventLabelEnum.WORKFLOW_RECEIVE_START_REQUEST_LABEL.value,
            property=EventPropertyEnum.WORKFLOW_ID.value,
            value="test-workflow-123",
        ),
        category=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT.value,
    )


@pytest.mark.asyncio
async def test_send_events_sends_all_checkpoints_if_number_increases():
    yielded = []

    events = [
        contract_pb2.Action(
            newCheckpoint=contract_pb2.NewCheckpoint(),
        ),
        contract_pb2.Action(
            newCheckpoint=contract_pb2.NewCheckpoint(),
        ),
        OutboxSignal.NO_MORE_OUTBOUND_REQUESTS,
    ]

    mock_workflow_task = MagicMock()

    mock_workflow = MagicMock()
    mock_workflow.get_from_outbox = AsyncMock(side_effect=events)

    # Mock checkpoint_number to keep increasing so we send all checkpoints
    mock_notifier = MagicMock()
    mock_notifier.most_recent_checkpoint_number = MagicMock(side_effect=[1, 2])

    # Then mock the checkpoints returned to be different
    checkpoint1 = contract_pb2.NewCheckpoint(checkpoint='{"checkpoint":1}')
    checkpoint2 = contract_pb2.NewCheckpoint(checkpoint='{"checkpoint":2}')
    mock_notifier.most_recent_new_checkpoint = MagicMock(
        side_effect=[checkpoint1, checkpoint2]
    )
    mock_workflow.checkpoint_notifier = mock_notifier
    servicer = DuoWorkflowService()
    async for action in servicer.send_events(mock_workflow, mock_workflow_task):
        yielded.append(action)

    assert len(yielded) == 2
    assert yielded[0].newCheckpoint.checkpoint == '{"checkpoint":1}'
    assert yielded[1].newCheckpoint.checkpoint == '{"checkpoint":2}'


@pytest.mark.asyncio
async def test_send_events_sends_skips_checkpoint_if_already_sent():
    yielded = []

    # Add 2 newCheckpoint but we only expect to yield them once
    events = [
        contract_pb2.Action(
            newCheckpoint=contract_pb2.NewCheckpoint(),
        ),
        contract_pb2.Action(
            newCheckpoint=contract_pb2.NewCheckpoint(),
        ),
        OutboxSignal.NO_MORE_OUTBOUND_REQUESTS,
    ]

    mock_workflow_task = MagicMock()

    mock_workflow = MagicMock()
    mock_workflow.get_from_outbox = AsyncMock(side_effect=events)

    # Now we mock most_recent_checkpoint_number to always return the same
    # value. This should ensure that send_events skips the later checkpoints as
    # it assumes it has already been sent.
    mock_notifier = MagicMock()
    mock_notifier.most_recent_checkpoint_number = MagicMock(side_effect=[1, 1])

    checkpoint1 = contract_pb2.NewCheckpoint(checkpoint='{"checkpoint":1}')
    mock_notifier.most_recent_new_checkpoint = MagicMock(side_effect=[checkpoint1])
    mock_workflow.checkpoint_notifier = mock_notifier
    servicer = DuoWorkflowService()
    async for action in servicer.send_events(mock_workflow, mock_workflow_task):
        yielded.append(action)

    assert len(yielded) == 1
    assert yielded[0].newCheckpoint.checkpoint == '{"checkpoint":1}'
