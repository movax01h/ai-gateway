# pylint: disable=direct-environment-variable-reference,too-many-lines
import asyncio
import json
import os
import signal
from datetime import datetime, timedelta, timezone
from typing import AsyncIterable, List, Optional
from unittest.mock import AsyncMock, MagicMock, call, patch

import grpc
import litellm
import pytest
from gitlab_cloud_connector import (
    CloudConnectorConfig,
    CloudConnectorUser,
    GitLabUnitPrimitive,
    UserClaims,
)
from google.protobuf import struct_pb2
from google.protobuf.json_format import MessageToDict
from langchain.globals import get_llm_cache
from langchain_community.cache import SQLiteCache

from ai_gateway.config import Config, ConfigCustomModels, ConfigGoogleCloudPlatform
from contract import contract_pb2
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    list_configs,
)
from duo_workflow_service.executor.outbox import OutboxSignal
from duo_workflow_service.interceptors.authentication_interceptor import current_user
from duo_workflow_service.server import (
    DuoWorkflowService,
    clean_start_request,
    configure_cache,
    next_client_event,
    run,
    serve,
    setup_signal_handlers,
    string_to_category_enum,
)
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.workflows.type_definitions import (
    AIO_CANCEL_STOP_WORKFLOW_REQUEST,
)
from lib.internal_events.context import InternalEventAdditionalProperties
from lib.internal_events.event_enum import (
    CategoryEnum,
    EventEnum,
    EventLabelEnum,
    EventPropertyEnum,
)


@pytest.fixture
def simple_flow_config():
    return {
        "version": "1.0",
        "environment": "test",
        "components": [{"name": "test_agent", "type": "AgentComponent"}],
        "flow": {"entry_point": "test_agent"},
    }


def create_mock_internal_event_client():
    """Helper function to create a mock internal event client for tests."""
    mock_client = MagicMock()
    mock_client.track_event = MagicMock()
    return mock_client


def test_configure_cache_disabled():
    with patch.dict(os.environ, {"LLM_CACHE": "false"}):
        configure_cache()
        assert get_llm_cache() is None


def test_configure_cache_enabled():
    with patch.dict(os.environ, {"LLM_CACHE": "true"}):
        configure_cache()
        cache = get_llm_cache()
        assert isinstance(cache, SQLiteCache)
        assert cache is not None


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
    ## avoid duplicated mock tool with the same tool name
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
    mock_context.abort = AsyncMock()

    real_workflow_task: asyncio.Task = None  # type: ignore
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
    "workflow_error,successful_execution,expected_status,expected_detail_prefix",
    [
        (None, True, grpc.StatusCode.OK, "workflow execution success"),
        (
            AIO_CANCEL_STOP_WORKFLOW_REQUEST,
            False,
            grpc.StatusCode.OK,
            "workflow execution stopped:",
        ),
        (
            ValueError("Some error"),
            False,
            grpc.StatusCode.INTERNAL,
            "workflow execution failure: ValueError: Some error",
        ),
        (
            None,
            False,
            grpc.StatusCode.UNKNOWN,
            "RPC ended with unknown workflow state:",
        ),
    ],
)
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.resolve_workflow_class")
async def test_execute_workflow_status_codes(
    mock_resolve_workflow,
    mock_abstract_workflow_class,
    workflow_error,
    successful_execution,
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
    """Test that ExecuteWorkflow handles different CancelledError scenarios correctly."""
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
            startRequest=contract_pb2.StartWorkflowRequest(goal="test")
        )

        for _ in range(request_iterator_count):
            yield contract_pb2.ClientEvent(
                actionResponse=contract_pb2.ActionResponse(response="the response")
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
        workflow_type=CategoryEnum.UNKNOWN,
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
        workflow_type=CategoryEnum.UNKNOWN,
        user=user,
        additional_context=None,
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


def test_string_to_category_enum():
    # Test a valid category string
    assert (
        string_to_category_enum(CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT)
        == CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT
    )

    # Test an invalid category string
    with patch("duo_workflow_service.server.log") as mock_log:
        assert string_to_category_enum("INVALID_CATEGORY") == "unknown"
        mock_log.warning.assert_called_once_with(
            "Unknown category string: INVALID_CATEGORY"
        )

    # Test Flow Registry flows:
    for config in list_configs():
        assert (
            string_to_category_enum(
                getattr(CategoryEnum, config["flow_identifier"].upper())
            )
            == config["flow_identifier"]
        )


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
            actionResponse=contract_pb2.ActionResponse(response="the response")
        )

    iterator = mock_request_iterator()

    with patch("duo_workflow_service.server.log") as mock_log:
        result = await next_client_event(iterator)
        assert result.HasField("heartbeat")
        result = await next_client_event(iterator)
        assert result.HasField("heartbeat")
        result = await next_client_event(iterator)
        assert result.actionResponse.response == "the response"
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
    "flow_config_name,flow_config_schema_version",
    [("simple_flow_config", "experimental")],
)
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.resolve_workflow_class")
async def test_execute_workflow_with_flow_config_schema_version_parameterized(
    mock_resolve_workflow,
    mock_abstract_workflow_class,
    request,
    flow_config_name,
    flow_config_schema_version,
):
    # Setup mocks
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

    mock_resolve_workflow.assert_called_once_with(
        "test", flow_config, flow_config_schema_version
    )


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
