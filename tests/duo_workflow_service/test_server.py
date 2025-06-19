# pylint: disable=direct-environment-variable-reference
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import AsyncIterable
from unittest.mock import AsyncMock, MagicMock, patch

import grpc
import pytest
from gitlab_cloud_connector import CloudConnectorConfig, CloudConnectorUser, UserClaims
from langchain.globals import get_llm_cache
from langchain_community.cache import SQLiteCache

from contract import contract_pb2
from duo_workflow_service.interceptors.authentication_interceptor import current_user
from duo_workflow_service.internal_events.client import DuoWorkflowInternalEvent
from duo_workflow_service.internal_events.event_enum import CategoryEnum
from duo_workflow_service.server import (
    DuoWorkflowService,
    clean_start_request,
    configure_cache,
    run,
    serve,
    string_to_category_enum,
)


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


def test_run():
    with patch(
        "duo_workflow_service.server.setup_profiling"
    ) as mock_setup_profiling, patch(
        "duo_workflow_service.server.setup_error_tracking"
    ) as mock_setup_error_tracking, patch(
        "duo_workflow_service.server.setup_monitoring"
    ) as mock_setup_monitoring, patch(
        "duo_workflow_service.server.setup_logging"
    ) as mock_setup_logging, patch(
        "duo_workflow_service.server.validate_llm_access"
    ) as mock_validate_llm_access, patch.object(
        DuoWorkflowInternalEvent, "setup"
    ) as mock_internal_event_setup, patch(
        "asyncio.get_event_loop"
    ) as mock_get_loop:
        mock_loop = MagicMock()
        mock_get_loop.return_value = mock_loop

        run()

        mock_setup_profiling.assert_called_once()
        mock_setup_error_tracking.assert_called_once()
        mock_setup_monitoring.assert_called_once()
        mock_setup_logging.assert_called_once_with(json_format=True, to_file=None)
        mock_validate_llm_access.assert_called_once()
        mock_internal_event_setup.assert_called_once()

        assert mock_loop.run_until_complete.call_count == 1
        actual_arg = mock_loop.run_until_complete.call_args[0][0]
        assert asyncio.iscoroutine(actual_arg)


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

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(goal="test")
        )

    current_user.set(CloudConnectorUser(authenticated=True, is_debug=True))
    mock_context = MagicMock(spec=grpc.ServicerContext)
    servicer = DuoWorkflowService()
    result = servicer.ExecuteWorkflow(mock_request_iterator(), mock_context)
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
        raise TimeoutError

    mock_workflow.get_from_outbox = AsyncMock(side_effect=side_effect)

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(goal="test")
        )

    current_user.set(CloudConnectorUser(authenticated=True, is_debug=True))
    mock_context = MagicMock(spec=grpc.ServicerContext)
    servicer = DuoWorkflowService()
    result = servicer.ExecuteWorkflow(mock_request_iterator(), mock_context)
    assert isinstance(result, AsyncIterable)
    with pytest.raises(StopAsyncIteration):
        await anext(result)


@pytest.mark.asyncio
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.resolve_workflow_class")
async def test_workflow_is_cancelled_on_parent_task_cancellation(
    mock_resolve_workflow, mock_abstract_workflow_class
):
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = False
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
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
        real_workflow_task = original_create_task(coro, **kwargs)
        return real_workflow_task

    with patch("asyncio.create_task", side_effect=mock_create_task):
        servicer = DuoWorkflowService()
        result = servicer.ExecuteWorkflow(mock_request_iterator(), mock_context)

        with pytest.raises(StopAsyncIteration):
            await anext(result)

        assert real_workflow_task is not None
        assert real_workflow_task.cancelled()

        mock_context.abort.assert_called_once_with(
            grpc.StatusCode.INTERNAL, "Something went wrong"
        )


@pytest.mark.asyncio
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.resolve_workflow_class")
async def test_execute_workflow(mock_resolve_workflow, mock_abstract_workflow_class):
    mock_workflow_instance = mock_abstract_workflow_class.return_value
    mock_workflow_instance.is_done = False
    mock_workflow_instance.run = AsyncMock()
    mock_workflow_instance.cleanup = AsyncMock()

    checkpoint_action = contract_pb2.Action(newCheckpoint=contract_pb2.NewCheckpoint())

    mock_workflow_instance.get_from_streaming_outbox = MagicMock(
        side_effect=[
            checkpoint_action,
            checkpoint_action,
            checkpoint_action,
            checkpoint_action,
        ]
    )
    mock_workflow_instance.outbox_empty = MagicMock(
        side_effect=[False, False, True, True]
    )
    mock_workflow_instance.get_from_outbox = AsyncMock(
        side_effect=[contract_pb2.Action(), contract_pb2.Action()]
    )
    mock_resolve_workflow.return_value = mock_abstract_workflow_class

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        while True:
            yield contract_pb2.ClientEvent(
                startRequest=contract_pb2.StartWorkflowRequest(goal="test")
            )

    current_user.set(CloudConnectorUser(authenticated=True, is_debug=True))
    mock_context = MagicMock(spec=grpc.ServicerContext)
    servicer = DuoWorkflowService()
    result = servicer.ExecuteWorkflow(mock_request_iterator(), mock_context)
    assert isinstance(result, AsyncIterable)
    assert (await anext(result)).WhichOneof("action") == "newCheckpoint"
    assert (await anext(result)).WhichOneof("action") != "newCheckpoint"
    assert (await anext(result)).WhichOneof("action") == "newCheckpoint"
    assert (await anext(result)).WhichOneof("action") != "newCheckpoint"
    assert (await anext(result)).WhichOneof("action") == "newCheckpoint"
    assert (await anext(result)).WhichOneof("action") == "newCheckpoint"

    assert mock_workflow_instance.add_to_inbox.call_count == 2


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
            scopes=["duo_workflow_execute_workflow"],
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
async def test_grpc_serve():
    mock_server = AsyncMock()
    mock_server.add_insecure_port.return_value = None
    mock_server.start.return_value = None
    mock_server.wait_for_termination.return_value = None

    with patch(
        "duo_workflow_service.server.grpc.aio.server",
        return_value=mock_server,
    ), patch(
        "duo_workflow_service.server.contract_pb2_grpc.add_DuoWorkflowServicer_to_server"
    ) as mock_add_servicer, patch(
        "duo_workflow_service.server.reflection.enable_server_reflection"
    ) as mock_enable_reflection:
        await serve(50052)

    mock_server.add_insecure_port.assert_called_once_with("[::]:50052")
    mock_server.start.assert_called_once()
    mock_server.wait_for_termination.assert_called_once()
    mock_add_servicer.assert_called_once()
    mock_enable_reflection.assert_called_once()


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
    mock_resolve_workflow.return_value = mock_abstract_workflow_class

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(workflowID="123")
        )

    user = CloudConnectorUser(authenticated=True, is_debug=True)
    current_user.set(user)
    mock_context = MagicMock(spec=grpc.ServicerContext)
    servicer = DuoWorkflowService()
    result = servicer.ExecuteWorkflow(mock_request_iterator(), mock_context)
    with pytest.raises(StopAsyncIteration):
        await anext(result)

    mock_abstract_workflow_class.assert_called_once_with(
        workflow_id="123",
        workflow_metadata={},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user=user,
        additional_context=None,
        context_elements=[],
        invocation_metadata={"base_url": "", "gitlab_token": ""},
        mcp_tools=[],
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
    mock_resolve_workflow.return_value = mock_abstract_workflow_class
    mcp_tools = [
        contract_pb2.McpTool(name="get_issue", description="Tool to get issue")
    ]

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(
                workflowID="123",
                workflowMetadata=json.dumps({"key": "value"}),
                mcpTools=mcp_tools,
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
    result = servicer.ExecuteWorkflow(mock_request_iterator(), mock_context)
    assert isinstance(result, AsyncIterable)
    with pytest.raises(StopAsyncIteration):
        await anext(result)

    mock_abstract_workflow_class.assert_called_once_with(
        workflow_id="123",
        workflow_metadata={"key": "value"},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
        user=user,
        additional_context=None,
        context_elements=[],
        invocation_metadata={"base_url": "http://test.url", "gitlab_token": "123"},
        mcp_tools=mcp_tools,
    )


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
        string_to_category_enum("WORKFLOW_SOFTWARE_DEVELOPMENT")
        == CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT
    )

    # Test an invalid category string
    with patch("duo_workflow_service.server.log") as mock_log:
        assert (
            string_to_category_enum("INVALID_CATEGORY")
            == CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT
        )
        mock_log.warning.assert_called_once_with(
            "Unknown category string: INVALID_CATEGORY"
        )
