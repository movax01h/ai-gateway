import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import AsyncIterable
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import grpc
import pytest
from gitlab_cloud_connector import (
    CloudConnectorConfig,
    CloudConnectorUser,
    GitLabUnitPrimitive,
    UserClaims,
)

from contract import contract_pb2
from duo_workflow_service.interceptors.authentication_interceptor import current_user
from duo_workflow_service.internal_events.event_enum import CategoryEnum
from duo_workflow_service.servers.grpc_server import (
    GrpcServer,
    grpc_serve,
    string_to_category_enum,
)


@pytest.mark.asyncio
@patch("duo_workflow_service.servers.grpc_server.AbstractWorkflow")
@patch("duo_workflow_service.servers.grpc_server.resolve_workflow_class")
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
    servicer = GrpcServer()
    result = servicer.ExecuteWorkflow(mock_request_iterator(), mock_context)
    assert isinstance(result, AsyncIterable)
    with pytest.raises(StopAsyncIteration):
        await anext(result)


@pytest.mark.asyncio
@patch("asyncio.sleep")
@patch("duo_workflow_service.servers.grpc_server.AbstractWorkflow")
@patch("duo_workflow_service.servers.grpc_server.resolve_workflow_class")
async def test_execute_workflow_when_nothing_in_outbox(
    mock_resolve_workflow, mock_abstract_workflow_class, mock_sleep
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
    servicer = GrpcServer()
    result = servicer.ExecuteWorkflow(mock_request_iterator(), mock_context)
    assert isinstance(result, AsyncIterable)
    with pytest.raises(StopAsyncIteration):
        await anext(result)


@pytest.mark.asyncio
@patch("duo_workflow_service.servers.grpc_server.AbstractWorkflow")
@patch("duo_workflow_service.servers.grpc_server.resolve_workflow_class")
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
        servicer = GrpcServer()
        result = servicer.ExecuteWorkflow(mock_request_iterator(), mock_context)

        with pytest.raises(StopAsyncIteration):
            await anext(result)

        assert real_workflow_task is not None
        assert real_workflow_task.cancelled()

        mock_context.abort.assert_called_once_with(
            grpc.StatusCode.INTERNAL, "Something went wrong"
        )


@pytest.mark.asyncio
@patch("duo_workflow_service.servers.grpc_server.AbstractWorkflow")
@patch("duo_workflow_service.servers.grpc_server.resolve_workflow_class")
async def test_execute_workflow(mock_resolve_workflow, mock_abstract_workflow_class):
    mock_workflow_instance = mock_abstract_workflow_class.return_value
    mock_workflow_instance.is_done = False
    mock_workflow_instance.run = AsyncMock()
    mock_workflow_instance.cleanup = AsyncMock()
    mock_workflow_instance.get_from_outbox = AsyncMock(
        return_value=contract_pb2.Action()
    )
    mock_resolve_workflow.return_value = mock_abstract_workflow_class

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(goal="test")
        )

    current_user.set(CloudConnectorUser(authenticated=True, is_debug=True))
    mock_context = MagicMock(spec=grpc.ServicerContext)
    servicer = GrpcServer()
    result = servicer.ExecuteWorkflow(mock_request_iterator(), mock_context)
    assert isinstance(result, AsyncIterable)
    assert isinstance(await anext(result), contract_pb2.Action)


@pytest.mark.asyncio
@patch("duo_workflow_service.servers.grpc_server.TokenAuthority")
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
            issuer="gitlab.com", scopes=["duo_workflow_execute_workflow"]
        ),
    )
    current_user.set(user)

    servicer = GrpcServer()
    await servicer.GenerateToken(contract_pb2.GenerateTokenRequest(), mock_context)

    mock_token_authority.return_value.encode.assert_called_once_with(
        None, None, user, None, [GitLabUnitPrimitive.DUO_WORKFLOW_EXECUTE_WORKFLOW]
    )
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

    servicer = GrpcServer()
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
        "duo_workflow_service.servers.grpc_server.grpc.aio.server",
        return_value=mock_server,
    ), patch(
        "duo_workflow_service.servers.grpc_server.contract_pb2_grpc.add_DuoWorkflowServicer_to_server"
    ) as mock_add_servicer, patch(
        "duo_workflow_service.servers.grpc_server.reflection.enable_server_reflection"
    ) as mock_enable_reflection:
        await grpc_serve(50052)

    mock_server.add_insecure_port.assert_called_once_with("[::]:50052")
    mock_server.start.assert_called_once()
    mock_server.wait_for_termination.assert_called_once()
    mock_add_servicer.assert_called_once()
    mock_enable_reflection.assert_called_once()


@pytest.mark.asyncio
@patch("duo_workflow_service.servers.grpc_server.AbstractWorkflow")
@patch("duo_workflow_service.servers.grpc_server.resolve_workflow_class")
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

    current_user.set(CloudConnectorUser(authenticated=True, is_debug=True))
    mock_context = MagicMock(spec=grpc.ServicerContext)
    servicer = GrpcServer()
    result = servicer.ExecuteWorkflow(mock_request_iterator(), mock_context)
    with pytest.raises(StopAsyncIteration):
        await anext(result)

    mock_abstract_workflow_class.assert_called_once_with(
        workflow_id="123",
        workflow_metadata={},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.servers.grpc_server.AbstractWorkflow")
@patch("duo_workflow_service.servers.grpc_server.resolve_workflow_class")
async def test_execute_workflow_valid_workflow_metadata(
    mock_resolve_workflow, mock_abstract_workflow_class
):
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = True
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
    mock_resolve_workflow.return_value = mock_abstract_workflow_class

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(
                workflowID="123", workflowMetadata=json.dumps({"key": "value"})
            )
        )

    current_user.set(CloudConnectorUser(authenticated=True, is_debug=True))
    mock_context = MagicMock(spec=grpc.ServicerContext)
    servicer = GrpcServer()
    result = servicer.ExecuteWorkflow(mock_request_iterator(), mock_context)
    assert isinstance(result, AsyncIterable)
    with pytest.raises(StopAsyncIteration):
        await anext(result)

    mock_abstract_workflow_class.assert_called_once_with(
        workflow_id="123",
        workflow_metadata={"key": "value"},
        workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
    )


def test_string_to_category_enum():
    # Test a valid category string
    assert (
        string_to_category_enum("WORKFLOW_SOFTWARE_DEVELOPMENT")
        == CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT
    )

    # Test an invalid category string
    with patch("duo_workflow_service.servers.grpc_server.log") as mock_log:
        assert (
            string_to_category_enum("INVALID_CATEGORY")
            == CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT
        )
        mock_log.warning.assert_called_once_with(
            "Unknown category string: INVALID_CATEGORY"
        )
