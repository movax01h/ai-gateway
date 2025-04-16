import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import AsyncIterable
from unittest.mock import ANY, AsyncMock, MagicMock, call, patch

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
from duo_workflow_service.server import DuoWorkflowService, run, serve


@pytest.mark.asyncio
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.Registry")
async def test_execute_workflow_when_no_events_ends(
    mock_registry_class, mock_abstract_workflow_class
):
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = True
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
    mock_registry_class.resolve.return_value = mock_abstract_workflow_class

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
        await result.__anext__()


@pytest.mark.asyncio
@patch("asyncio.sleep")
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.Registry")
async def test_execute_workflow_when_nothing_in_outbox(
    mock_registry_class, mock_abstract_workflow_class, mock_sleep
):
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = False
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
    mock_registry_class.resolve.return_value = mock_abstract_workflow_class

    def side_effect():
        mock_workflow.is_done = True
        raise asyncio.QueueEmpty()

    mock_workflow.get_from_outbox = MagicMock(side_effect=side_effect)

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
        await result.__anext__()
    mock_sleep.assert_called_once_with(DuoWorkflowService.OUTBOX_CHECK_INTERVAL)


@pytest.mark.asyncio
@patch("asyncio.create_task")
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.Registry")
async def test_workflow_is_cancelled_on_parent_task_cancellation(
    mock_registry_class, mock_abstract_workflow_class, mock_task
):
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = False
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
    mock_registry_class.resolve.return_value = mock_abstract_workflow_class

    def side_effect():
        raise asyncio.CancelledError()

    mock_workflow.get_from_outbox = MagicMock(side_effect=side_effect)

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(goal="test")
        )

    current_user.set(CloudConnectorUser(authenticated=True, is_debug=True))
    mock_context = MagicMock(spec=grpc.ServicerContext)
    mock_context.abort = AsyncMock()
    result = DuoWorkflowService().ExecuteWorkflow(mock_request_iterator(), mock_context)
    with pytest.raises(StopAsyncIteration):
        await result.__anext__()
    assert mock_task.mock_calls[-1] == call().cancel(ANY)

    mock_context.abort.assert_called_once_with(
        grpc.StatusCode.INTERNAL, "Something went wrong"
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.Registry")
async def test_execute_workflow(mock_registry_class, mock_abstract_workflow_class):
    mock_workflow_instance = mock_abstract_workflow_class.return_value
    mock_workflow_instance.is_done = False
    mock_workflow_instance.run = AsyncMock()
    mock_workflow_instance.cleanup = AsyncMock()
    mock_workflow_instance.get_from_outbox.return_value = contract_pb2.Action()
    mock_registry_class.resolve.return_value = mock_abstract_workflow_class

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(goal="test")
        )

    current_user.set(CloudConnectorUser(authenticated=True, is_debug=True))
    mock_context = MagicMock(spec=grpc.ServicerContext)
    servicer = DuoWorkflowService()
    result = servicer.ExecuteWorkflow(mock_request_iterator(), mock_context)
    assert isinstance(result, AsyncIterable)
    assert isinstance(await result.__anext__(), contract_pb2.Action)


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
            issuer="gitlab.com", scopes=["duo_workflow_execute_workflow"]
        ),
    )
    current_user.set(user)

    servicer = DuoWorkflowService()
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

    servicer = DuoWorkflowService()
    with pytest.raises(grpc.RpcError):
        await servicer.GenerateToken(contract_pb2.GenerateTokenRequest(), mock_context)

    mock_context.abort.assert_called_once_with(
        grpc.StatusCode.PERMISSION_DENIED, "Unauthorized to generate token"
    )


@pytest.mark.asyncio
async def test_serve():
    mock_server = AsyncMock()
    mock_server.add_insecure_port.return_value = None
    mock_server.start.return_value = None
    mock_server.wait_for_termination.return_value = None

    with patch("duo_workflow_service.server.grpc.aio.server", return_value=mock_server):
        await serve(50052)

    mock_server.add_insecure_port.assert_called_once_with("[::]:50052")
    mock_server.start.assert_called_once()
    mock_server.wait_for_termination.assert_called_once()


def test_run(monkeypatch):
    server_port = "1234"

    monkeypatch.setenv("PORT", server_port)

    with patch(
        "duo_workflow_service.server.setup_monitoring"
    ) as mock_monitoring, patch(
        "duo_workflow_service.server.validate_llm_access"
    ) as mock_validate, patch(
        "duo_workflow_service.server.serve"
    ) as mock_serve:
        run()

    mock_monitoring.assert_called_once()
    mock_validate.assert_called_once()
    mock_serve.assert_called_once_with(int(server_port))


def test_run_without_llm_access():
    with patch("duo_workflow_service.server.setup_monitoring"), patch(
        "duo_workflow_service.server.validate_llm_access",
        side_effect=RuntimeError("error"),
    ), patch("duo_workflow_service.server.serve") as mock_serve:
        with pytest.raises(RuntimeError):
            run()

    mock_serve.assert_not_called()


@pytest.mark.asyncio
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.Registry")
async def test_execute_workflow_missing_workflow_metadata(
    mock_registry_class, mock_abstract_workflow_class
):
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = True
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
    mock_registry_class.resolve.return_value = mock_abstract_workflow_class

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(workflowID="123")
        )

    current_user.set(CloudConnectorUser(authenticated=True, is_debug=True))
    mock_context = MagicMock(spec=grpc.ServicerContext)
    servicer = DuoWorkflowService()
    result = servicer.ExecuteWorkflow(mock_request_iterator(), mock_context)
    with pytest.raises(StopAsyncIteration):
        await result.__anext__()

    mock_abstract_workflow_class.assert_called_once_with(
        workflow_id="123",
        workflow_metadata={},
    )


@pytest.mark.asyncio
@patch("duo_workflow_service.server.AbstractWorkflow")
@patch("duo_workflow_service.server.Registry")
async def test_execute_workflow_valid_workflow_metadata(
    mock_registry_class, mock_abstract_workflow_class
):
    mock_workflow = mock_abstract_workflow_class.return_value
    mock_workflow.is_done = True
    mock_workflow.run = AsyncMock()
    mock_workflow.cleanup = AsyncMock()
    mock_registry_class.resolve.return_value = mock_abstract_workflow_class

    async def mock_request_iterator() -> AsyncIterable[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(
                workflowID="123", workflowMetadata=json.dumps({"key": "value"})
            )
        )

    current_user.set(CloudConnectorUser(authenticated=True, is_debug=True))
    mock_context = MagicMock(spec=grpc.ServicerContext)
    servicer = DuoWorkflowService()
    result = servicer.ExecuteWorkflow(mock_request_iterator(), mock_context)
    assert isinstance(result, AsyncIterable)
    with pytest.raises(StopAsyncIteration):
        await result.__anext__()

    mock_abstract_workflow_class.assert_called_once_with(
        workflow_id="123",
        workflow_metadata={"key": "value"},
    )
