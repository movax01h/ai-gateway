from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from grpc import StatusCode
from grpc.aio import ServicerContext

from contract import contract_pb2
from duo_workflow_service.interceptors.route.usage_quota import (
    has_sufficient_usage_quota,
)
from lib.usage_quota import EventType, InsufficientCredits


@pytest.fixture(name="mock_service")
def mock_service_fixture():
    with patch(
        "duo_workflow_service.interceptors.route.usage_quota.UsageQuotaService"
    ) as mock:
        service_instance = MagicMock()
        service_instance.execute = AsyncMock()
        mock.return_value = service_instance
        yield service_instance


@pytest.fixture(name="mock_context")
def mock_context_fixture():
    context = MagicMock()
    context.abort = AsyncMock()
    return context


@pytest.mark.asyncio
async def test_stream_decorator_success(mock_service) -> None:
    """Test stream decorator with sufficient quota."""

    @has_sufficient_usage_quota(
        EventType.DUO_AGENT_PLATFORM_FLOW_ON_EXECUTE,
        "https://customers.example.com",
    )
    async def ExecuteWorkflow(
        _self: Any,
        request: AsyncIterator[contract_pb2.ClientEvent],
        _context: ServicerContext,
    ) -> AsyncIterator[contract_pb2.ClientEvent]:
        async for item in request:
            yield item

    async def mock_request() -> AsyncIterator[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(
                workflowDefinition="test_workflow"
            )
        )

    result = []
    async for item in ExecuteWorkflow(None, mock_request(), MagicMock()):
        result.append(item)

    assert len(result) == 1
    mock_service.execute.assert_called_once()


@pytest.mark.asyncio
async def test_stream_decorator_insufficient_credits(
    mock_service, mock_context
) -> None:
    """Test stream decorator with insufficient credits."""
    mock_service.execute.side_effect = InsufficientCredits("Insufficient credits")

    @has_sufficient_usage_quota(
        EventType.DUO_AGENT_PLATFORM_FLOW_ON_EXECUTE,
        "https://customers.example.com",
    )
    async def ExecuteWorkflow(
        _self: Any,
        request: AsyncIterator[contract_pb2.ClientEvent],
        _context: ServicerContext,
    ) -> AsyncIterator[contract_pb2.ClientEvent]:
        async for item in request:
            yield item

    async def mock_request() -> AsyncIterator[contract_pb2.ClientEvent]:
        yield contract_pb2.ClientEvent(
            startRequest=contract_pb2.StartWorkflowRequest(
                workflowDefinition="test_workflow"
            )
        )

    result = []
    async for _ in ExecuteWorkflow(None, mock_request(), mock_context):
        result.append(_)

    mock_context.abort.assert_called_once_with(
        StatusCode.RESOURCE_EXHAUSTED,
        "Insufficient credits. Error code: USAGE_QUOTA_EXCEEDED",
    )


@pytest.mark.asyncio
async def test_unary_decorator_success(mock_service) -> None:
    """Test unary decorator with sufficient quota."""

    @has_sufficient_usage_quota(
        EventType.DUO_AGENT_PLATFORM_FLOW_ON_EXECUTE,
        "https://customers.example.com",
    )
    async def unary_method(_self: Any, _request: Any, _context: ServicerContext) -> str:
        return "success"

    result = await unary_method(None, MagicMock(), MagicMock())

    assert result == "success"
    mock_service.execute.assert_called_once()


@pytest.mark.asyncio
async def test_decorator_with_credentials(mock_service) -> None:
    """Test decorator with custom credentials."""

    @has_sufficient_usage_quota(
        EventType.DUO_AGENT_PLATFORM_FLOW_ON_EXECUTE,
        "https://customers.example.com",
        user="test_user",
        token="**********",
    )
    async def unary_method(_self: Any, _request: Any, _context: ServicerContext) -> str:
        return "success"

    await unary_method(None, MagicMock(), MagicMock())

    mock_service.execute.assert_called_once()


def test_unsupported_stream_method() -> None:
    """Test that unsupported stream methods raise TypeError."""

    with pytest.raises(TypeError, match="unsupported method to intercept"):

        @has_sufficient_usage_quota(
            EventType.DUO_AGENT_PLATFORM_FLOW_ON_EXECUTE,
            "https://customers.example.com",
        )
        async def unsupported_stream(
            _self: Any, _request: Any, _context: ServicerContext
        ) -> AsyncIterator[str]:
            yield "item"
