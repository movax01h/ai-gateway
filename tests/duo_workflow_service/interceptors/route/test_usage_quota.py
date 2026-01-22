from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from grpc import StatusCode
from grpc.aio import ServicerContext

from contract import contract_pb2
from duo_workflow_service.interceptors.route.usage_quota import (
    has_sufficient_usage_quota,
)
from lib.events import FeatureQualifiedNameStatic, GLReportingEventContext
from lib.usage_quota import InsufficientCredits, UsageQuotaEvent
from lib.usage_quota.client import SKIP_USAGE_CUTOFF_CLAIM


@pytest.fixture(name="mock_user_with_skip_usage_cutoff")
def mock_user_with_skip_usage_cutoff_fixture():
    return CloudConnectorUser(
        authenticated=True,
        claims=UserClaims(extra={SKIP_USAGE_CUTOFF_CLAIM: True}),
    )


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


async def mock_request_generator() -> AsyncIterator[contract_pb2.ClientEvent]:
    """Helper to generate mock request stream."""
    yield contract_pb2.ClientEvent(
        startRequest=contract_pb2.StartWorkflowRequest(
            workflowDefinition="test_workflow"
        )
    )


@pytest.mark.asyncio
async def test_execute_workflow_with_sufficient_quota(mock_service) -> None:
    """Test ExecuteWorkflow with sufficient quota."""

    @has_sufficient_usage_quota(
        UsageQuotaEvent.DAP_FLOW_ON_EXECUTE, "https://customers.example.com"
    )
    async def ExecuteWorkflow(
        _self: Any,
        request: AsyncIterator[contract_pb2.ClientEvent],
        _context: ServicerContext,
    ) -> AsyncIterator[contract_pb2.ClientEvent]:
        async for item in request:
            yield item

    result = [
        item
        async for item in ExecuteWorkflow(None, mock_request_generator(), MagicMock())
    ]

    assert len(result) == 1
    mock_service.execute.assert_called_once()

    gl_context = mock_service.execute.call_args[0][0]
    assert isinstance(gl_context, GLReportingEventContext)
    assert gl_context.feature_qualified_name == "test_workflow"
    assert gl_context.value == "test_workflow"
    assert gl_context.feature_ai_catalog_item is False


@pytest.mark.asyncio
async def test_execute_workflow_with_insufficient_credits(
    mock_service, mock_context
) -> None:
    """Test ExecuteWorkflow with insufficient credits."""
    mock_service.execute.side_effect = InsufficientCredits("Insufficient credits")

    @has_sufficient_usage_quota(
        UsageQuotaEvent.DAP_FLOW_ON_EXECUTE, "https://customers.example.com"
    )
    async def ExecuteWorkflow(
        _self: Any,
        request: AsyncIterator[contract_pb2.ClientEvent],
        _context: ServicerContext,
    ) -> AsyncIterator[contract_pb2.ClientEvent]:
        async for item in request:
            yield item

    _ = [_ async for _ in ExecuteWorkflow(None, mock_request_generator(), mock_context)]

    mock_context.abort.assert_called_once_with(
        StatusCode.RESOURCE_EXHAUSTED,
        "Insufficient credits. Error code: USAGE_QUOTA_EXCEEDED",
    )


@pytest.mark.asyncio
async def test_generate_token_with_sufficient_quota(mock_service) -> None:
    """Test GenerateToken with sufficient quota."""

    @has_sufficient_usage_quota(
        UsageQuotaEvent.DAP_FLOW_ON_GENERATE_TOKEN,
        "https://customers.example.com",
    )
    async def GenerateToken(
        _self: Any, _request: Any, _context: ServicerContext
    ) -> str:
        return "success"

    result = await GenerateToken(None, MagicMock(), MagicMock())

    assert result == "success"
    mock_service.execute.assert_called_once()


@pytest.mark.asyncio
async def test_generate_token_with_insufficient_credits(
    mock_service, mock_context
) -> None:
    """Test GenerateToken with insufficient credits."""
    mock_service.execute.side_effect = InsufficientCredits("Insufficient credits")

    @has_sufficient_usage_quota(
        UsageQuotaEvent.DAP_FLOW_ON_GENERATE_TOKEN, "https://customers.example.com"
    )
    async def GenerateToken(
        _self: Any, _request: Any, _context: ServicerContext
    ) -> str:
        return "success"

    result = await GenerateToken(None, MagicMock(), mock_context)

    assert result is None
    mock_context.abort.assert_called_once_with(
        StatusCode.RESOURCE_EXHAUSTED,
        "Insufficient credits. Error code: USAGE_QUOTA_EXCEEDED",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "workflow_definition,expected_name,expected_catalog_item",
    [
        ("test_workflow", "test_workflow", None),
        ("", FeatureQualifiedNameStatic.DAP_FEATURE_LEGACY, None),
    ],
    ids=["with_workflow_definition", "legacy_without_workflow_definition"],
)
async def test_generate_token_workflow_definition_handling(
    mock_service, workflow_definition, expected_name, expected_catalog_item
) -> None:
    """Test GenerateToken with and without workflowDefinition."""

    @has_sufficient_usage_quota(
        UsageQuotaEvent.DAP_FLOW_ON_GENERATE_TOKEN, "https://customers.example.com"
    )
    async def GenerateToken(
        _self: Any,
        request: contract_pb2.GenerateTokenRequest,
        _context: ServicerContext,
    ) -> str:
        return "token_generated"

    request = contract_pb2.GenerateTokenRequest(workflowDefinition=workflow_definition)
    result = await GenerateToken(None, request, MagicMock())

    assert result == "token_generated"
    mock_service.execute.assert_called_once()

    gl_context = mock_service.execute.call_args[0][0]
    assert isinstance(gl_context, GLReportingEventContext)
    assert gl_context.feature_qualified_name == expected_name
    assert gl_context.value == expected_name
    assert gl_context.feature_ai_catalog_item is expected_catalog_item


def test_execute_workflow_with_wrong_event_type() -> None:
    """Test ExecuteWorkflow with wrong event type raises ValueError."""

    with pytest.raises(ValueError, match="Unsupported event type.*Expected to be"):

        @has_sufficient_usage_quota(
            UsageQuotaEvent.DAP_FLOW_ON_GENERATE_TOKEN, "https://customers.example.com"
        )
        async def ExecuteWorkflow(
            _self: Any,
            request: AsyncIterator[contract_pb2.ClientEvent],
            _context: ServicerContext,
        ) -> AsyncIterator[contract_pb2.ClientEvent]:
            async for item in request:
                yield item


def test_generate_token_with_wrong_event_type() -> None:
    """Test GenerateToken with wrong event type raises ValueError."""

    with pytest.raises(ValueError, match="Unsupported event type.*Expected to be"):

        @has_sufficient_usage_quota(
            UsageQuotaEvent.DAP_FLOW_ON_EXECUTE, "https://customers.example.com"
        )
        async def GenerateToken(
            _self: Any, _request: Any, _context: ServicerContext
        ) -> str:
            return "success"


def test_unsupported_method_raises_type_error() -> None:
    """Test that unsupported methods raise TypeError."""

    with pytest.raises(TypeError, match="unsupported method to intercept"):

        @has_sufficient_usage_quota(
            UsageQuotaEvent.DAP_FLOW_ON_EXECUTE, "https://customers.example.com"
        )
        async def unsupported_method(
            _self: Any, _request: Any, _context: ServicerContext
        ) -> AsyncIterator[str]:
            yield "item"
