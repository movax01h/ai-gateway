"""Shared fixtures for UsageQuotaClient tests."""

from typing import AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import pytest_asyncio

from lib.billing_events.context import UsageQuotaEventContext
from lib.usage_quota.client import UsageQuotaClient


@pytest_asyncio.fixture(name="usage_quota_client")
async def usage_quota_client_fixture() -> AsyncGenerator[UsageQuotaClient, None]:
    """Create a UsageQuotaClient instance for testing."""
    usage_quota_client = UsageQuotaClient(
        customersdot_url="https://customers.gitlab.local/",
        customersdot_api_user="aigw@gitlab.local",
        customersdot_api_token="customersdot_api_token",
        request_timeout=1.0,
    )

    await usage_quota_client.cache.clear()

    yield usage_quota_client

    await usage_quota_client.aclose()
    await usage_quota_client.cache.clear()


@pytest.fixture(name="usage_quota_context")
def usage_quota_context_fixture() -> UsageQuotaEventContext:
    """Create a UsageQuotaEventContext for testing."""
    return UsageQuotaEventContext(
        environment="production",
        realm="saas",
        deployment_type="saas",
        instance_id="00000000-1111-2222-3333-000000000000",
        unique_instance_id="00000000-1111-2222-3333-000000000000",
        feature_enablement_type="duo_pro",
        ultimate_parent_namespace_id=123,
        namespace_id=456,
        user_id="user_123",
        global_user_id="gid://gitlab/User/123",
        correlation_id="correlation_id",
    )


@pytest.fixture(name="mock_http_client")
def mock_http_client_fixture() -> AsyncMock:
    """Create a mock AsyncClient for testing."""
    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=None)
    return mock_client


def _make_response(status_code: int, cache_control: str | None = None) -> MagicMock:
    headers = httpx.Headers()
    if cache_control is not None:
        headers = httpx.Headers({"cache-control": cache_control})
    mock_response = MagicMock()
    mock_response.status_code = status_code
    mock_response.headers = headers
    if status_code < 400:
        mock_response.raise_for_status = MagicMock()
    return mock_response


@pytest.fixture(name="mock_success_response")
def mock_success_response_fixture() -> MagicMock:
    """Create a mock response for successful quota check (200 OK)."""
    return _make_response(200, "max-age=1800, private")


@pytest.fixture(name="mock_quota_exhausted_response")
def mock_quota_exhausted_response_fixture() -> MagicMock:
    """Create a mock response for exhausted quota (402 Payment Required)."""
    return _make_response(402, "max-age=300, private")


@pytest.fixture(name="mock_error_response")
def mock_error_response_fixture() -> MagicMock:
    """Create a mock response for HTTP errors."""
    return _make_response(500)
