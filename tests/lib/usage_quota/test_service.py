from unittest.mock import AsyncMock, patch

import pytest

from lib.events import GLReportingEventContext
from lib.internal_events.context import EventContext
from lib.usage_quota import InsufficientCredits, UsageQuotaService
from lib.usage_quota.errors import (
    UsageQuotaConnectionError,
    UsageQuotaHTTPError,
    UsageQuotaTimeoutError,
)
from lib.usage_quota.service import EventType


@pytest.fixture(name="internal_event_context")
def internal_event_context_fixture():
    return EventContext(
        environment="test",
        source="duo_chat",
        realm="saas",
        deployment_type="saas",
        instance_id="4398e2e6-012d-49d9-bada-d419458fe75f",
        unique_instance_id="4398e2e6-012d-49d9-bada-d419458fe75f",
        feature_enablement_type="duo_pro",
        host_name="gitlab.local",
        instance_version="17.5.0",
        global_user_id="user-123",
        user_id=None,
        project_id=789,
        ultimate_parent_namespace_id=456,
        namespace_id=456,
        correlation_id="12345",
    )


@pytest.fixture(autouse=True)
def mock_context(internal_event_context):
    with patch("lib.usage_quota.service.current_event_context") as mock_ctx:
        mock_ctx.get.return_value = internal_event_context
        yield


@pytest.fixture(name="service")
def service_fixture():
    return UsageQuotaService(
        customersdot_url="https://customers.gitlab.local/",
        customersdot_api_user="aigw@gitlab.local",
        customersdot_api_token="customersdot_api_token",
    )


@pytest.fixture(name="service_inputs")
def service_inputs_fixture():
    return GLReportingEventContext.from_workflow_definition(
        "duo_chat",
        has_flow_config=True,
    )


class TestQuotaEnforcement:
    """Tests for quota check enforcement."""

    @pytest.mark.asyncio
    async def test_authorized_request_continues(self, service, service_inputs):
        """Test that a request with available quota executes successfully."""
        with patch.object(
            service.usage_quota_client,
            "check_quota_available",
            new_callable=AsyncMock,
            return_value=True,
        ):
            # Should not raise any exception
            await service.execute(
                service_inputs, EventType.DUO_AGENT_PLATFORM_FLOW_ON_EXECUTE
            )

    @pytest.mark.asyncio
    async def test_no_usage_quota_left_raises_exception(self, service, service_inputs):
        """Test that a request without quota raises InsufficientCredits."""
        with patch.object(
            service.usage_quota_client,
            "check_quota_available",
            new_callable=AsyncMock,
            return_value=False,
        ):
            with pytest.raises(InsufficientCredits):
                await service.execute(
                    service_inputs, EventType.DUO_AGENT_PLATFORM_FLOW_ON_EXECUTE
                )


class TestFailOpenBehavior:
    """Tests for fail-open behavior on errors."""

    @pytest.mark.parametrize(
        "exception_class",
        [
            UsageQuotaTimeoutError,
            UsageQuotaHTTPError,
            UsageQuotaConnectionError,
        ],
    )
    @pytest.mark.asyncio
    async def test_fails_open_on_usage_quota_errors(
        self, service, service_inputs, exception_class
    ):
        """Test that service fails open on UsageQuotaError exceptions."""
        if exception_class == UsageQuotaHTTPError:
            exception = exception_class(status_code=500)
        else:
            exception = exception_class()

        with (
            patch.object(
                service.usage_quota_client,
                "check_quota_available",
                new_callable=AsyncMock,
                side_effect=exception,
            ),
            patch("lib.usage_quota.service.USAGE_QUOTA_CHECK_TOTAL") as mock_metrics,
        ):
            # Should not raise exception (fail-open behavior)
            await service.execute(
                service_inputs, EventType.DUO_AGENT_PLATFORM_FLOW_ON_EXECUTE
            )

        mock_metrics.labels.assert_called_once_with(result="fail_open", realm="saas")
        mock_metrics.labels.return_value.inc.assert_called_once()


class TestMetrics:
    """Tests for metrics recording."""

    @pytest.mark.asyncio
    async def test_records_allow_metric_on_quota_available(
        self, service, service_inputs
    ):
        """Test that 'allow' metric is recorded when quota is available."""
        with (
            patch.object(
                service.usage_quota_client,
                "check_quota_available",
                new_callable=AsyncMock,
                return_value=True,
            ),
            patch("lib.usage_quota.service.USAGE_QUOTA_CHECK_TOTAL") as mock_metrics,
        ):
            await service.execute(
                service_inputs, EventType.DUO_AGENT_PLATFORM_FLOW_ON_EXECUTE
            )

        mock_metrics.labels.assert_called_once_with(result="allow", realm="saas")
        mock_metrics.labels.return_value.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_records_deny_metric_on_quota_exhausted(
        self, service, service_inputs
    ):
        """Test that 'deny' metric is recorded when quota is exhausted."""
        with (
            patch.object(
                service.usage_quota_client,
                "check_quota_available",
                new_callable=AsyncMock,
                return_value=False,
            ),
            patch("lib.usage_quota.service.USAGE_QUOTA_CHECK_TOTAL") as mock_metrics,
        ):
            with pytest.raises(InsufficientCredits):
                await service.execute(
                    service_inputs, EventType.DUO_AGENT_PLATFORM_FLOW_ON_EXECUTE
                )

        mock_metrics.labels.assert_called_once_with(result="deny", realm="saas")
        mock_metrics.labels.return_value.inc.assert_called_once()
