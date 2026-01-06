from unittest.mock import AsyncMock, patch

import pytest

from lib.billing_events.context import UsageQuotaEventContext
from lib.events.base import GLReportingEventContext
from lib.internal_events.context import EventContext
from lib.usage_quota import EventType, InsufficientCredits, UsageQuotaService
from lib.usage_quota.errors import (
    UsageQuotaConnectionError,
    UsageQuotaHTTPError,
    UsageQuotaTimeoutError,
)


@pytest.fixture(name="internal_event_context")
def internal_event_context_fixture():
    """Create a mock internal event context."""
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
def mock_current_event_context(internal_event_context):
    """Mock the current_event_context for all tests."""
    with patch("lib.usage_quota.service.current_event_context") as mock_ctx:
        mock_ctx.get.return_value = internal_event_context
        yield


@pytest.fixture(name="service")
def service_fixture():
    """Create a UsageQuotaService instance for testing."""
    return UsageQuotaService(
        customersdot_url="https://customers.gitlab.local/",
        customersdot_api_user="aigw@gitlab.local",
        customersdot_api_token="test_token",
    )


@pytest.fixture(name="gl_context")
def gl_context_fixture():
    """Create a GLReportingEventContext for testing."""
    return GLReportingEventContext.from_workflow_definition("code_suggestions")


class TestServiceInitialization:
    """Tests for UsageQuotaService initialization."""

    def test_initializes_with_credentials(self):
        """Test that service initializes with provided credentials."""
        service = UsageQuotaService(
            customersdot_url="https://customers.gitlab.local/",
            customersdot_api_user="test_user",
            customersdot_api_token="test_token",
        )

        assert (
            service.usage_quota_client.customersdot_url
            == "https://customers.gitlab.local/"
        )
        assert service.usage_quota_client.customersdot_api_user == "test_user"
        assert service.usage_quota_client.customersdot_api_token == "test_token"
        assert service.usage_quota_client.enabled is True

    def test_initializes_without_credentials(self):
        """Test that service initializes without credentials (disabled)."""
        service = UsageQuotaService(
            customersdot_url="https://customers.gitlab.local/",
            customersdot_api_user=None,
            customersdot_api_token=None,
        )

        assert service.usage_quota_client.enabled is False


class TestQuotaEnforcement:
    """Tests for quota check enforcement."""

    @pytest.mark.asyncio
    async def test_skips_check_when_disabled(self, gl_context):
        """Test that quota check is skipped when client is disabled."""
        service = UsageQuotaService(
            customersdot_url="https://customers.gitlab.local/",
            customersdot_api_user=None,
            customersdot_api_token=None,
        )

        # Should not raise any exception and not call check_quota_available
        await service.execute(gl_context, EventType.CODE_SUGGESTIONS_CODE_COMPLETIONS)

    @pytest.mark.asyncio
    async def test_authorized_request_continues(self, service, gl_context):
        """Test that a request with available quota executes successfully."""
        with patch.object(
            service.usage_quota_client,
            "check_quota_available",
            new_callable=AsyncMock,
            return_value=True,
        ):
            # Should not raise any exception
            await service.execute(
                gl_context, EventType.CODE_SUGGESTIONS_CODE_COMPLETIONS
            )

    @pytest.mark.asyncio
    async def test_insufficient_quota_raises_exception(self, service, gl_context):
        """Test that a request without quota raises InsufficientCredits."""
        with patch.object(
            service.usage_quota_client,
            "check_quota_available",
            new_callable=AsyncMock,
            return_value=False,
        ):
            with pytest.raises(InsufficientCredits):
                await service.execute(
                    gl_context, EventType.CODE_SUGGESTIONS_CODE_COMPLETIONS
                )


class TestEventTypeHandling:
    """Tests for different event types."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "event_type",
        [
            EventType.CODE_SUGGESTIONS_CODE_COMPLETIONS,
            EventType.CODE_SUGGESTIONS_CODE_GENERATIONS,
            EventType.DUO_CHAT_CLASSIC,
            EventType.DUO_AGENT_PLATFORM_FLOW_ON_EXECUTE,
            EventType.AMAZON_Q_INTEGRATION,
            EventType.AI_GATEWAY_PROXY_USE,
        ],
    )
    async def test_handles_all_event_types(self, service, gl_context, event_type):
        """Test that service handles all EventType values."""
        with patch.object(
            service.usage_quota_client,
            "check_quota_available",
            new_callable=AsyncMock,
            return_value=True,
        ):
            await service.execute(gl_context, event_type)


class TestContextExtension:
    """Tests for context extension and passing to client."""

    @pytest.mark.asyncio
    async def test_extends_context_with_gl_attributes(self, service, gl_context):
        """Test that service extends context with GLReportingEventContext attributes."""
        with patch.object(
            service.usage_quota_client,
            "check_quota_available",
            new_callable=AsyncMock,
            return_value=True,
        ) as mock_check:
            await service.execute(
                gl_context, EventType.CODE_SUGGESTIONS_CODE_COMPLETIONS
            )

            # Verify check_quota_available was called with extended context
            call_args = mock_check.call_args
            extended_context = call_args[0][0]

            assert isinstance(extended_context, UsageQuotaEventContext)
            assert (
                extended_context.event_type
                == EventType.CODE_SUGGESTIONS_CODE_COMPLETIONS.value
            )
            assert (
                extended_context.feature_qualified_name
                == gl_context.feature_qualified_name
            )
            assert (
                extended_context.feature_ai_catalog_item
                == gl_context.feature_ai_catalog_item
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
        self, service, gl_context, exception_class
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
                gl_context, EventType.CODE_SUGGESTIONS_CODE_COMPLETIONS
            )

        mock_metrics.labels.assert_called_once_with(result="fail_open", realm="saas")
        mock_metrics.labels.return_value.inc.assert_called_once()


class TestMetrics:
    """Tests for metrics recording."""

    @pytest.mark.asyncio
    async def test_records_allow_metric_on_quota_available(self, service, gl_context):
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
                gl_context, EventType.CODE_SUGGESTIONS_CODE_COMPLETIONS
            )

        mock_metrics.labels.assert_called_once_with(result="allow", realm="saas")
        mock_metrics.labels.return_value.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_records_deny_metric_on_quota_exhausted(self, service, gl_context):
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
                    gl_context, EventType.CODE_SUGGESTIONS_CODE_COMPLETIONS
                )

        mock_metrics.labels.assert_called_once_with(result="deny", realm="saas")
        mock_metrics.labels.return_value.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_records_fail_open_metric_on_error(self, service, gl_context):
        """Test that 'fail_open' metric is recorded on service errors."""
        with (
            patch.object(
                service.usage_quota_client,
                "check_quota_available",
                new_callable=AsyncMock,
                side_effect=UsageQuotaTimeoutError(),
            ),
            patch("lib.usage_quota.service.USAGE_QUOTA_CHECK_TOTAL") as mock_metrics,
        ):
            await service.execute(
                gl_context, EventType.CODE_SUGGESTIONS_CODE_COMPLETIONS
            )

        mock_metrics.labels.assert_called_once_with(result="fail_open", realm="saas")
        mock_metrics.labels.return_value.inc.assert_called_once()


class TestInsufficientCreditsException:
    """Tests for InsufficientCredits exception."""

    def test_default_message(self):
        """Test that InsufficientCredits has default message."""
        exc = InsufficientCredits()
        assert "sufficient credits" in str(exc)

    def test_custom_message(self):
        """Test that InsufficientCredits accepts custom message."""
        custom_msg = "Custom error message"
        exc = InsufficientCredits(custom_msg)
        assert str(exc) == custom_msg

    def test_is_exception(self):
        """Test that InsufficientCredits is an Exception."""
        exc = InsufficientCredits()
        assert isinstance(exc, Exception)
