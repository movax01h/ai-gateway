"""Unit tests for UsageQuotaClient."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims

from lib.billing_events.context import UsageQuotaEventContext
from lib.context import StarletteUser
from lib.usage_quota.client import (
    JITTER_FACTOR,
    SKIP_USAGE_CUTOFF_CLAIM,
    UsageQuotaClient,
    _apply_jitter,
    _parse_max_age,
    should_skip_usage_quota_for_user,
)
from lib.usage_quota.errors import (
    UsageQuotaConnectionError,
    UsageQuotaHTTPError,
    UsageQuotaTimeoutError,
)
from tests.lib.usage_quota.conftest import _make_response


@pytest.mark.parametrize(
    "user,expected",
    [
        # Case 1: User is None - should return False
        (None, False),
        # Case 2: User with no claims - should return False
        (CloudConnectorUser(authenticated=True), False),
        # Case 3: User with claims but no extra - should return False
        (CloudConnectorUser(authenticated=True, claims=UserClaims()), False),
        # Case 4: User with claims and extra but without skip_usage_cutoff - should return False
        (
            CloudConnectorUser(
                authenticated=True,
                claims=UserClaims(extra={"other_claim": "value"}),
            ),
            False,
        ),
        # Case 5: User with skip_usage_cutoff claim set to True - should return True
        (
            CloudConnectorUser(
                authenticated=True,
                claims=UserClaims(extra={SKIP_USAGE_CUTOFF_CLAIM: True}),
            ),
            True,
        ),
        # Case 6: User with skip_usage_cutoff claim set to False - should return False
        (
            CloudConnectorUser(
                authenticated=True,
                claims=UserClaims(extra={SKIP_USAGE_CUTOFF_CLAIM: False}),
            ),
            False,
        ),
        # Case 7: User with skip_usage_cutoff claim set to None - should return False
        (
            CloudConnectorUser(
                authenticated=True,
                claims=UserClaims(extra={SKIP_USAGE_CUTOFF_CLAIM: None}),
            ),
            False,
        ),
        # Case 8: User with skip_usage_cutoff claim set to 0 - should return False
        (
            CloudConnectorUser(
                authenticated=True,
                claims=UserClaims(extra={SKIP_USAGE_CUTOFF_CLAIM: 0}),
            ),
            False,
        ),
        # Case 9: User with skip_usage_cutoff claim set to empty string - should return False
        (
            CloudConnectorUser(
                authenticated=True,
                claims=UserClaims(extra={SKIP_USAGE_CUTOFF_CLAIM: ""}),
            ),
            False,
        ),
        # Case 10: StarletteUser with skip_usage_cutoff claim set to True - should return True
        (
            StarletteUser(
                CloudConnectorUser(
                    authenticated=True,
                    claims=UserClaims(extra={SKIP_USAGE_CUTOFF_CLAIM: True}),
                )
            ),
            True,
        ),
        # Case 11: StarletteUser without skip_usage_cutoff claim - should return False
        (
            StarletteUser(CloudConnectorUser(authenticated=True)),
            False,
        ),
        # Case 12: StarletteUser with skip_usage_cutoff claim set to False - should return False
        (
            StarletteUser(
                CloudConnectorUser(
                    authenticated=True,
                    claims=UserClaims(extra={SKIP_USAGE_CUTOFF_CLAIM: False}),
                )
            ),
            False,
        ),
    ],
)
def test_should_skip_usage_quota_for_user(user, expected):
    """Test should_skip_usage_quota_for_user with various user configurations.

    Tests that the function correctly identifies users who should skip usage quota checks based on:
    1. The presence of the skip_usage_cutoff claim in their extra claims
    2. The claim having a truthy value (not False, None, 0, empty string, etc.)
    """
    result = should_skip_usage_quota_for_user(user)
    if expected:
        assert result is True
    else:
        assert not result


class TestCheckQuotaAvailable:
    """Tests for check_quota_available method."""

    @pytest.mark.asyncio
    async def test_returns_true_on_200_response(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
        mock_success_response: MagicMock,
    ):
        """Should return True when CustomersDot returns 200 OK."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=mock_success_response)
            mock_client_class.return_value = mock_http_client

            result = await usage_quota_client.check_quota_available(usage_quota_context)

            assert result is True

    @pytest.mark.asyncio
    async def test_returns_false_on_402_response(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
        mock_quota_exhausted_response: MagicMock,
    ):
        """Should return False when CustomersDot returns 402 Payment Required."""

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(
                return_value=mock_quota_exhausted_response
            )
            mock_client_class.return_value = mock_http_client

            result = await usage_quota_client.check_quota_available(usage_quota_context)

            assert result is False

    @pytest.mark.asyncio
    async def test_raises_timeout_error_on_timeout(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
    ):
        """Should raise UsageQuotaTimeoutError when request times out."""

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(
                side_effect=httpx.TimeoutException("Request timed out")
            )
            mock_client_class.return_value = mock_http_client

            with pytest.raises(UsageQuotaTimeoutError):
                await usage_quota_client.check_quota_available(usage_quota_context)

    @pytest.mark.asyncio
    async def test_raises_http_error_on_unexpected_status(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
        mock_error_response,
    ):
        """Should raise UsageQuotaHTTPError on unexpected HTTP status codes."""

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=mock_error_response)
            mock_http_client.head.return_value.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "Server error",
                    request=MagicMock(),
                    response=mock_error_response,
                )
            )
            mock_client_class.return_value = mock_http_client

            with pytest.raises(UsageQuotaHTTPError):
                await usage_quota_client.check_quota_available(usage_quota_context)

    @pytest.mark.asyncio
    async def test_raises_connection_error_on_request_error(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
    ):
        """Should raise UsageQuotaConnectionError on connection failures."""

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(
                side_effect=httpx.RequestError("Connection failed")
            )
            mock_client_class.return_value = mock_http_client

            with pytest.raises(UsageQuotaConnectionError):
                await usage_quota_client.check_quota_available(usage_quota_context)

    @pytest.mark.asyncio
    async def test_sends_head_request_to_correct_endpoint(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
        mock_success_response,
    ):
        """Should send HEAD request to /api/v1/consumers/resolve endpoint."""

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=mock_success_response)
            mock_client_class.return_value = mock_http_client

            await usage_quota_client.check_quota_available(usage_quota_context)

            mock_http_client.head.assert_called_once()
            call_args = mock_http_client.head.call_args
            assert "api/v1/consumers/resolve" in call_args[0][0]

    @pytest.mark.asyncio
    async def test_passes_context_as_query_parameters(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
        mock_success_response,
    ):
        """Should pass context fields as query parameters."""

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=mock_success_response)
            mock_client_class.return_value = mock_http_client

            await usage_quota_client.check_quota_available(usage_quota_context)

            call_args = mock_http_client.head.call_args
            assert "params" in call_args[1]
            params = call_args[1]["params"]
            assert params["realm"] == "saas"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "feature_ai_catalog_item_value",
        [True, False, None],
    )
    async def test_always_includes_feature_ai_catalog_item(
        self,
        usage_quota_client: UsageQuotaClient,
        mock_http_client: AsyncMock,
        mock_success_response: MagicMock,
        feature_ai_catalog_item_value: bool | None,
    ):
        """Should always include feature_ai_catalog_item in params regardless of value.

        This is important because None indicates we were unable to resolve the value when processing legacy logic, and
        CustomersDot needs to know this.
        """
        context = UsageQuotaEventContext(
            environment="production",
            realm="saas",
            feature_ai_catalog_item=feature_ai_catalog_item_value,
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=mock_success_response)
            mock_client_class.return_value = mock_http_client

            await usage_quota_client.check_quota_available(context)

            call_args = mock_http_client.head.call_args
            params = call_args[1]["params"]
            assert "feature_ai_catalog_item" in params
            assert params["feature_ai_catalog_item"] is feature_ai_catalog_item_value

    @pytest.mark.asyncio
    async def test_uses_configured_timeout(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
        mock_success_response,
    ):
        """Should use the configured request timeout."""

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=mock_success_response)
            mock_client_class.return_value = mock_http_client

            await usage_quota_client.check_quota_available(usage_quota_context)

            call_args = mock_client_class.call_args
            timeout_arg = call_args[1]["timeout"]
            assert timeout_arg.read == 1.0
            assert timeout_arg.write == 1.0
            assert timeout_arg.connect == 1.0

    @pytest.mark.asyncio
    async def test_handles_minimal_context(
        self,
        usage_quota_client: UsageQuotaClient,
        mock_http_client: AsyncMock,
        mock_success_response: MagicMock,
    ):
        """Should handle context with only required fields."""
        minimal_context = UsageQuotaEventContext()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=mock_success_response)
            mock_client_class.return_value = mock_http_client

            result = await usage_quota_client.check_quota_available(minimal_context)

            assert result is True

    @pytest.mark.asyncio
    async def test_returns_true_when_client_disabled(
        self, usage_quota_context: UsageQuotaEventContext
    ):
        """Should return True without making HTTP request when client is disabled."""
        disabled_client = UsageQuotaClient(
            customersdot_url="https://customers.gitlab.local/",
            customersdot_api_user=None,
            customersdot_api_token=None,
            request_timeout=1.0,
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            result = await disabled_client.check_quota_available(usage_quota_context)

            assert result is True
            mock_client_class.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_authorization_headers(
        self,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
        mock_success_response: MagicMock,
    ):
        """Should send X-Admin-Email and X-Admin-Token headers."""
        client = UsageQuotaClient(
            customersdot_url="https://customers.gitlab.local/",
            customersdot_api_user="aigw@gitlab.local",
            customersdot_api_token="customersdot_api_token",
            request_timeout=1.0,
        )

        await client.cache.clear()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=mock_success_response)
            mock_client_class.return_value = mock_http_client

            await client.check_quota_available(usage_quota_context)

            call_args = mock_http_client.head.call_args
            assert "headers" in call_args[1]
            headers = call_args[1]["headers"]
            assert headers["X-Admin-Email"] == "aigw@gitlab.local"
            assert headers["X-Admin-Token"] == "customersdot_api_token"

    @pytest.mark.asyncio
    async def test_url_joining_with_trailing_slash(
        self,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
        mock_success_response: MagicMock,
    ):
        """Should correctly join URL with trailing slash."""
        client = UsageQuotaClient(
            customersdot_url="https://customers.gitlab.local/",
            customersdot_api_user="aigw@gitlab.local",
            customersdot_api_token="customersdot_api_token",
            request_timeout=1.0,
        )

        await client.cache.clear()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=mock_success_response)
            mock_client_class.return_value = mock_http_client

            await client.check_quota_available(usage_quota_context)

            call_args = mock_http_client.head.call_args
            url = call_args[0][0]
            assert url == "https://customers.gitlab.local/api/v1/consumers/resolve"

    @pytest.mark.asyncio
    async def test_url_joining_without_trailing_slash(
        self,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
        mock_success_response: MagicMock,
    ):
        """Should correctly join URL without trailing slash."""
        client = UsageQuotaClient(
            customersdot_url="https://customers.gitlab.local",
            customersdot_api_user="aigw@gitlab.local",
            customersdot_api_token="customersdot_api_token",
            request_timeout=1.0,
        )

        await client.cache.clear()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=mock_success_response)
            mock_client_class.return_value = mock_http_client

            await client.check_quota_available(usage_quota_context)

            call_args = mock_http_client.head.call_args
            url = call_args[0][0]
            assert url == "https://customers.gitlab.local/api/v1/consumers/resolve"


class TestClientInitialization:
    """Tests for UsageQuotaClient initialization."""

    def test_initializes_with_default_timeout(self):
        """Should use default timeout of 1.0 seconds."""
        client = UsageQuotaClient(
            customersdot_url="https://customers.gitlab.local/",
            customersdot_api_user="aigw@gitlab.local",
            customersdot_api_token="customersdot_api_token",
        )
        assert client.request_timeout == 1.0

    def test_initializes_with_custom_timeout(self):
        """Should accept custom timeout value."""
        client = UsageQuotaClient(
            customersdot_url="https://customers.gitlab.local/",
            customersdot_api_user="aigw@gitlab.local",
            customersdot_api_token="customersdot_api_token",
            request_timeout=5.0,
        )
        assert client.request_timeout == 5.0

    def test_stores_customersdot_url(self):
        """Should store the CustomersDot URL."""
        url = "https://customers.gitlab.local/"
        client = UsageQuotaClient(
            customersdot_url=url,
            customersdot_api_user="aigw@gitlab.local",
            customersdot_api_token="customersdot_api_token",
        )
        assert client.customersdot_url == url

    def test_raises_value_error_customersdot_url_empty(self):
        """Should raise ValueError if the CustomersDot URL is an empty string."""
        with pytest.raises(ValueError):
            UsageQuotaClient(
                customersdot_url="   ",
                customersdot_api_user="aigw@gitlab.local",
                customersdot_api_token="customersdot_api_token",
            )

    def test_enabled_true_when_api_token_and_user_provided(self):
        """Should set enabled to True when both API token and user are provided."""
        client = UsageQuotaClient(
            customersdot_url="https://customers.gitlab.local/",
            customersdot_api_user="aigw@gitlab.local",
            customersdot_api_token="customersdot_api_token",
        )
        assert client.enabled is True

    def test_enabled_false_when_api_token_is_none(self):
        """Should set enabled to False when API token is None."""
        client = UsageQuotaClient(
            customersdot_url="https://customers.gitlab.local/",
            customersdot_api_user="aigw@gitlab.local",
            customersdot_api_token=None,
        )
        assert client.enabled is False

    def test_enabled_false_when_api_user_is_none(self):
        """Should set enabled to False when API user is None."""
        client = UsageQuotaClient(
            customersdot_url="https://customers.gitlab.local/",
            customersdot_api_user=None,
            customersdot_api_token="customersdot_api_token",
        )
        assert client.enabled is False

    def test_enabled_false_when_both_api_credentials_are_none(self):
        """Should set enabled to False when both API user and token are None."""
        client = UsageQuotaClient(
            customersdot_url="https://customers.gitlab.local/",
            customersdot_api_user=None,
            customersdot_api_token=None,
        )
        assert client.enabled is False

    def test_stores_customersdot_api_user(self):
        """Should store the CustomersDot API user."""
        client = UsageQuotaClient(
            customersdot_url="https://customers.gitlab.local/",
            customersdot_api_user="aigw@gitlab.local",
            customersdot_api_token="customersdot_api_token",
        )
        assert client.customersdot_api_user == "aigw@gitlab.local"

    def test_stores_customersdot_api_token(self):
        """Should store the CustomersDot API token."""
        client = UsageQuotaClient(
            customersdot_url="https://customers.gitlab.local/",
            customersdot_api_user="aigw@gitlab.local",
            customersdot_api_token="customersdot_api_token",
        )
        assert client.customersdot_api_token == "customersdot_api_token"


class TestErrorHandling:
    """Tests for error handling and exception details."""

    @pytest.mark.asyncio
    async def test_timeout_error_preserves_original_exception(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
    ):
        """Should preserve the original exception in UsageQuotaTimeoutError."""
        original_error = httpx.TimeoutException("Original timeout")
        mock_http_client = AsyncMock()
        mock_http_client.__aenter__ = AsyncMock(return_value=mock_http_client)
        mock_http_client.__aexit__ = AsyncMock(return_value=None)
        mock_http_client.head = AsyncMock(side_effect=original_error)

        with patch("httpx.AsyncClient", return_value=mock_http_client):
            with pytest.raises(UsageQuotaTimeoutError) as exc_info:
                await usage_quota_client.check_quota_available(usage_quota_context)

            assert exc_info.value.original_error is original_error

    @pytest.mark.asyncio
    async def test_http_error_includes_status_code(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
        mock_error_response: MagicMock,
    ):
        """Should include status code in UsageQuotaHTTPError."""

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=mock_error_response)
            mock_http_client.head.return_value.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "Service unavailable",
                    request=MagicMock(),
                    response=mock_error_response,
                )
            )
            mock_client_class.return_value = mock_http_client

            with pytest.raises(UsageQuotaHTTPError) as exc_info:
                await usage_quota_client.check_quota_available(usage_quota_context)

            assert exc_info.value.status_code == 500

    @pytest.mark.asyncio
    async def test_connection_error_preserves_original_exception(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
    ):
        """Should preserve the original exception in UsageQuotaConnectionError."""
        original_error = httpx.RequestError("Network unreachable")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(side_effect=original_error)
            mock_client_class.return_value = mock_http_client

            with pytest.raises(UsageQuotaConnectionError) as exc_info:
                await usage_quota_client.check_quota_available(usage_quota_context)

            assert exc_info.value.original_error is original_error


class TestConnectionPoolConfiguration:
    """Tests for HTTP connection pool settings."""

    @pytest.mark.asyncio
    async def test_configures_connection_limits(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: MagicMock,
        mock_success_response: MagicMock,
    ):
        """Should configure connection pool limits."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=mock_success_response)
            mock_client_class.return_value = mock_http_client

            await usage_quota_client.check_quota_available(usage_quota_context)

            call_args = mock_client_class.call_args
            limits = call_args[1]["limits"]
            assert limits.max_keepalive_connections == 20
            assert limits.max_connections == 100

    @pytest.mark.asyncio
    async def test_aclose_cleanup(self):
        """Should properly cleanup HTTP client resources."""
        client = UsageQuotaClient(
            customersdot_url="https://customers.gitlab.local/",
            customersdot_api_user="aigw@gitlab.local",
            customersdot_api_token="customersdot_api_token",
            request_timeout=1.0,
        )

        # Initialize client by calling _get_client
        http_client = await client._get_client()
        assert client._http_client is not None
        assert client._http_client is http_client

        # Cleanup
        await client.aclose()
        assert client._http_client is None

    @pytest.mark.asyncio
    async def test_aclose_when_client_not_initialized(self):
        """Should handle aclose gracefully when client never initialized."""
        client = UsageQuotaClient(
            customersdot_url="https://customers.gitlab.local/",
            customersdot_api_user="aigw@gitlab.local",
            customersdot_api_token="customersdot_api_token",
            request_timeout=1.0,
        )

        # Client never initialized
        assert client._http_client is None

        # Should not raise any errors
        await client.aclose()
        assert client._http_client is None


class TestModelIdParameter:
    """Tests for model_id parameter in usage quota context."""

    @pytest.mark.asyncio
    async def test_includes_model_id_in_request_params(
        self,
        usage_quota_client: UsageQuotaClient,
        mock_http_client: AsyncMock,
        mock_success_response: MagicMock,
    ):
        """Should include model_id in query parameters when provided."""
        context = UsageQuotaEventContext(
            environment="production",
            realm="saas",
            model_id="claude-3-5-sonnet-20241022",
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=mock_success_response)
            mock_client_class.return_value = mock_http_client

            await usage_quota_client.check_quota_available(context)

            call_args = mock_http_client.head.call_args
            params = call_args[1]["params"]
            assert "model_id" in params
            assert params["model_id"] == "claude-3-5-sonnet-20241022"

    @pytest.mark.asyncio
    async def test_excludes_model_id_when_none(
        self,
        usage_quota_client: UsageQuotaClient,
        mock_http_client: AsyncMock,
        mock_success_response: MagicMock,
    ):
        """Should exclude model_id from params when None."""
        context = UsageQuotaEventContext(
            environment="production",
            realm="saas",
            model_id=None,
        )

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=mock_success_response)
            mock_client_class.return_value = mock_http_client

            await usage_quota_client.check_quota_available(context)

            call_args = mock_http_client.head.call_args
            params = call_args[1]["params"]
            assert "model_id" not in params


class TestParseMaxAge:
    """Tests for _parse_max_age helper."""

    @pytest.mark.parametrize(
        "header_value,expected",
        [
            ("max-age=1800", 1800),
            ("max-age=300, private", 300),
            ("public, max-age=60", 60),
            ("max-age=0", 0),
            ("max-age=3600, must-revalidate", 3600),
        ],
    )
    def test_parses_valid_max_age(self, header_value, expected):
        """Should extract max-age value from valid Cache-Control headers."""
        headers = httpx.Headers({"cache-control": header_value})
        assert _parse_max_age(headers) == expected

    @pytest.mark.parametrize(
        "header_value",
        [
            "no-cache",
            "private",
            "no-store",
            "max-age=abc",
        ],
    )
    def test_returns_none_for_missing_max_age(self, header_value):
        """Should return None when max-age is absent or malformed."""
        headers = httpx.Headers({"cache-control": header_value})
        assert _parse_max_age(headers) is None

    def test_returns_none_when_no_cache_control_header(self):
        """Should return None when Cache-Control header is absent."""
        headers = httpx.Headers()
        assert _parse_max_age(headers) is None


class TestCacheControlIntegration:
    """Tests for Cache-Control driven caching behavior."""

    @pytest.mark.asyncio
    async def test_uses_max_age_from_response(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
    ):
        """Should cache using max-age from Cache-Control header."""
        response = _make_response(200, "max-age=120, private")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=response)
            mock_client_class.return_value = mock_http_client

            await usage_quota_client.check_quota_available(usage_quota_context)

        cache_key = usage_quota_context.to_cache_key()
        cached_value = await usage_quota_client.cache.get(cache_key)
        assert cached_value is True

    @pytest.mark.asyncio
    async def test_falls_back_to_default_ttl_without_header(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
    ):
        """Should fall back to CACHE_TTL when Cache-Control header is absent."""
        response = _make_response(200)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=response)
            mock_client_class.return_value = mock_http_client

            await usage_quota_client.check_quota_available(usage_quota_context)

        cache_key = usage_quota_context.to_cache_key()
        cached_value = await usage_quota_client.cache.get(cache_key)
        assert cached_value is True

    @pytest.mark.asyncio
    async def test_caches_402_with_max_age(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
    ):
        """Should cache 402 responses using max-age from header."""
        response = _make_response(402, "max-age=300, private")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=response)
            mock_client_class.return_value = mock_http_client

            result = await usage_quota_client.check_quota_available(usage_quota_context)

        assert result is False
        cache_key = usage_quota_context.to_cache_key()
        cached_value = await usage_quota_client.cache.get(cache_key)
        assert cached_value is False

    @pytest.mark.asyncio
    async def test_cache_hit_skips_http_call(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
    ):
        """Should return cached value without making HTTP request on cache hit."""
        cache_key = usage_quota_context.to_cache_key()
        await usage_quota_client.cache.set(cache_key, True, ttl=600)

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock()
            mock_client_class.return_value = mock_http_client

            result = await usage_quota_client.check_quota_available(usage_quota_context)

        assert result is True
        mock_http_client.head.assert_not_called()

    @pytest.mark.asyncio
    async def test_cache_hit_returns_false_for_denied(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
    ):
        """Should return False from cache when quota was denied."""
        cache_key = usage_quota_context.to_cache_key()
        await usage_quota_client.cache.set(cache_key, False, ttl=600)

        result = await usage_quota_client.check_quota_available(usage_quota_context)

        assert result is False

    @pytest.mark.asyncio
    async def test_second_call_uses_cache(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
    ):
        """Should use cache on second call instead of making another HTTP request."""
        response = _make_response(200, "max-age=1800, private")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=response)
            mock_client_class.return_value = mock_http_client

            result1 = await usage_quota_client.check_quota_available(
                usage_quota_context
            )
            result2 = await usage_quota_client.check_quota_available(
                usage_quota_context
            )

        assert result1 is True
        assert result2 is True
        assert mock_http_client.head.call_count == 1


class TestJitter:
    """Tests for TTL jitter application."""

    @pytest.mark.asyncio
    async def test_jitter_varies_cache_ttl(
        self,
    ):
        """Should apply jitter so that not all entries expire at the same time."""
        ttls = [_apply_jitter(1000) for _ in range(50)]

        min_expected = 1000 * (1 - JITTER_FACTOR)
        max_expected = 1000 * (1 + JITTER_FACTOR)
        for ttl in ttls:
            assert min_expected <= ttl <= max_expected

        assert max(ttls) - min(ttls) > 10


class TestConcurrentRequests:
    """Tests for concurrent request coalescing."""

    @pytest.mark.asyncio
    async def test_concurrent_requests_coalesced_to_single_http_call(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
    ):
        """Should make only one HTTP request when multiple coroutines check the same key concurrently."""
        response = _make_response(200, "max-age=120, private")

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(return_value=response)
            mock_client_class.return_value = mock_http_client

            results = await asyncio.gather(
                usage_quota_client.check_quota_available(usage_quota_context),
                usage_quota_client.check_quota_available(usage_quota_context),
                usage_quota_client.check_quota_available(usage_quota_context),
            )

        assert results == [True, True, True]
        assert mock_http_client.head.call_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_requests_propagate_errors(
        self,
        usage_quota_client: UsageQuotaClient,
        usage_quota_context: UsageQuotaEventContext,
        mock_http_client: AsyncMock,
    ):
        """Should propagate errors to all waiting coroutines."""
        with patch("httpx.AsyncClient") as mock_client_class:
            mock_http_client.head = AsyncMock(
                side_effect=httpx.TimeoutException("timeout")
            )
            mock_client_class.return_value = mock_http_client

            results = await asyncio.gather(
                usage_quota_client.check_quota_available(usage_quota_context),
                usage_quota_client.check_quota_available(usage_quota_context),
                return_exceptions=True,
            )

        for result in results:
            assert isinstance(result, UsageQuotaTimeoutError)
