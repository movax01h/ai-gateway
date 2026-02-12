# pylint: disable=direct-environment-variable-reference
"""Tests for UsageQuota DI container."""

import os
from unittest.mock import patch

from lib.usage_quota.container import (
    ContainerUsageQuota,
    _get_customersdot_api_token,
    _get_customersdot_api_user,
    _get_customersdot_url,
)


class TestCustomersDotURLResolution:
    """Tests for _get_customersdot_url function."""

    def test_returns_default_url(self):
        """Test that default URL is returned when no env vars set."""
        with patch.dict(os.environ, {}, clear=True):
            url = _get_customersdot_url()
            assert url == "https://customers.gitlab.com"

    def test_returns_aigw_customer_portal_url(self):
        """Test that AIGW_CUSTOMER_PORTAL_URL is used when set."""
        with patch.dict(
            os.environ,
            {"AIGW_CUSTOMER_PORTAL_URL": "https://customers.gitlab.local"},
            clear=True,
        ):
            url = _get_customersdot_url()
            assert url == "https://customers.gitlab.local"

    def test_returns_duo_workflow_url_over_aigw(self):
        """Test that DUO_WORKFLOW_AUTH__OIDC_CUSTOMER_PORTAL_URL takes precedence."""
        with patch.dict(
            os.environ,
            {
                "AIGW_CUSTOMER_PORTAL_URL": "https://customers.gitlab.local/aigw",
                "DUO_WORKFLOW_AUTH__OIDC_CUSTOMER_PORTAL_URL": "https://customers.gitlab.local/dws",
            },
            clear=True,
        ):
            url = _get_customersdot_url()
            assert url == "https://customers.gitlab.local/dws"

    def test_returns_mock_url_when_mock_enabled(self):
        """Test that AIGW_MOCK_CRED_CD_URL is used when AIGW_MOCK_USAGE_CREDITS=true."""
        with patch.dict(
            os.environ,
            {
                "AIGW_MOCK_USAGE_CREDITS": "true",
                "AIGW_MOCK_CRED_CD_URL": "https://mock.customers.gitlab.com",
                "AIGW_CUSTOMER_PORTAL_URL": "https://customers.gitlab.com",
            },
            clear=True,
        ):
            url = _get_customersdot_url()
            assert url == "https://mock.customers.gitlab.com"

    def test_ignores_mock_url_when_mock_disabled(self):
        """Test that AIGW_MOCK_CRED_CD_URL is ignored when AIGW_MOCK_USAGE_CREDITS=false."""
        with patch.dict(
            os.environ,
            {
                "AIGW_MOCK_USAGE_CREDITS": "false",
                "AIGW_MOCK_CRED_CD_URL": "https://mock.customers.gitlab.com",
                "AIGW_CUSTOMER_PORTAL_URL": "https://customers.gitlab.com",
            },
            clear=True,
        ):
            url = _get_customersdot_url()
            assert url == "https://customers.gitlab.com"


class TestCustomersDotAPIUser:
    """Tests for _get_customersdot_api_user function."""

    def test_returns_api_user_when_set(self):
        """Test that API user is returned when env var is set."""
        with patch.dict(
            os.environ, {"CUSTOMER_PORTAL_USAGE_QUOTA_API_USER": "test_user"}
        ):
            user = _get_customersdot_api_user()
            assert user == "test_user"

    def test_returns_none_when_not_set(self):
        """Test that None is returned when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            user = _get_customersdot_api_user()
            assert user is None


class TestCustomersDotAPIToken:
    """Tests for _get_customersdot_api_token function."""

    def test_returns_api_token_when_set(self):
        """Test that API token is returned when env var is set."""
        with patch.dict(
            os.environ, {"CUSTOMER_PORTAL_USAGE_QUOTA_API_TOKEN": "test_token"}
        ):
            token = _get_customersdot_api_token()
            assert token == "test_token"

    def test_returns_none_when_not_set(self):
        """Test that None is returned when env var is not set."""
        with patch.dict(os.environ, {}, clear=True):
            token = _get_customersdot_api_token()
            assert token is None


class TestContainerUsageQuota:
    """Tests for ContainerUsageQuota dependency injection container."""

    def test_provides_singleton_service(self):
        """Test that container provides a singleton service instance."""
        with patch.dict(
            os.environ,
            {
                "AIGW_CUSTOMER_PORTAL_URL": "https://customers.gitlab.local",
                "CUSTOMER_PORTAL_USAGE_QUOTA_API_USER": "test_user",
                "CUSTOMER_PORTAL_USAGE_QUOTA_API_TOKEN": "test_token",
            },
        ):
            container = ContainerUsageQuota()
            service1 = container.service()
            service2 = container.service()

            # Should return same instance (singleton)
            assert service1 is service2

    def test_service_configured_with_env_vars(self):
        """Test that service is configured with values from env vars."""
        with patch.dict(
            os.environ,
            {
                "AIGW_CUSTOMER_PORTAL_URL": "https://customers.test.local",
                "CUSTOMER_PORTAL_USAGE_QUOTA_API_USER": "test_user",
                "CUSTOMER_PORTAL_USAGE_QUOTA_API_TOKEN": "test_token",
                "AIGW_MOCK_USAGE_CREDITS": "false",
            },
            clear=True,
        ):
            container = ContainerUsageQuota()
            service = container.service()

            assert (
                service.usage_quota_client.customersdot_url
                == "https://customers.test.local"
            )
            assert service.usage_quota_client.customersdot_api_user == "test_user"
            assert service.usage_quota_client.customersdot_api_token == "test_token"
