import os

from dependency_injector import containers, providers

from lib.usage_quota.service import UsageQuotaService

__all__ = [
    "ContainerUsageQuota",
]


def _get_customersdot_url() -> str:
    """Get CustomersDot URL from environment variables.

    Priority:
    1. AIGW_MOCK_CRED_CD_URL (when AIGW_MOCK_USAGE_CREDITS=true) for testing
    2. DUO_WORKFLOW_AUTH__OIDC_CUSTOMER_PORTAL_URL (Duo Workflow Service config)
    3. AIGW_CUSTOMER_PORTAL_URL (AI Gateway config)
    4. Default: https://customers.gitlab.com
    """
    # pylint: disable=direct-environment-variable-reference
    if os.environ.get(
        "AIGW_MOCK_USAGE_CREDITS", ""
    ).lower() == "true" and os.environ.get("AIGW_MOCK_CRED_CD_URL"):
        return os.environ.get("AIGW_MOCK_CRED_CD_URL", "")
    return os.environ.get(
        "DUO_WORKFLOW_AUTH__OIDC_CUSTOMER_PORTAL_URL",
        os.environ.get("AIGW_CUSTOMER_PORTAL_URL", "https://customers.gitlab.com"),
    )
    # pylint: enable=direct-environment-variable-reference


def _get_customersdot_api_user() -> str | None:
    """Get CustomersDot API user from environment variables."""
    # pylint: disable=direct-environment-variable-reference
    return os.environ.get("CUSTOMER_PORTAL_USAGE_QUOTA_API_USER")
    # pylint: enable=direct-environment-variable-reference


def _get_customersdot_api_token() -> str | None:
    """Get CustomersDot API token from environment variables."""
    # pylint: disable=direct-environment-variable-reference
    return os.environ.get("CUSTOMER_PORTAL_USAGE_QUOTA_API_TOKEN")
    # pylint: enable=direct-environment-variable-reference


class ContainerUsageQuota(containers.DeclarativeContainer):
    """Dependency injection container for usage quota functionality."""

    service = providers.Singleton(
        UsageQuotaService,
        customersdot_url=providers.Callable(_get_customersdot_url),
        customersdot_api_user=providers.Callable(_get_customersdot_api_user),
        customersdot_api_token=providers.Callable(_get_customersdot_api_token),
    )
