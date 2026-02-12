import asyncio
import os
from typing import Any, Callable
from urllib.parse import urljoin

import httpx
import structlog
from aiocache import Cache, cached
from aiocache.plugins import BasePlugin
from gitlab_cloud_connector import CloudConnectorUser

from ai_gateway.instrumentators.usage_quota import (
    USAGE_QUOTA_CUSTOMERSDOT_LATENCY_SECONDS,
    USAGE_QUOTA_CUSTOMERSDOT_REQUESTS_TOTAL,
)
from lib.billing_events.context import UsageQuotaEventContext
from lib.context import StarletteUser
from lib.internal_events import current_event_context
from lib.usage_quota.errors import (
    UsageQuotaConnectionError,
    UsageQuotaHTTPError,
    UsageQuotaTimeoutError,
)

log = structlog.stdlib.get_logger("usage_quota")

# pylint: disable=direct-environment-variable-reference
CACHE_TTL = (
    5 if os.environ.get("AIGW_MOCK_USAGE_CREDITS", "").lower() == "true" else 3600
)
# pylint: enable=direct-environment-variable-reference

MAX_KEEPALIVE_CONNECTIONS = 20  # Per client instance
MAX_CONNECTIONS = 100  # Total connection pool size
MAX_CACHE_SIZE = 10_000

SKIP_USAGE_CUTOFF_CLAIM = "skip_usage_cutoff"


def should_skip_usage_quota_for_user(user: CloudConnectorUser | StarletteUser | None):
    return (
        user
        and user.claims
        and user.claims.extra
        # Returns the value of SKIP_USAGE_CUTOFF_CLAIM or False if it's omitted
        and user.claims.extra.get(SKIP_USAGE_CUTOFF_CLAIM, False)
    )


def usage_quota_cache_key_builder(
    _fn: Callable[..., Any], *args, **_kwargs: dict[str, Any]
) -> str:
    context: UsageQuotaEventContext = args[1]
    key = context.to_cache_key()
    log.debug("Computed cache key", cache_key=key)
    return key


class LoggingCachePlugin(BasePlugin):
    async def pre_get(self, *args, **kwargs):
        pass

    async def post_get(self, _client, key: str, ret: bool | None = None, **_kwargs):
        message = (
            "Cache HIT - value retrieved from usage quota cache"
            if ret is not None
            else "Cache MISS - value not found in usage quota cache"
        )

        log.info(message, key=key, value=ret, cache_hit=ret is not None)


class UsageQuotaClient:
    """Client for checking usage quota availability via CustomersDot API.

    This client implements a fail-open strategy: if the CustomersDot API
    is unavailable or returns an error, quota checks will pass to avoid
    blocking legitimate users.

    Results are cached for 1 hour (configurable via AIGW_MOCK_USAGE_CREDITS).

    Args:
        customersdot_url: Base URL of the CustomersDot service
        customersdot_api_user: API username for calling the Customers Portal
        customersdot_api_token: API Token for calling the Customers Portal
        request_timeout: Maximum time to wait for API response (default: 1.0s)
    """

    def __init__(
        self,
        customersdot_url: str,
        customersdot_api_user: str | None,
        customersdot_api_token: str | None,
        request_timeout: float = 1.0,
    ) -> None:
        if not customersdot_url or not customersdot_url.strip():
            raise ValueError("customersdot_url cannot be empty")

        self.customersdot_url = customersdot_url
        self.customersdot_api_user = customersdot_api_user
        self.customersdot_api_token = customersdot_api_token
        self.enabled = (
            self.customersdot_api_token is not None
            and self.customersdot_api_user is not None
        )
        self.request_timeout = request_timeout
        self._http_client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create persistent HTTP client (lazy initialization).

        This method implements a coroutine-safe lazy initialization pattern using
        double-checked locking to ensure only one HTTP client is created even
        when called concurrently from multiple coroutines.

        The persistent client enables connection pooling and reuse, significantly
        reducing overhead compared to creating a new client per request.

        Returns:
            Singleton httpx.AsyncClient instance configured with timeout and connection limits.
        """
        if self._http_client is None:
            async with self._client_lock:
                self._http_client = httpx.AsyncClient(
                    timeout=httpx.Timeout(self.request_timeout),
                    limits=httpx.Limits(
                        max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
                        max_connections=MAX_CONNECTIONS,
                    ),
                )
                log.debug(
                    "Created persistent HTTP client",
                    client_id=id(self._http_client),
                    max_connections=MAX_CONNECTIONS,
                    max_keepalive_connections=MAX_KEEPALIVE_CONNECTIONS,
                    timeout=self.request_timeout,
                )
        return self._http_client

    @cached(
        ttl=CACHE_TTL,
        cache=Cache.MEMORY,
        noself=True,
        key_builder=usage_quota_cache_key_builder,
        plugins=[LoggingCachePlugin()],
    )
    async def check_quota_available(self, context: UsageQuotaEventContext) -> bool:
        """Check if the consumer has available usage quota.

        Makes a HEAD request to CustomersDot's /api/v1/consumers/resolve endpoint.
        Results are cached for 1 hour to reduce load on CustomersDot and to avoid affecting latency of AIGW.

        Args:
            context: Usage quota context containing consumer identification

        Returns:
            True if consumer has quota available (200 response)
            False if consumer quota exhausted (402 response)

        Raises:
            UsageQuotaTimeoutError: Request timed out (fail-open in middleware)
            UsageQuotaHTTPError: Unexpected HTTP error from CustomersDot (fail-open in middleware)
            UsageQuotaConnectionError: Connection to CustomersDot failed (fail-open in middleware)

        Note:
            Callers should implement fail-open error handling to avoid blocking
            legitimate users when CustomersDot is unavailable.
        """
        if not self.enabled:
            log.debug("Usage quota is disabled")
            return True

        realm = getattr(context, "realm", "unknown")
        params = context.model_dump(exclude_none=True, exclude_unset=True)
        # We always send feature_ai_catalog_item even when it's None
        # since None means we were not able to resolve the value when processing the legacy logic.
        params["feature_ai_catalog_item"] = context.feature_ai_catalog_item
        context_correlation_id = getattr(
            current_event_context.get(), "correlation_id", None
        )
        cache_key = context.to_cache_key()

        headers = {
            "X-Admin-Email": str(self.customersdot_api_user),
            "X-Admin-Token": str(self.customersdot_api_token),
        }

        try:
            client = await self._get_client()
            url = urljoin(self.customersdot_url, "api/v1/consumers/resolve")

            log.info(
                "Making usage quota request to CustomersDot",
                url=url,
                realm=realm,
                timeout=self.request_timeout,
                cache_key=cache_key,
                correlation_id=context_correlation_id,
            )

            with USAGE_QUOTA_CUSTOMERSDOT_LATENCY_SECONDS.labels(realm=realm).time():
                response = await client.head(url, params=params, headers=headers)

            # The Customers Portal responds with two HTTP status codes:
            # - Payment Required (402):
            #     returned when the customer does not have enough credits.
            # - Forbidden (403):
            #     returned when the entitlement check fails.
            # - OK (200):
            #     returned when the customer has sufficient credits and the entitlement check passes.
            # For all other HTTP status codes, we allow the request to proceed,
            # but we currently mark them as fail-open.

            status = response.status_code

            if status == httpx.codes.PAYMENT_REQUIRED:
                log.info(
                    "Usage quota denied",
                    status_code=status,
                    realm=realm,
                    correlation_id=context_correlation_id,
                )
                USAGE_QUOTA_CUSTOMERSDOT_REQUESTS_TOTAL.labels(
                    outcome="denied", status=str(status)
                ).inc()
                return False

            response.raise_for_status()

            USAGE_QUOTA_CUSTOMERSDOT_REQUESTS_TOTAL.labels(
                outcome="success", status=str(status)
            ).inc()
            log.info(
                "Usage quota check succeeded",
                status_code=status,
                realm=realm,
                correlation_id=context_correlation_id,
            )
            return True

        except httpx.TimeoutException as e:
            USAGE_QUOTA_CUSTOMERSDOT_REQUESTS_TOTAL.labels(
                outcome="timeout", status="timeout"
            ).inc()
            raise UsageQuotaTimeoutError(original_error=e) from e
        except httpx.HTTPStatusError as e:
            USAGE_QUOTA_CUSTOMERSDOT_REQUESTS_TOTAL.labels(
                outcome="http_error", status=str(e.response.status_code)
            ).inc()
            raise UsageQuotaHTTPError(
                status_code=e.response.status_code, original_error=e
            ) from e
        except httpx.RequestError as e:
            USAGE_QUOTA_CUSTOMERSDOT_REQUESTS_TOTAL.labels(
                outcome="unexpected", status="client_error"
            ).inc()
            raise UsageQuotaConnectionError(original_error=e) from e

    async def aclose(self):
        """Cleanup HTTP client resources."""
        if self._http_client is not None:
            log.debug("Closing HTTP client", client_id=id(self._http_client))
            await self._http_client.aclose()
            self._http_client = None
