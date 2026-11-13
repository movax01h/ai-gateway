import asyncio
import os
import random
import re
from typing import NamedTuple
from urllib.parse import urljoin

import httpx
import structlog
from aiocache import Cache
from aiocache.plugins import BasePlugin
from gitlab_cloud_connector import CloudConnectorUser
from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_random,
)

from ai_gateway.instrumentators.usage_quota import (
    USAGE_QUOTA_CUSTOMERSDOT_LATENCY_SECONDS,
    USAGE_QUOTA_CUSTOMERSDOT_REQUESTS_TOTAL,
)
from lib.billing_events.context import UsageQuotaEventContext
from lib.context import StarletteUser
from lib.feature_flags.context import FeatureFlag, is_feature_enabled
from lib.internal_events import current_event_context
from lib.usage_quota.circuit_breaker import CircuitBreaker
from lib.usage_quota.errors import (
    UsageQuotaCircuitOpenError,
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
JITTER_FACTOR = 0.1  # ±10% jitter on cache TTLs to stagger expiry across instances

# Retry transient CustomersDot failures once. The check is on the request path of
# latency-sensitive features such as code completions, so the worst case is kept to two
# request timeouts plus a short backoff.
_MAX_ATTEMPTS = 2
_RETRY_WAIT_MIN_SECONDS = 0.1
_RETRY_WAIT_MAX_SECONDS = 0.2
# Only server errors that are usually temporary. Other 5xx codes, such as 501, would fail
# the same way again.
_TRANSIENT_STATUS_CODES = frozenset({500, 502, 503, 504})

# Without a breaker, the retry would double the requests to CustomersDot and the wait for
# callers during an outage. After this many consecutive transient failures, checks that
# miss the cache fail fast. The short recovery time keeps a brief outage from turning into
# a much longer one.
_CIRCUIT_BREAKER_FAILURE_THRESHOLD = 10
_CIRCUIT_BREAKER_RECOVERY_SECONDS = 5.0

SKIP_USAGE_CUTOFF_CLAIM = "skip_usage_cutoff"

_CACHE_CONTROL_MAX_AGE_RE = re.compile(r"max-age=(\d+)")


class FetchedQuota(NamedTuple):
    allowed: bool
    ttl: int


class LoggingCachePlugin(BasePlugin):
    async def pre_get(self, *args, **kwargs):
        pass

    async def post_get(self, _client, key: str, ret=None, **_kwargs):
        message = (
            "Cache HIT - value retrieved from usage quota cache"
            if ret is not None
            else "Cache MISS - value not found in usage quota cache"
        )

        log.info(message, key=key, value=ret, cache_hit=ret is not None)


def should_skip_usage_quota_for_user(
    user: CloudConnectorUser | StarletteUser | None,
) -> bool:
    """Return whether the user skips the usage quota.

    The signed `skip_usage_cutoff` claim turns the skip on. The `internal_ai_usage`
    feature flag can only turn it off, so the unsigned flag header never grants a skip.
    """
    has_skip_claim = (
        user
        and user.claims
        and user.claims.extra
        # Returns the value of SKIP_USAGE_CUTOFF_CLAIM or False if it's omitted
        and user.claims.extra.get(SKIP_USAGE_CUTOFF_CLAIM, False)
    )
    internal_ai_usage = is_feature_enabled(FeatureFlag.INTERNAL_AI_USAGE)
    skip_usage_quota = bool(has_skip_claim) and not internal_ai_usage
    log.info(
        "Usage quota skip check",
        skip_usage_quota=skip_usage_quota,
        internal_ai_usage=internal_ai_usage,
        global_user_id=(getattr(user, "global_user_id", None) if user else None),
    )
    return skip_usage_quota


def _parse_max_age(headers: httpx.Headers) -> int | None:
    cache_control = headers.get("cache-control")
    if not cache_control:
        return None
    match = _CACHE_CONTROL_MAX_AGE_RE.search(cache_control)
    if not match:
        return None
    return int(match.group(1))


def _apply_jitter(ttl: int) -> float:
    # Apply ±10% random jitter to prevent thundering herd when multiple
    # AI Gateway instances expire the same cache key simultaneously.
    delta = ttl * JITTER_FACTOR
    return ttl + random.uniform(-delta, delta)


def _is_transient_error(exc: BaseException) -> bool:
    """Return whether a failed quota request is worth retrying.

    Timeouts, connection failures and 500, 502, 503 or 504 responses are transient. Any other HTTP status is an answer
    from CustomersDot (for example 403 for missing entitlements), so retrying it would only repeat the answer.
    """
    if isinstance(exc, (UsageQuotaTimeoutError, UsageQuotaConnectionError)):
        return True
    return (
        isinstance(exc, UsageQuotaHTTPError)
        and exc.status_code in _TRANSIENT_STATUS_CODES
    )


def _log_retry(retry_state: RetryCallState) -> None:
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    log.warning(
        "Retrying usage quota request to CustomersDot",
        attempt=retry_state.attempt_number,
        error_type=type(exc).__name__,
        correlation_id=getattr(current_event_context.get(), "correlation_id", None),
    )


class UsageQuotaClient:
    """Client for checking usage quota availability via CustomersDot API.

    This client raises typed exceptions when the CustomersDot API is unavailable
    or returns an unexpected error, allowing callers to apply the quota failure
    policy consistently.

    Results are cached in-memory using Cache-Control max-age from CustomersDot
    responses, with a fallback TTL when the header is absent.

    Timeouts, connection failures and 500, 502, 503 or 504 responses are retried
    once before the error is raised. A circuit breaker stops requests to CustomersDot after
    repeated transient failures, so checks that miss the cache fail fast instead
    of each waiting for the timeouts. Cached results are served either way.

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
        self.enabled = bool(
            self.customersdot_api_user and self.customersdot_api_user.strip()
        ) and bool(self.customersdot_api_token and self.customersdot_api_token.strip())
        self.request_timeout = request_timeout
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=_CIRCUIT_BREAKER_FAILURE_THRESHOLD,
            recovery_timeout=_CIRCUIT_BREAKER_RECOVERY_SECONDS,
        )
        self._http_client: httpx.AsyncClient | None = None
        self._client_lock = asyncio.Lock()
        self.cache = Cache(Cache.MEMORY, plugins=[LoggingCachePlugin()])
        self._inflight: dict[str, asyncio.Task[bool]] = {}

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
                if self._http_client is None:
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

    async def check_quota_available(self, context: UsageQuotaEventContext) -> bool:
        """Check if the consumer has available usage quota.

        Makes a HEAD request to CustomersDot's /api/v1/consumers/resolve endpoint.
        Results are cached using the Cache-Control max-age header from the response,
        falling back to CACHE_TTL when the header is absent.

        Args:
            context: Usage quota context containing consumer identification

        Returns:
            True if consumer has quota available (200 response)
            False if consumer quota exhausted (402 response)

        Raises:
            UsageQuotaTimeoutError: Request timed out
            UsageQuotaHTTPError: Unexpected HTTP error from CustomersDot
            UsageQuotaConnectionError: Connection to CustomersDot failed
            UsageQuotaCircuitOpenError: Circuit breaker is open, no request was made
        """
        if not self.enabled:
            log.debug("Usage quota is disabled")
            return True

        cache_key = context.to_cache_key()
        cached_value = await self.cache.get(cache_key)
        if cached_value is not None:
            return cached_value

        # Coalesce concurrent requests for the same cache key: only the first
        # coroutine makes the HTTP call; others await the same future.
        if cache_key in self._inflight:
            return await asyncio.shield(self._inflight[cache_key])

        async def _fetch_and_cache() -> bool:
            allowed, ttl = await self._fetch_quota(context, cache_key)
            jittered_ttl = _apply_jitter(ttl)
            await self.cache.set(cache_key, allowed, ttl=jittered_ttl)
            return allowed

        task = asyncio.get_running_loop().create_task(_fetch_and_cache())
        self._inflight[cache_key] = task

        try:
            return await asyncio.shield(task)
        finally:
            self._inflight.pop(cache_key, None)

    async def _fetch_quota(
        self, context: UsageQuotaEventContext, cache_key: str
    ) -> FetchedQuota:
        """Fetch the quota decision through the circuit breaker.

        Runs inside the coalesced task, so concurrent checks for the same key share one fetch, retry included. A check
        counts as one breaker outcome, however many attempts it took.
        """
        token = self.circuit_breaker.try_acquire()
        if token is None:
            # The service logs the denial, so only count it here.
            USAGE_QUOTA_CUSTOMERSDOT_REQUESTS_TOTAL.labels(
                outcome="circuit_open", status="circuit_open"
            ).inc()
            raise UsageQuotaCircuitOpenError()

        try:
            result = await self._request_quota(context, cache_key)
        except (
            UsageQuotaTimeoutError,
            UsageQuotaConnectionError,
            UsageQuotaHTTPError,
        ) as e:
            if _is_transient_error(e):
                self.circuit_breaker.record_failure(token)
            else:
                # Any other HTTP status is an answer, so CustomersDot is reachable.
                self.circuit_breaker.record_reachable(token)
            raise
        except BaseException:
            # No outcome about CustomersDot's health, for example on cancellation. Free a
            # half-open probe so the breaker can't get stuck.
            self.circuit_breaker.release(token)
            raise

        self.circuit_breaker.record_reachable(token)
        return result

    @retry(
        reraise=True,
        stop=stop_after_attempt(_MAX_ATTEMPTS),
        wait=wait_random(min=_RETRY_WAIT_MIN_SECONDS, max=_RETRY_WAIT_MAX_SECONDS),
        retry=retry_if_exception(_is_transient_error),
        before_sleep=_log_retry,
    )
    async def _request_quota(
        self, context: UsageQuotaEventContext, cache_key: str
    ) -> FetchedQuota:
        realm = getattr(context, "realm", "unknown")
        params = context.model_dump(exclude_none=True, exclude_unset=True)
        # We always send feature_ai_catalog_item even when it's None
        # since None means we were not able to resolve the value when processing the legacy logic.
        params["feature_ai_catalog_item"] = context.feature_ai_catalog_item
        context_correlation_id = getattr(
            current_event_context.get(), "correlation_id", None
        )

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
            # All other HTTP status codes raise UsageQuotaHTTPError for callers
            # to apply the quota failure policy.

            status = response.status_code
            max_age = _parse_max_age(response.headers)
            ttl = max_age if max_age is not None else CACHE_TTL

            log.info(
                "Parsed Cache-Control header",
                max_age=max_age,
                effective_ttl=ttl,
                status_code=status,
            )

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
                return FetchedQuota(False, ttl)

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
            return FetchedQuota(True, ttl)

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
