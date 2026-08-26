import asyncio
import json
import logging
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, Optional, Union
from urllib.parse import urlencode

from langchain_core.tools import ToolException
from tenacity import (
    RetryCallState,
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)
from tenacity.wait import wait_base

from contract import contract_pb2
from duo_workflow_service.executor.action import (
    _execute_action_and_get_action_response,
)
from duo_workflow_service.executor.outbox import Outbox
from duo_workflow_service.gitlab.http_client import GitlabHttpClient, GitLabHttpResponse

logger = logging.getLogger(__name__)

# Retry budget: 4 attempts with 3s / 9s / 27s of backoff between them (39s total).
#
# The dominant transient failure in production is GitLab.com Postgres replica
# lag.  A Duo Agent Platform OAuth token is created on the primary and pinned
# there via `Gitlab::Database::LoadBalancing::Sticking`, whose Redis entry
# expires after 30 seconds.  Once it expires, token lookups fall back to a read
# replica; during a lag spike that replica does not have the token row yet and
# GitLab answers 401.  A 39s retry budget on top of the 30s sticking window
# covers replica lag up to ~69s, which is the bulk of the observed spikes.
_MAX_RETRY_ATTEMPTS = 4
_RETRY_WAIT_MULTIPLIER = 3
_RETRY_WAIT_EXP_BASE = 3
_RETRY_WAIT_MIN_SECONDS = 3
_RETRY_WAIT_MAX_SECONDS = 27

# HTTP status codes that are retried in addition to 5xx.
#
# 401: replica lag makes a valid, unexpired token temporarily invisible to
# GitLab; the same token succeeds again once the replica catches up.  A token
# that is genuinely expired or revoked keeps returning 401 and is surfaced to
# the caller once the retry budget is spent.
#
# 429: a rate limit, which is by definition "try again later".  GitLab's
# throttled responder tells us exactly how much later via `Retry-After`, so
# these waits are driven by the header rather than by the ladder above.
_RETRYABLE_STATUS_CODES = frozenset({401, 429})

# Network-error keywords that indicate a transient connectivity problem.
_NETWORK_ERROR_KEYWORDS = (
    "connection refused",
    "connection reset",
    "connection aborted",
    "broken pipe",
    "network unreachable",
    "name or service not known",
    "temporary failure in name resolution",
    "failed to establish",
    "remote end closed connection",
    "cannot connect to host",
)


class _RetryableStatusError(Exception):
    """Raised internally to trigger a retry when the executor returns a retryable HTTP status code.

    This exception is never propagated to callers.  :meth:`ExecutorGitLabHttpClient._call` catches it
    after all retry attempts are exhausted and converts it back to a :class:`GitLabHttpResponse`
    carrying the original status code; :meth:`ExecutorGitLabHttpClient.graphql` converts it to a
    descriptive :class:`Exception`.
    """

    def __init__(self, status_code: int, body: str, headers: Any):
        super().__init__(f"Retryable HTTP status: {status_code}")
        self.status_code = status_code
        self.body = body
        self.headers = headers


def _is_retryable_status(status_code: int) -> bool:
    return status_code >= 500 or status_code in _RETRYABLE_STATUS_CODES


class _RetryAfter:
    """The delay a response asked the client to wait for, per its RFC 7231 `Retry-After` header.

    Wraps one set of response headers and answers a single question -- "how long did the server ask us to wait?" --
    leaving what to do about that answer to :class:`_WaitRetryAfter`.
    """

    HEADER = "retry-after"

    def __init__(self, headers: Any) -> None:
        self._headers = headers

    def seconds(self) -> Optional[float]:
        """Return the requested delay in seconds, or None when the header is absent or unparsable."""
        value = self._header_value()

        return None if value is None else self._parse(value)

    def _header_value(self) -> Optional[str]:
        """Find the header, matching case-insensitively.

        Workhorse canonicalises the name to `Retry-After`, but other executors are free to use a different case.
        """
        try:
            items = self._headers.items()
        except AttributeError:
            return None

        for name, value in items:
            if name.lower() == self.HEADER:
                return value

        return None

    def _parse(self, value: str) -> Optional[float]:
        """Convert a header value to a delay in seconds.

        RFC 7231 permits either delta-seconds or an HTTP-date. GitLab sends delta-seconds, but the date form is handled
        too rather than being silently ignored.
        """
        candidate = (value or "").strip()
        if not candidate:
            return None

        for parse in (self._as_delta, self._as_http_date):
            seconds = parse(candidate)
            if seconds is not None:
                return seconds

        # An executor that joins repeated headers yields e.g. "42, 42"; read the
        # first value. Attempted last, because an HTTP-date contains a comma of
        # its own and splitting on it first would mangle the date into "Wed".
        head = candidate.split(",")[0].strip()

        return self._as_delta(head) if head != candidate else None

    def _as_delta(self, value: str) -> Optional[float]:
        """Parse the delta-seconds form, clamping a negative delay to "retry now"."""
        try:
            return max(0.0, float(int(value)))
        except ValueError:
            return None

    def _as_http_date(self, value: str) -> Optional[float]:
        """Parse the HTTP-date form into a delay measured from now, clamped at zero."""
        try:
            retry_at = parsedate_to_datetime(value)
        except (TypeError, ValueError):
            return None

        # An HTTP-date without a timezone is UTC by definition (RFC 7231).
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)

        return max(0.0, (retry_at - self._now()).total_seconds())

    def _now(self) -> datetime:
        return datetime.now(timezone.utc)


class _WaitRetryAfter(wait_base):
    """Wait for the duration the server asked for, falling back to exponential backoff.

    A `Retry-After` header is a direct instruction about when the resource becomes available again, so it beats any
    locally chosen interval. Anything else -- a timeout, a network error, or a status response without the header --
    falls through to `fallback`.

    The header is read at wait time rather than when the error is raised, so the HTTP-date form is measured against the
    moment we are about to sleep, and responses that never lead to a wait are never parsed.
    """

    # Upper bound on how long a `Retry-After` header can hold up a request.
    #
    # GitLab computes the value as `period - (now % period)` (see
    # `Gitlab::RackAttack::RequestThrottleData#retry_after`), so it can be as
    # large as the whole throttle period.  Waiting for the quota window to
    # actually reset makes the next attempt very likely to succeed -- one
    # well-timed wait beats three badly-timed ones -- but the cap stops a
    # rate-limited namespace from stalling a single tool call for minutes.
    MAX_SECONDS = 30

    def __init__(self, fallback: wait_base) -> None:
        self._fallback = fallback

    def __call__(self, retry_state: RetryCallState) -> float:
        outcome = retry_state.outcome
        error = outcome.exception() if outcome and outcome.failed else None

        requested = (
            _RetryAfter(error.headers).seconds()
            if isinstance(error, _RetryableStatusError)
            else None
        )

        if requested is None:
            return self._fallback(retry_state)

        return min(requested, self.MAX_SECONDS)


def _is_retryable_error(exc: BaseException) -> bool:
    """Return True if the exception represents a transient, retryable error.

    Retryable conditions:
    - Timeout errors (asyncio.TimeoutError or messages containing "timed out")
    - Network connectivity errors (connection refused, reset, etc.)
    - Retryable HTTP statuses: 5xx, and 401 caused by database replica lag
    """
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, _RetryableStatusError):
        return True
    if isinstance(exc, Exception):
        message = str(exc).lower()
        if "timed out" in message:
            return True
        if any(keyword in message for keyword in _NETWORK_ERROR_KEYWORDS):
            return True
    return False


_retry_on_transient_error = retry(
    reraise=True,
    stop=stop_after_attempt(_MAX_RETRY_ATTEMPTS),
    wait=_WaitRetryAfter(
        fallback=wait_exponential(
            multiplier=_RETRY_WAIT_MULTIPLIER,
            exp_base=_RETRY_WAIT_EXP_BASE,
            min=_RETRY_WAIT_MIN_SECONDS,
            max=_RETRY_WAIT_MAX_SECONDS,
        )
    ),
    retry=retry_if_exception(_is_retryable_error),
    before_sleep=before_sleep_log(logger, logging.WARNING),
)


class ExecutorGitLabHttpClient(GitlabHttpClient):
    """GitLab HTTP client implementation that uses the executor service."""

    def __init__(self, outbox: Outbox):
        self.outbox = outbox

    async def _call(
        self,
        path: str,
        method: str,
        parse_json: bool = True,
        data: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        object_hook: Union[Callable, None] = None,
    ) -> Any:
        if params:
            query_string = urlencode(params)
            path = f"{path}?{query_string}"

        @_retry_on_transient_error
        async def _call_with_retry() -> Any:
            action_response = await _execute_action_and_get_action_response(
                {"outbox": self.outbox},
                contract_pb2.Action(
                    runHTTPRequest=contract_pb2.RunHTTPRequest(
                        path=path, method=method, body=data
                    )
                ),
            )
            status_code = action_response.httpResponse.statusCode
            if _is_retryable_status(status_code):
                raise _RetryableStatusError(
                    status_code,
                    action_response.httpResponse.body,
                    action_response.httpResponse.headers,
                )
            body = self._parse_response(
                action_response.httpResponse.body,
                parse_json=parse_json,
                object_hook=object_hook,
            )
            return GitLabHttpResponse(
                status_code=status_code,
                body=body,
                headers=action_response.httpResponse.headers,
            )

        try:
            return await _call_with_retry()
        except _RetryableStatusError as exc:
            # All retry attempts exhausted.  Return the last response as a
            # regular response so callers can inspect the status code, matching
            # the pre-retry behaviour.
            body = self._parse_response(
                exc.body,
                parse_json=parse_json,
                object_hook=object_hook,
            )
            return GitLabHttpResponse(
                status_code=exc.status_code,
                body=body,
                headers=exc.headers,
            )

    async def graphql(
        self, query: str, variables: Optional[dict] = None, timeout: float = 10.0
    ) -> Any:
        payload = {
            "query": query,
            "variables": variables or {},
        }

        @_retry_on_transient_error
        async def _graphql_with_retry() -> str:
            try:
                action_response = await asyncio.wait_for(
                    _execute_action_and_get_action_response(
                        {"outbox": self.outbox},
                        contract_pb2.Action(
                            runHTTPRequest=contract_pb2.RunHTTPRequest(
                                path="/api/graphql",
                                method="POST",
                                body=json.dumps(payload),
                            )
                        ),
                    ),
                    timeout=timeout,
                )
            except asyncio.TimeoutError:
                raise Exception(f"GraphQL request timed out after {timeout} seconds")

            return self._graphql_response_body(action_response)

        try:
            response = await _graphql_with_retry()
        except _RetryableStatusError as exc:
            # All retry attempts exhausted.  GraphQL callers have no status code
            # to inspect, so surface the status in the message instead.
            raise Exception(
                f"GraphQL request failed with HTTP {exc.status_code}: {exc.body}"
            ) from exc

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON response from GraphQL: {response}")

        if "errors" in data:
            raise Exception(f"GraphQL errors: {data['errors']}")

        return data["data"]

    @staticmethod
    def _graphql_response_body(
        action_response: contract_pb2.ActionResponse,
    ) -> str:
        """Extract the GraphQL response body, retrying on retryable HTTP statuses.

        Unlike ``_execute_action``, which discards the status code and returns only the body, this
        inspects ``httpResponse.statusCode`` so that a 401 caused by database replica lag can be
        retried instead of being surfaced as an unrecoverable ``GraphQL errors: [{'message': 'Invalid
        token'}]``.

        Executors that answer ``runHTTPRequest`` with a ``plainTextResponse`` carry no status code, so
        their body is returned as-is.
        """
        response_type = action_response.WhichOneof("response_type")

        if response_type == "httpResponse":
            status_code = action_response.httpResponse.statusCode
            if _is_retryable_status(status_code):
                raise _RetryableStatusError(
                    status_code,
                    action_response.httpResponse.body,
                    action_response.httpResponse.headers,
                )
            return action_response.httpResponse.body

        if response_type == "plainTextResponse":
            return action_response.plainTextResponse.response

        raise ToolException("Executor doesn't return expected response fields")
