import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional, Union
from urllib.parse import urlencode

from langchain_core.tools import ToolException
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

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

# HTTP status codes that are retried in addition to 5xx.  401 is included
# because replica lag makes a valid, unexpired token temporarily invisible to
# GitLab; the same token succeeds again once the replica catches up.  A token
# that is genuinely expired or revoked keeps returning 401 and is surfaced to
# the caller once the retry budget is spent.
_RETRYABLE_STATUS_CODES = frozenset({401})

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
    wait=wait_exponential(
        multiplier=_RETRY_WAIT_MULTIPLIER,
        exp_base=_RETRY_WAIT_EXP_BASE,
        min=_RETRY_WAIT_MIN_SECONDS,
        max=_RETRY_WAIT_MAX_SECONDS,
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
