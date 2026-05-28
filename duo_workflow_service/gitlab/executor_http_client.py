import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional, Union
from urllib.parse import urlencode

from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from contract import contract_pb2
from duo_workflow_service.executor.action import (
    _execute_action,
    _execute_action_and_get_action_response,
)
from duo_workflow_service.executor.outbox import Outbox
from duo_workflow_service.gitlab.http_client import GitlabHttpClient, GitLabHttpResponse

logger = logging.getLogger(__name__)

_MAX_RETRY_ATTEMPTS = 3
_RETRY_WAIT_MIN_SECONDS = 1
_RETRY_WAIT_MAX_SECONDS = 10

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


class ServerErrorResponse(Exception):
    """Raised internally when the executor returns a 5xx HTTP status code."""

    def __init__(self, status_code: int):
        super().__init__(f"Server error: HTTP {status_code}")
        self.status_code = status_code


def _is_retryable_error(exc: BaseException) -> bool:
    """Return True if the exception represents a transient, retryable error.

    Retryable conditions:
    - Timeout errors (asyncio.TimeoutError or messages containing "timed out")
    - Network connectivity errors (connection refused, reset, etc.)
    - Server-side 5xx responses (ServerErrorResponse)
    """
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, ServerErrorResponse):
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
        multiplier=1, min=_RETRY_WAIT_MIN_SECONDS, max=_RETRY_WAIT_MAX_SECONDS
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
            if status_code >= 500:
                raise ServerErrorResponse(status_code)
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

        return await _call_with_retry()

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
                return await asyncio.wait_for(
                    _execute_action(
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

        response = await _graphql_with_retry()

        try:
            data = json.loads(response)
        except json.JSONDecodeError:
            raise Exception(f"Invalid JSON response from GraphQL: {response}")

        if "errors" in data:
            raise Exception(f"GraphQL errors: {data['errors']}")

        return data["data"]
