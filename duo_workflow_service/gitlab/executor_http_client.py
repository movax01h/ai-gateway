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


def _is_timeout_error(exc: BaseException) -> bool:
    """Return True if the exception represents a retryable timeout."""
    if isinstance(exc, asyncio.TimeoutError):
        return True
    if isinstance(exc, Exception) and "timed out" in str(exc).lower():
        return True
    return False


_retry_on_timeout = retry(
    reraise=True,
    stop=stop_after_attempt(_MAX_RETRY_ATTEMPTS),
    wait=wait_exponential(
        multiplier=1, min=_RETRY_WAIT_MIN_SECONDS, max=_RETRY_WAIT_MAX_SECONDS
    ),
    retry=retry_if_exception(_is_timeout_error),
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

        @_retry_on_timeout
        async def _call_with_retry() -> Any:
            action_response = await _execute_action_and_get_action_response(
                {"outbox": self.outbox},
                contract_pb2.Action(
                    runHTTPRequest=contract_pb2.RunHTTPRequest(
                        path=path, method=method, body=data
                    )
                ),
            )
            body = self._parse_response(
                action_response.httpResponse.body,
                parse_json=parse_json,
                object_hook=object_hook,
            )
            return GitLabHttpResponse(
                status_code=action_response.httpResponse.statusCode,
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

        @_retry_on_timeout
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
