import asyncio
import json
import logging
from typing import Any, Callable, Dict, Optional, Union
from urllib.parse import urlencode

from contract import contract_pb2
from duo_workflow_service.executor.action import (
    _execute_action,
    _execute_action_and_get_action_response,
)
from duo_workflow_service.executor.outbox import Outbox
from duo_workflow_service.gitlab.http_client import GitlabHttpClient, GitLabHttpResponse

logger = logging.getLogger(__name__)


class ExecutorGitLabHttpClient(GitlabHttpClient):
    """GitLab HTTP client implementation that uses the executor service."""

    MAX_RETRIES = 3
    RETRY_DELAY_SECONDS = 2

    def __init__(self, outbox: Outbox):
        self.outbox = outbox

    async def _call(
        self,
        path: str,
        method: str,
        parse_json: bool = True,
        use_http_response: bool = False,
        data: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        object_hook: Union[Callable, None] = None,
    ) -> Any:
        if params:
            query_string = urlencode(params)
            path = f"{path}?{query_string}"

        if use_http_response:
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

        # The following code will be removed once all tools use the new http response
        response = await _execute_action(
            {"outbox": self.outbox},
            contract_pb2.Action(
                runHTTPRequest=contract_pb2.RunHTTPRequest(
                    path=path, method=method, body=data
                )
            ),
        )
        return self._parse_response(
            response, parse_json=parse_json, object_hook=object_hook
        )

    async def graphql(
        self, query: str, variables: Optional[dict] = None, timeout: float = 10.0
    ) -> Any:
        """Execute a GraphQL query with retry logic.

        Args:
            query: The GraphQL query string
            variables: Optional dictionary of variables for the query
            timeout: Timeout in seconds for each request attempt

        Returns:
            The data part of the GraphQL response

        Raises:
            Exception: If all retry attempts fail or timeout
        """
        payload = {
            "query": query,
            "variables": variables or {},
        }

        for attempt in range(self.MAX_RETRIES):
            try:
                response = await asyncio.wait_for(
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

                try:
                    data = json.loads(response)
                except json.JSONDecodeError:
                    raise Exception(f"Invalid JSON response from GraphQL: {response}")

                if "errors" in data:
                    raise Exception(f"GraphQL errors: {data['errors']}")

                return data["data"]

            except Exception as ex:
                logger.warning(
                    f"GraphQL request attempt {attempt + 1}/{self.MAX_RETRIES} failed: {ex}"
                )

                if attempt < self.MAX_RETRIES - 1:
                    await asyncio.sleep(self.RETRY_DELAY_SECONDS)
                else:
                    raise
