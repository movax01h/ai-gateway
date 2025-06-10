import asyncio
import logging
from typing import Any, Callable, Dict, Optional, Union
from urllib.parse import urlencode

from contract import contract_pb2
from duo_workflow_service.executor.action import (
    _execute_action,
    _execute_action_and_get_action_response,
)
from duo_workflow_service.gitlab.http_client import GitlabHttpClient, GitLabHttpResponse

logger = logging.getLogger(__name__)


class ExecutorGitLabHttpClient(GitlabHttpClient):
    """GitLab HTTP client implementation that uses the executor service."""

    def __init__(self, outbox: asyncio.Queue, inbox: asyncio.Queue):
        self.outbox = outbox
        self.inbox = inbox

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
                {"outbox": self.outbox, "inbox": self.inbox},
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
            {"outbox": self.outbox, "inbox": self.inbox},
            contract_pb2.Action(
                runHTTPRequest=contract_pb2.RunHTTPRequest(
                    path=path, method=method, body=data
                )
            ),
        )

        return self._parse_response(
            response, parse_json=parse_json, object_hook=object_hook
        )
