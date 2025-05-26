from typing import Any, Callable, Dict, Optional, Union
from urllib.parse import urljoin

import structlog

from duo_workflow_service.gitlab.connection_pool import connection_pool
from duo_workflow_service.gitlab.http_client import GitlabHttpClient

log = structlog.stdlib.get_logger(__name__)


class DirectGitLabHttpClient(GitlabHttpClient):
    """GitLab HTTP client implementation that directly calls the GitLab API with connection pooling."""

    base_url: str
    gitlab_token: str

    def __init__(self, base_url: str, gitlab_token: str):
        self.base_url = base_url
        self.gitlab_token = gitlab_token

    async def _call(
        self,
        path: str,
        method: str,
        parse_json: bool = True,
        data: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        object_hook: Union[Callable, None] = None,
    ) -> Any:
        """Execute a request to the GitLab API.

        Args:
            path: The API endpoint path
            method: HTTP method (GET, POST, etc.)
            parse_json: Whether to parse the response as JSON
            data: Request body data
            params: Query parameters
            object_hook: Optional JSON decoder hook

        Returns:
            The API response, parsed as JSON if parse_json=True
        """

        url = urljoin(self.base_url, path)

        # Handle request arguments
        kwargs = {}
        if params:
            kwargs["params"] = params
        if data:
            # Pass data directly as a string parameter, not as a dict
            kwargs["data"] = data  # type: ignore

        headers = {
            "Authorization": f"Bearer {self.gitlab_token}",
            "Content-Type": "application/json",
        }

        # Get the session from the singleton connection pool
        session = connection_pool.session

        async with session.request(method, url, headers=headers, **kwargs) as response:  # type: ignore
            raw_response = await response.text()
            return self._parse_response(raw_response, parse_json, object_hook)
