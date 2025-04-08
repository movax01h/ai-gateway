from typing import Union

from starlette.requests import HTTPConnection, Request
from starlette_context import context
from starlette_context.plugins import Plugin

HEADER_KEY = "x-gitlab-enabled-instance-verbose-ai-logs"


class EnabledInstanceVerboseAiLogsHeaderPlugin(Plugin):
    key = "enabled-instance-verbose-ai-logs"

    async def process_request(self, request: Union[Request, HTTPConnection]) -> bool:
        """
        Extract the 'my-header' value from the request headers.

        Args:
            request: The incoming HTTP request

        Returns:
            The value of 'my-header' if present, None otherwise
        """
        return request.headers.get(HEADER_KEY) == "true"


def enabled_instance_verbose_ai_logs():
    return context["enabled-instance-verbose-ai-logs"]
