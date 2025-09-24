from typing import Union

from starlette.requests import HTTPConnection, Request
from starlette_context.plugins import Plugin

from lib.verbose_ai_logs import current_verbose_ai_logs_context

HEADER_KEY = "x-gitlab-enabled-instance-verbose-ai-logs"


class EnabledInstanceVerboseAiLogsHeaderPlugin(Plugin):
    key = "enabled-instance-verbose-ai-logs"

    async def process_request(self, request: Union[Request, HTTPConnection]) -> bool:
        """Extract the header value and sets in both current_verbose_ai_logs_context and the starlette context.

        Args:
            request: The incoming HTTP request

        Returns:
            The value of the header as a boolean
        """
        is_enabled = request.headers.get(HEADER_KEY) == "true"
        # sets the value in the shared context too, so that it can be reused by
        # Duo workflow service.
        current_verbose_ai_logs_context.set(is_enabled)
        return is_enabled
