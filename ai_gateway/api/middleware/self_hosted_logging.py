from typing import Union, override

from starlette.requests import HTTPConnection, Request
from starlette_context.plugins import Plugin

from lib.verbose_ai_logs import VERBOSE_AI_LOGS_HEADER, current_verbose_ai_logs_context


class EnabledInstanceVerboseAiLogsHeaderPlugin(Plugin):
    key = "enabled-instance-verbose-ai-logs"

    @override
    async def process_request(self, request: Union[Request, HTTPConnection]) -> bool:
        """Extract the header value and sets in both current_verbose_ai_logs_context and the starlette context.

        Args:
            request: The incoming HTTP request

        Returns:
            The value of the header as a boolean
        """
        is_enabled = request.headers.get(VERBOSE_AI_LOGS_HEADER) == "true"
        # sets the value in the shared context too, so that it can be reused by
        # Duo workflow service.
        current_verbose_ai_logs_context.set(is_enabled)
        return is_enabled
