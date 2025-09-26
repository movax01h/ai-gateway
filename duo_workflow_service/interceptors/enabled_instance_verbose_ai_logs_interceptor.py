import grpc

from lib.verbose_ai_logs import VERBOSE_AI_LOGS_HEADER, current_verbose_ai_logs_context


class EnabledInstanceVerboseAiLogsInterceptor(grpc.aio.ServerInterceptor):
    """Interceptor that handles verbose AI logs flag propagation."""

    def __init__(self):
        pass

    async def intercept_service(
        self,
        continuation,
        handler_call_details,
    ):
        """Intercept incoming requests to inject verbose AI logs context."""
        metadata = dict(handler_call_details.invocation_metadata)

        is_enabled = metadata.get(VERBOSE_AI_LOGS_HEADER) == "true"

        current_verbose_ai_logs_context.set(is_enabled)

        return await continuation(handler_call_details)
