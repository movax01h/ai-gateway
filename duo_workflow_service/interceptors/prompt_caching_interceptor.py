import grpc

from lib.prompts.caching import (
    X_GITLAB_MODEL_PROMPT_CACHE_ENABLED,
    set_prompt_caching_enabled_to_current_request,
)


class PromptCachingInterceptor(grpc.aio.ServerInterceptor):
    """Interceptor that handles language server version propagation."""

    async def intercept_service(
        self,
        continuation,
        handler_call_details,
    ):
        """Intercept incoming requests to track language server version."""
        metadata = dict(handler_call_details.invocation_metadata)

        set_prompt_caching_enabled_to_current_request(
            metadata.get(X_GITLAB_MODEL_PROMPT_CACHE_ENABLED.lower(), None)
        )

        return await continuation(handler_call_details)
