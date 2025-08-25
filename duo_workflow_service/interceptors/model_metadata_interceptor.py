import json

import grpc

from ai_gateway.model_metadata import (
    create_model_metadata,
    current_model_metadata_context,
)


class ModelMetadataInterceptor(grpc.aio.ServerInterceptor):
    """Interceptor that handles model metadata propagation."""

    X_GITLAB_AGENT_PLATFORM_MODEL_METADATA = "x-gitlab-agent-platform-model-metadata"

    async def intercept_service(
        self,
        continuation,
        handler_call_details,
    ):
        """Intercept incoming requests to inject feature flags context."""
        metadata = dict(handler_call_details.invocation_metadata)

        try:
            data = json.loads(
                metadata.get(self.X_GITLAB_AGENT_PLATFORM_MODEL_METADATA, "")
            )
            current_model_metadata_context.set(create_model_metadata(data))
        except json.JSONDecodeError:
            pass

        return await continuation(handler_call_details)
