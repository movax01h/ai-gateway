import json

import grpc

from ai_gateway.model_metadata import (
    create_model_metadata,
    current_model_metadata_context,
)
from duo_workflow_service.interceptors.authentication_interceptor import (
    current_user as current_user_context_var,
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

            model_metadata = create_model_metadata(data)
            if model_metadata:
                model_metadata.add_user(current_user_context_var.get())
            current_model_metadata_context.set(model_metadata)

        except json.JSONDecodeError:
            pass

        return await continuation(handler_call_details)
