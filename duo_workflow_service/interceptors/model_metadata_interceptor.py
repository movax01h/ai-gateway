import json

import grpc

from ai_gateway.config import get_config
from ai_gateway.model_metadata import (
    ModelMetadataBySize,
    build_default_feature_setting_metadata,
    create_model_metadata_by_size,
)
from duo_workflow_service.interceptors.authentication_interceptor import (
    current_user as current_user_context_var,
)
from lib.context import (
    current_model_metadata_context,
    current_model_metadata_with_size_context,
)


class ModelMetadataInterceptor(grpc.aio.ServerInterceptor):
    """Interceptor that handles model metadata propagation."""

    X_GITLAB_AGENT_PLATFORM_MODEL_METADATA = "x-gitlab-agent-platform-model-metadata"

    async def intercept_service(self, continuation, handler_call_details):

        metadata = dict(handler_call_details.invocation_metadata)
        try:
            data = json.loads(
                metadata.get(self.X_GITLAB_AGENT_PLATFORM_MODEL_METADATA, "")
            )

            if (
                isinstance(data, dict)
                and data.get("provider") == "gitlab"
                and (data.get("identifier") or data.get("feature_setting"))
            ):
                config = get_config()
                model_keys = (
                    config.model_keys.model_dump()
                    if hasattr(config.model_keys, "model_dump")
                    else dict(config.model_keys)
                )

                default = build_default_feature_setting_metadata(
                    feature_setting=data.get("feature_setting"),
                    identifier=data.get("identifier") or None,
                    model_keys=model_keys,
                    fireworks_api_base_url=config.fireworks_api_base_url,
                    user=current_user_context_var.get(),
                )

                model_metadata_by_size = ModelMetadataBySize(default=default)
            else:
                model_metadata_by_size = create_model_metadata_by_size(data)

            model_metadata_by_size.add_user(current_user_context_var.get())
            current_model_metadata_context.set(model_metadata_by_size.default)
            current_model_metadata_with_size_context.set(model_metadata_by_size)
        except ValueError:
            pass

        return await continuation(handler_call_details)
