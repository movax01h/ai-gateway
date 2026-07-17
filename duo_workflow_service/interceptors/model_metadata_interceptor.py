import json

import grpc
import structlog

from ai_gateway.config import get_config
from ai_gateway.model_metadata import (
    ModelMetadataByTag,
    build_default_feature_setting_metadata,
    build_model_metadata_by_tag,
    create_model_metadata_by_tag,
)
from duo_workflow_service.interceptors.authentication_interceptor import (
    current_user as current_user_context_var,
)
from lib.context import (
    current_model_metadata_context,
    current_model_metadata_with_size_context,
)

log = structlog.stdlib.get_logger("model_metadata_interceptor")


class ModelMetadataInterceptor(grpc.aio.ServerInterceptor):
    """Interceptor that handles model metadata propagation."""

    X_GITLAB_AGENT_PLATFORM_MODEL_METADATA = "x-gitlab-agent-platform-model-metadata"

    async def intercept_service(self, continuation, handler_call_details):

        metadata = dict(handler_call_details.invocation_metadata)
        raw_metadata = metadata.get(self.X_GITLAB_AGENT_PLATFORM_MODEL_METADATA, "")
        try:
            data = json.loads(raw_metadata)

            config = get_config()
            model_keys = (
                config.model_keys.model_dump()
                if hasattr(config.model_keys, "model_dump")
                else dict(config.model_keys)
            )

            if (
                isinstance(data, dict)
                and data.get("provider") == "gitlab"
                and (data.get("identifier") or data.get("feature_setting"))
            ):
                default = build_default_feature_setting_metadata(
                    feature_setting=data.get("feature_setting"),
                    identifier=data.get("identifier") or None,
                    model_keys=model_keys,
                    fireworks_api_base_url=config.fireworks_api_base_url,
                    user=current_user_context_var.get(),
                )

                # Populate the tag -> model map from the feature setting's
                # models_for_tags config so component `model_tags` resolve for
                # gitlab-provider requests (the standard Duo Agent Platform path).
                # Without this, by_tag stays empty and every tag falls back to
                # the default model.
                by_tag = build_model_metadata_by_tag(
                    data.get("feature_setting"),
                    provider_keys=model_keys,
                    fireworks_api_base_url=config.fireworks_api_base_url,
                )
                model_metadata_by_tag = ModelMetadataByTag(
                    default=default, by_tag=by_tag
                )
            else:
                # `data` may be a "provider stickiness" replay of previously checkpointed
                # metadata — GitLab Rails echoes it back verbatim on workflow resume. That
                # checkpoint blob serializes the resolved model as flat `api_key`/`endpoint`
                # fields, not the `provider_keys`/`fireworks_api_base_url` shape this parser
                # reads, so a replayed Fireworks/Mistral model never carries a usable key or
                # endpoint on its own. Backfill both from server config whenever the request
                # doesn't supply its own, so stickiness replay still authenticates. Genuine
                # client-supplied values (e.g. self-hosted BYO key) always win.
                if isinstance(data, dict):
                    if not data.get("provider_keys"):
                        data["provider_keys"] = model_keys
                    if not data.get("fireworks_api_base_url"):
                        data["fireworks_api_base_url"] = config.fireworks_api_base_url

                model_metadata_by_tag = create_model_metadata_by_tag(data)

            model_metadata_by_tag.add_user(current_user_context_var.get())
            current_model_metadata_context.set(model_metadata_by_tag.default)
            current_model_metadata_with_size_context.set(model_metadata_by_tag)
        except ValueError as error:
            # Never log raw_metadata itself — it carries provider API keys.
            # header_present separates an absent header from a malformed one.
            log.warning(
                "Model metadata not applied",
                error=str(error),
                header_present=bool(raw_metadata),
            )

        return await continuation(handler_call_details)
