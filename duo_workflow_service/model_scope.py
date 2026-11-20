"""Keep the models of a flat-rate flow inside the allowlist of the flow's feature.

A flow in ``BILL_ONCE_PER_WORKFLOW_FEATURES`` is billed once per workflow, whatever models it runs. The client chooses
those models through the ``x-gitlab-agent-platform-model-metadata`` header, so the header alone could run any model at
the flat rate. This module takes the feature from the name the workflow is billed as, not from the client, and keeps
every model of the request inside that feature's allowlist.

For every other flow, the allowlist still uses the client's feature. ``log_flow_model_feature`` logs that feature next to
the flow and the features of the flow, with what a check against the flow's features would do, so the move to the
flow's own features can be measured before it changes anything.
"""

from typing import Optional

import structlog
from gitlab_cloud_connector import CloudConnectorUser

from ai_gateway.config import get_config
from ai_gateway.feature_allowlist import (
    ClientModelRequest,
    dws_client_model_request_context,
)
from ai_gateway.model_metadata import (
    AmazonQModelMetadata,
    ModelMetadataByTag,
    TypeModelMetadata,
    build_default_feature_setting_metadata,
    build_model_metadata_by_tag,
)
from ai_gateway.model_selection import ModelSelectionConfig
from lib.billing_events import BILL_ONCE_PER_WORKFLOW_FEATURES
from lib.context.model import (
    current_model_metadata_context,
    current_model_metadata_with_size_context,
)

__all__ = [
    "CLIENT_FLOW_FEATURES",
    "FLAT_RATE_FLOW_FEATURE_SETTINGS",
    "flow_features",
    "log_flow_model_feature",
    "scope_flat_rate_flow_models",
]

log = structlog.stdlib.get_logger("model_scope")

# The feature setting of each flow in BILL_ONCE_PER_WORKFLOW_FEATURES. It is the `ai_feature` of the foundational flow
# definition in GitLab Rails (`ee/app/models/ai/catalog/foundational_flow/<flow>/definition.rb`).
FLAT_RATE_FLOW_FEATURE_SETTINGS: dict[str, str] = {
    "business_context_security_guidelines": "vulnerability_context_flow",
    "code_review": "review_merge_request_dap",
    "resolve_sast_vulnerability": "sast_vulnerability_resolution",
    "sast_fp_detection": "sast_vulnerability_fp_detection",
    "secrets_fp_detection": "secret_vulnerability_fp_detection",
}

# The features of a flow that the client sends as a config: an inline `flowConfig` or a fetched AI Catalog flow. The
# client chooses the name of such a flow, so the name does not select an entry of the `flows` map. GitLab Rails sends
# `duo_agent_platform` for these flows, and `duo_agent_platform_agentic_chat` when chat runs them.
CLIENT_FLOW_FEATURES: tuple[str, ...] = (
    "duo_agent_platform",
    "duo_agent_platform_agentic_chat",
)


def flow_features(
    feature_qualified_name: Optional[str], *, client_flow: bool
) -> tuple[str, ...]:
    """Return the features that a flow runs as, main feature first.

    Args:
        feature_qualified_name: The name the workflow is billed as, such as ``code_review/v1``.
        client_flow: Whether the client sent the flow as a config, inline or as a fetched AI Catalog flow.

    Returns:
        The entry of the flow's name, without its version, in the ``flows`` map of ``unit_primitives.yml``. A flow that
        the client sent gets ``CLIENT_FLOW_FEATURES`` instead, except a flow with a flat-rate name. Empty for a flow
        without an entry.
    """
    flow = (feature_qualified_name or "").split("/", 1)[0]
    # A flat-rate flow is billed by its name, so its models follow the name too, also for a flow that the client sent.
    # scope_flat_rate_flow_models does the same.
    if client_flow and flow not in BILL_ONCE_PER_WORKFLOW_FEATURES:
        return CLIENT_FLOW_FEATURES

    return ModelSelectionConfig.instance().get_flow_features().get(flow, ())


def _flow_check(
    request: ClientModelRequest,
    served: Optional[TypeModelMetadata],
    features: tuple[str, ...],
) -> str:
    """Return what a check of the request's default model against the flow's features would do. It changes nothing.

    The tag models are not checked: they come from the config of a feature, not from the client. The values are:

    - ``no_entry``: the flow has no features to check against.
    - ``unchanged``: the flow lists the client's feature, or the request is Amazon Q, which carries its own role.
    - ``feature_default``: the request names no model, or the interceptor kept none. The main feature's default runs.
    - ``allowed``: the main feature of the flow allows the model that passed the interceptor.
    - ``replaced``: the main feature does not allow it. ``fallback`` replaces it, and ``enforce`` refuses the request.
    """
    if not features:
        return "no_entry"
    if request.feature_setting in features or isinstance(served, AmazonQModelMetadata):
        return "unchanged"
    if not request.pinned or served is None:
        return "feature_default"

    main_feature = features[0]
    allowed = ModelSelectionConfig.instance().allowed_models_for_feature(main_feature)
    return (
        "allowed" if served.llm_definition.gitlab_identifier in allowed else "replaced"
    )


def scope_flat_rate_flow_models(
    feature_qualified_name: Optional[str], user: CloudConnectorUser
) -> None:
    """Rebuild the request's models of a flat-rate flow from the allowlist of the flow's feature.

    The client's default model stays when the feature allows it. Otherwise the feature's default model is served. The
    tag models always come from the flow's feature. Every model is rebuilt from its definition, so no other client field,
    such as a raw ``identifier`` or the provider string, changes the model that runs.

    Nothing changes for a flow that is not billed at a flat rate, for a request without model metadata or with Amazon Q
    metadata, and on a gateway that serves custom models.

    Args:
        feature_qualified_name: The name the workflow is billed as, such as ``code_review/v1``.
        user: The user the workflow runs for.
    """
    flow = (feature_qualified_name or "").split("/", 1)[0]
    if flow not in BILL_ONCE_PER_WORKFLOW_FEATURES:
        return

    config = get_config()
    requested = current_model_metadata_context.get()
    if (
        requested is None
        or isinstance(requested, AmazonQModelMetadata)
        or config.custom_models.enabled
    ):
        return

    feature_setting = FLAT_RATE_FLOW_FEATURE_SETTINGS[flow]
    requested_model = requested.llm_definition.gitlab_identifier
    allowed = (
        requested_model
        in ModelSelectionConfig.instance().allowed_models_for_feature(feature_setting)
    )
    provider_keys = config.model_keys.model_dump()
    # The interceptor's helper: the client's model when the feature allows it, otherwise the feature default.
    default = build_default_feature_setting_metadata(
        feature_setting,
        provider_keys,
        config.fireworks_api_base_url,
        identifier=requested_model if allowed else None,
        user=user,
        mock_model_responses=config.mock_model_responses,
    )
    if not allowed:
        log.warning(
            "Flat-rate flow model not allowed, switched to the feature default",
            flow=feature_qualified_name,
            feature_setting=feature_setting,
            requested_model=requested_model,
            served_model=default.llm_definition.gitlab_identifier,
        )

    previous = current_model_metadata_with_size_context.get()
    # Model routing may rewrite a default that came from a feature setting, never a model that the client chose.
    routable = not allowed or (previous is not None and previous.feature_setting)
    metadata_by_tag = ModelMetadataByTag(
        default=default,
        by_tag=build_model_metadata_by_tag(
            feature_setting,
            config.mock_model_responses,
            provider_keys=provider_keys,
            fireworks_api_base_url=config.fireworks_api_base_url,
        ),
        feature_setting=feature_setting if routable else None,
    )
    metadata_by_tag.add_user(user)
    current_model_metadata_context.set(metadata_by_tag.default)
    current_model_metadata_with_size_context.set(metadata_by_tag)


def log_flow_model_feature(
    feature_qualified_name: Optional[str],
    workflow_definition: Optional[str],
    *,
    client_flow: bool,
) -> None:
    """Log the feature of the client's model metadata next to the flow that runs.

    The allowlist checks the model against the client's ``feature_setting``. This line shows, for each flow, which
    feature clients send, which model passed the check, the features of the flow, and what a check against the flow's
    features would do. It changes nothing.

    Nothing is logged for a request without model metadata, and on a gateway that serves custom models.

    Args:
        feature_qualified_name: The name the workflow is billed as, such as ``code_review/v1``.
        workflow_definition: The workflow definition of the start request.
        client_flow: Whether the client sent the flow as a config, inline or as a fetched AI Catalog flow.
    """
    request = dws_client_model_request_context.get()
    if request is None or get_config().custom_models.enabled:
        return

    served = current_model_metadata_context.get()
    try:
        features = flow_features(feature_qualified_name, client_flow=client_flow)
        flow_check = _flow_check(request, served, features)
    except ValueError as error:
        # This line changes nothing, so a broken flows map or feature must not stop the workflow. A pydantic
        # ValidationError is a ValueError too.
        log.warning(
            "Flow features not available",
            flow=feature_qualified_name,
            error=str(error),
        )
        features, flow_check = (), "error"

    log.info(
        "Flow model feature",
        flow=feature_qualified_name,
        workflow_definition=workflow_definition,
        client_feature_setting=request.feature_setting,
        provider=request.provider,
        pinned=request.pinned,
        requested_name=request.name,
        requested_identifier=request.identifier,
        served_identifier=served.llm_definition.gitlab_identifier if served else None,
        client_flow=client_flow,
        flow_features=list(features),
        flow_check=flow_check,
    )
