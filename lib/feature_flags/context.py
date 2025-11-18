from contextvars import ContextVar
from enum import StrEnum
from typing import Set

__all__ = ["is_feature_enabled", "current_feature_flag_context", "FeatureFlag"]


class FeatureFlag(StrEnum):
    # Definition: https://gitlab.com/gitlab-org/gitlab/-/blob/master/config/feature_flags/ops/expanded_ai_logging.yml
    EXPANDED_AI_LOGGING = "expanded_ai_logging"
    USE_DUO_CONTEXT_EXCLUSION = "use_duo_context_exclusion"
    DUO_USE_BILLING_ENDPOINT = "duo_use_billing_endpoint"
    USAGE_QUOTA_LEFT_CHECK = "usage_quota_left_check"
    COMPRESS_CHECKPOINT = "duo_workflow_compress_checkpoint"
    AI_GATEWAY_ALLOW_CONVERSATION_CACHING = "ai_gateway_allow_conversation_caching"
    USE_DUO_CHAT_UI_FOR_FLOW = "use_duo_chat_ui_for_flow"


def is_feature_enabled(feature_name: FeatureFlag | str) -> bool:
    """Check if a feature is enabled.

    Args:
        feature_name: The name of the feature. See:
        https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/docs/feature_flags.md
    """
    enabled_feature_flags: Set[str] = current_feature_flag_context.get()
    if isinstance(feature_name, FeatureFlag):
        feature_name = feature_name.value
    return feature_name in enabled_feature_flags


current_feature_flag_context: ContextVar[Set[str]] = ContextVar(
    "current_feature_flag_context", default=set()
)
