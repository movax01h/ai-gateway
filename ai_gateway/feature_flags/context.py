from contextvars import ContextVar
from typing import Set

__all__ = ["is_feature_enabled", "current_feature_flag_context", "FeatureFlag"]

from enum import StrEnum


class FeatureFlag(StrEnum):
    # Definition: https://gitlab.com/gitlab-org/gitlab/-/blob/master/config/feature_flags/ops/expanded_ai_logging.yml
    EXPANDED_AI_LOGGING = "expanded_ai_logging"
    AI_COMMIT_READER_FOR_CHAT = "ai_commit_reader_for_chat"
    ENABLE_ANTHROPIC_PROMPT_CACHING = "enable_anthropic_prompt_caching"


def is_feature_enabled(feature_name: FeatureFlag | str) -> bool:
    """
    Check if a feature is enabled.

    Args:
        feature_name: The name of the feature.

    See https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/docs/feature_flags.md
    """
    enabled_feature_flags: Set[str] = current_feature_flag_context.get()

    if isinstance(feature_name, FeatureFlag):
        feature_name = feature_name.value

    return feature_name in enabled_feature_flags


current_feature_flag_context: ContextVar[Set[str]] = ContextVar(
    "current_feature_flag_context", default=set()
)
