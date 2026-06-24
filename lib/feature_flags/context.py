from contextvars import ContextVar
from enum import StrEnum
from typing import Set

__all__ = ["FeatureFlag", "current_feature_flag_context", "is_feature_enabled"]


class FeatureFlag(StrEnum):
    # Definition: https://gitlab.com/gitlab-org/gitlab/-/blob/master/config/feature_flags/ops/expanded_ai_logging.yml
    EXPANDED_AI_LOGGING = "expanded_ai_logging"
    USE_GENERIC_GITLAB_API_TOOLS = "use_generic_gitlab_api_tools"
    AI_PROMPT_SCANNING = "ai_prompt_scanning"
    DAP_WEB_SEARCH = "dap_web_search"
    AGENTIC_FOUNDATIONAL_FLOW_TOOL = "agentic_foundational_flow_tool"
    DUO_CHAT_CLARIFICATION_QUESTION_TOOL = "duo_chat_clarification_question_tool"
    # Definition:
    # https://gitlab.com/gitlab-org/gitlab/-/blob/master/config/feature_flags/gitlab_com_derisk/ai_gateway_multi_default_models.yml
    AI_GATEWAY_MULTI_DEFAULT_MODELS = "ai_gateway_multi_default_models"


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
