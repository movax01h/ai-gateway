from contextvars import ContextVar
from typing import Set

__all__ = ["is_feature_enabled", "current_feature_flag_context", "FeatureFlag"]

from enum import StrEnum


class FeatureFlag(StrEnum):
    # Definition: https://gitlab.com/gitlab-org/gitlab/-/blob/master/config/feature_flags/ops/expanded_ai_logging.yml
    EXPANDED_AI_LOGGING = "expanded_ai_logging"
    ENABLE_ANTHROPIC_PROMPT_CACHING = "enable_anthropic_prompt_caching"
    DISABLE_CODE_GECKO_DEFAULT = "disable_code_gecko_default"
    CHAT_V1_REDIRECT = "redirect_v1_chat_request"
    DUO_CHAT_REACT_AGENT_CLAUDE_4_0 = "duo_chat_react_agent_claude_4_0"
    DUO_WORKFLOW_CLAUDE_SONNET_4 = "duo_workflow_claude_sonnet_4"
    DUO_WORKFLOW_CHAT_WORKFLOW_CLAUDE_SONNET_4 = (
        "duo_workflow_chat_workflow_claude_sonnet_4"
    )
    DUO_WORKFLOW_CHAT_MUTATION_TOOLS = "duo_workflow_chat_mutation_tools"
    DUO_WORKFLOW_MCP_SUPPORT = "duo_workflow_mcp_support"
    BATCH_DUO_WORKFLOW_PLANNER_TASKS = "batch_duo_workflow_planner_tasks"
    DUO_WORKFLOW_COMMIT_TOOLS = "duo_workflow_commit_tools"


def is_feature_enabled(feature_name: FeatureFlag | str) -> bool:
    """Check if a feature is enabled.

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
