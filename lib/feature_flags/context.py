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
    # Gates the experimental `for_each` fan-out, which runs a component's body
    # once per item of a list, see
    # duo_workflow_service/agent_platform/experimental/components/for_each. Not
    # yet defined in gitlab/config/feature_flags -- to be added alongside the
    # first flow config that opts in.
    DAP_FOR_EACH = "dap_for_each"
    # Gates SupervisorAgentComponentV2 (parallel subagent delegation via native
    # LangGraph Send tasks) on top of the per-component `parallel_subagents: true`
    # flow config flag, see
    # duo_workflow_service/agent_platform/v1/components/factory.py. Not yet defined
    # in gitlab/config/feature_flags -- to be added alongside the first flow config
    # that opts in.
    DAP_PARALLEL_SUBAGENTS = "dap_parallel_subagents"
    # Gates attaching customer-defined workspace agents to foundational flows as
    # subagents, see duo_workflow_service/agent_platform/v1/catalog. Definition:
    # https://gitlab.com/gitlab-org/gitlab/-/blob/master/ee/config/feature_flags/wip/dap_workspace_agents.yml
    DAP_WORKSPACE_AGENTS = "dap_workspace_agents"
    DEPENDENCY_BUMP_WEB_SEARCH = "dependency_bump_web_search"
    DUO_CHAT_CLARIFICATION_QUESTION_TOOL = "duo_chat_clarification_question_tool"
    DUO_CHAT_GENERATIVE_UI = "duo_chat_generative_ui"
    DUO_DEVELOPER_MODEL_ROUTING = "duo_developer_model_routing"
    AI_MODEL_RELEASE = "ai_model_release"
    CAP_CODE_COMPLETION_CONTEXT = "cap_code_completion_context"


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
