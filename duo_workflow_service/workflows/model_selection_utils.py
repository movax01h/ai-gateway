import os
from typing import Optional, Union

from ai_gateway.models import KindAnthropicModel
from duo_workflow_service.internal_events.event_enum import CategoryEnum
from duo_workflow_service.llm_factory import AnthropicConfig, VertexConfig
from lib.feature_flags.context import FeatureFlag, current_feature_flag_context


def get_sonnet_4_config_with_feature_flag(
    workflow_type: str,
) -> Optional[Union[AnthropicConfig, VertexConfig]]:
    """Get model configuration based on workflow type and feature flags.

    Determines which model configuration to use by checking workflow-specific
    feature flags. If the appropriate feature flag is enabled, returns Sonnet 4
    configuration (either Vertex or Anthropic API based on environment).
    Otherwise, falls back to the parent implementation.

    Args:
        workflow_type: The type of workflow ("software development" or "chat").

    Returns:
        Union[AnthropicConfig, VertexConfig]: The appropriate model configuration
            based on feature flags and deployment environment.
    """
    feature_flags = current_feature_flag_context.get()
    _vertex_project_id = os.getenv("DUO_WORKFLOW__VERTEX_PROJECT_ID")

    feature_flag_map = {
        CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT.value: FeatureFlag.DUO_WORKFLOW_CLAUDE_SONNET_4.value,
        CategoryEnum.WORKFLOW_CHAT.value: FeatureFlag.DUO_WORKFLOW_CHAT_WORKFLOW_CLAUDE_SONNET_4.value,
    }

    if workflow_type in feature_flag_map:
        feature_flag = feature_flag_map[workflow_type]
        if feature_flag in feature_flags:
            if bool(_vertex_project_id and len(_vertex_project_id) > 1):
                # Use Sonnet 4 on Vertex
                return VertexConfig(
                    model_name=KindAnthropicModel.CLAUDE_SONNET_4_VERTEX.value
                )
            else:
                # Use Sonnet 4 on Anthropic API
                return AnthropicConfig(
                    model_name=KindAnthropicModel.CLAUDE_SONNET_4.value
                )

    # Fall back to parent implementation if flag not set
    return None
