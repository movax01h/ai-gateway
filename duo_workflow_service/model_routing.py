"""Per-request model routing for Duo Developer.

v1 reads the goal of a `StartWorkflowRequest`, picks the `models_for_tags` tag
whose keywords appear in it, and makes that tag's model the request default.
"""

from typing import Optional

import structlog

from ai_gateway.model_selection import ModelSelectionConfig
from lib.context.model import (
    current_model_metadata_context,
    current_model_metadata_with_size_context,
)
from lib.feature_flags.context import FeatureFlag, is_feature_enabled

log = structlog.stdlib.get_logger("model_routing")


def route_default_model_by_goal(goal: str) -> Optional[str]:
    """Rewrite the request's default model from the goal's routing tag.

    Runs once per request, before the flow is built. Returns the tag applied, or None when the flag is off, the request
    pinned a model, no tag matched, or the matched tag has no resolvable model. In every None case the context is left
    untouched and the flow runs on the configured default.
    """
    if not is_feature_enabled(FeatureFlag.DUO_DEVELOPER_MODEL_ROUTING):
        return None

    metadata_by_tag = current_model_metadata_with_size_context.get()
    if metadata_by_tag is None or not metadata_by_tag.feature_setting:
        return None

    feature_setting = metadata_by_tag.feature_setting
    # TODO: when no keyword matches, ask a small LLM classifier for the tag.
    # https://gitlab.com/gitlab-org/gitlab/-/work_items/630219
    match = ModelSelectionConfig.instance().resolve_tag_for_goal(feature_setting, goal)
    if match is None:
        return None
    tag, keyword = match

    routed = metadata_by_tag.by_tag.get(tag)
    if routed is None:
        log.warning(
            "Routing tag has no resolvable model; keeping the default",
            feature_setting=feature_setting,
            tag=tag,
            keyword=keyword,
        )
        return None

    current_model_metadata_with_size_context.set(
        metadata_by_tag.model_copy(update={"default": routed})
    )
    current_model_metadata_context.set(routed)
    log.info(
        "Routed default model by goal",
        feature_setting=feature_setting,
        tag=tag,
        keyword=keyword,
        from_model=metadata_by_tag.default.llm_definition.gitlab_identifier,
        to_model=routed.llm_definition.gitlab_identifier,
    )
    return tag
