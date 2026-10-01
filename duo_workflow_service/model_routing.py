"""Per-request model routing for Duo Developer.

v1 reads the goal of a `StartWorkflowRequest`, picks the `models_for_tags` tag
whose keywords appear in it, and makes that tag's model the request default.
"""

from enum import StrEnum
from typing import Any, Optional

import structlog
from pydantic import BaseModel

from ai_gateway.model_metadata import TypeModelMetadata
from ai_gateway.model_selection import ModelSelectionConfig
from duo_workflow_service.gitlab.gitlab_api import extract_id_from_global_id
from lib.context.model import (
    current_model_metadata_context,
    current_model_metadata_with_size_context,
)
from lib.feature_flags.context import FeatureFlag, is_feature_enabled
from lib.internal_events.ai_context import AIContext
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import InternalEventAdditionalProperties
from lib.internal_events.event_enum import EventEnum

log = structlog.stdlib.get_logger("model_routing")

# Sampling parameters reported on the routing event. Other model params (model name, max_tokens, retries, headers)
# don't vary with the routing decision.
_REPORTED_PARAMS = ("temperature", "top_p", "top_k")


class RoutingOutcome(StrEnum):
    """What happened after a goal matched a routing tag.

    Attributes:
        ROUTED: The tag resolved to its own model, which is now the request's default.
        FALLBACK_DEFAULT: The tag has no resolvable model, so the request stays on the configured default.
    """

    ROUTED = "routed"
    FALLBACK_DEFAULT = "fallback_default"


class RoutingDecision(BaseModel):
    """The routing decision made for one request whose goal matched a tag.

    Returned by `route_default_model_by_goal` and emitted by `track_routing_decision`. Requests that take a bypass path
    (flag off, pinned model, no matching tag) have no decision.

    Attributes:
        feature_setting: The feature setting whose `models_for_tags` policy was matched.
        tag: The tag the goal matched.
        matched_keywords: Every keyword of the tag found in the goal. Each one alone is enough to select the tag; the
            order follows the policy in `unit_primitives.yml`.
        outcome: Whether the tag's model was applied or the request fell back to the default.
        gitlab_identifier: The model the request is served by: the tag's model when routed, the default when it fell
            back.
        params: Sampling parameters (`temperature`, `top_p`, `top_k`) of that model.
    """

    feature_setting: str
    tag: str
    matched_keywords: list[str]
    outcome: RoutingOutcome
    gitlab_identifier: str
    params: dict[str, Any]


def _sampling_params(metadata: TypeModelMetadata) -> dict[str, Any]:
    params = metadata.llm_definition.params
    return {
        name: getattr(params, name)
        for name in _REPORTED_PARAMS
        if getattr(params, name, None) is not None
    }


def route_default_model_by_goal(goal: str) -> Optional[RoutingDecision]:
    """Rewrite the request's default model from the goal's routing tag.

    Runs once per request, before the flow is built. Returns the decision when a tag matched, including a tag that has
    no resolvable model and so falls back to the default (the context is left untouched in that case). Returns None on
    every bypass path: the flag is off, the request pinned a model, no tag matched, or routing raised.
    """
    try:
        return _route_default_model_by_goal(goal)
    except Exception:
        # Routing is an optimisation; a failure must not block the workflow from starting.
        log.warning("Model routing failed; keeping the default", exc_info=True)
        return None


def _route_default_model_by_goal(goal: str) -> Optional[RoutingDecision]:
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
    tag, matched_keywords = match

    routed = metadata_by_tag.by_tag.get(tag)
    if routed is None:
        served = metadata_by_tag.default
        outcome = RoutingOutcome.FALLBACK_DEFAULT
    else:
        current_model_metadata_with_size_context.set(
            metadata_by_tag.model_copy(update={"default": routed})
        )
        current_model_metadata_context.set(routed)
        served = routed
        outcome = RoutingOutcome.ROUTED

    return RoutingDecision(
        feature_setting=feature_setting,
        tag=tag,
        matched_keywords=matched_keywords,
        outcome=outcome,
        gitlab_identifier=served.llm_definition.gitlab_identifier,
        params=_sampling_params(served),
    )


def track_routing_decision(
    decision: RoutingDecision,
    workflow_id: str,
    internal_event_client: InternalEventsClient,
) -> None:
    """Emit one log line and one internal event for a routing decision.

    `workflow_id` goes in the AI context and is the join key to the billing
    event, which records the model that actually served the workflow under `metadata.workflow_id`.
    """
    try:
        _track_routing_decision(decision, workflow_id, internal_event_client)
    except Exception:
        # Telemetry is best-effort; a failure must not block the workflow from starting.
        log.warning("Model routing telemetry failed", exc_info=True)


def _track_routing_decision(
    decision: RoutingDecision,
    workflow_id: str,
    internal_event_client: InternalEventsClient,
) -> None:
    fields = decision.model_dump(mode="json")

    if decision.outcome == RoutingOutcome.FALLBACK_DEFAULT:
        log.warning(
            "Routing tag has no resolvable model; keeping the default",
            workflow_id=workflow_id,
            **fields,
        )
    else:
        log.info("Routed model by goal", workflow_id=workflow_id, **fields)

    internal_event_client.track_event(
        event_name=EventEnum.WORKFLOW_MODEL_ROUTING_DECISION.value,
        additional_properties=InternalEventAdditionalProperties(
            label=decision.tag,
            property=decision.outcome.value,
            value=extract_id_from_global_id(workflow_id),
            **decision.model_dump(mode="json", exclude={"tag", "outcome"}),
        ),
        category=__name__,
        ai_context=AIContext(workflow_id=workflow_id),
    )
