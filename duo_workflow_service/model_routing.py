"""Per-request model routing for Duo Developer.

Reads the task once, before the flow is built: the triggering issue or merge request when there is
one, otherwise the goal. A small model picks the tag, with any BM25 `models_for_tags` keyword match
passed to it as a hint. The chosen tag's model becomes the request default.
"""

import asyncio
import json
import time
from enum import StrEnum
from typing import Any, Optional

import structlog
from gitlab_cloud_connector import CloudConnectorUser
from pydantic import BaseModel

from ai_gateway.model_metadata import TypeModelMetadata
from ai_gateway.model_selection import ModelSelectionConfig
from ai_gateway.prompts.base import BasePromptRegistry
from duo_workflow_service.gitlab.gitlab_api import Project, extract_id_from_global_id
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.workflows.type_definitions import AdditionalContext
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

CLASSIFIER_PROMPT_ID = "classify_goal_tier"
CLASSIFIER_TIMEOUT_S = 3.0
CLASSIFIER_MAX_CHARS = 6000
RESOURCE_PATHS = {"work_item": "issues", "merge_request": "merge_requests"}

# Sampling parameters reported on the routing event. Other model params (model name, max_tokens, retries, headers)
# don't vary with the routing decision.
_REPORTED_PARAMS = ("temperature", "top_p", "top_k")


class RoutingOutcome(StrEnum):
    """What happened after the classifier picked a routing tag.

    Attributes:
        ROUTED: The tag resolved to its own model, which is now the request's default.
        FALLBACK_DEFAULT: The tag has no resolvable model, so the request stays on the configured default.
    """

    ROUTED = "routed"
    FALLBACK_DEFAULT = "fallback_default"


class RoutingDecision(BaseModel):
    """The routing decision made for one workflow whose task the classifier assigned to a tag.

    Returned by `route_default_model` and emitted by `track_routing_decision`. Requests that take a bypass path
    (flag off, pinned model, an `unsure` answer, a timeout or an error) have no decision.

    Attributes:
        feature_setting: The feature setting whose `models_for_tags` policy was used.
        tag: The tag the classifier picked.
        matched_keywords: The keywords passed to the classifier as a hint: every keyword of the first declared tag
            found in the task, copied from `unit_primitives.yml`. Empty when no keyword matched. They can belong to
            a different tag than `tag` when the classifier overrode the hint.
        outcome: Whether the tag's model was applied or the request fell back to the default.
        gitlab_identifier: The model the request is served by: the tag's model when routed, the default when it fell
            back.
        classifier_identifier: The model that classified the task and picked `tag` (the `classifier` tag's model, or
            the `small` tag's when there is none).
        params: Sampling parameters (`temperature`, `top_p`, `top_k`) of that model.
    """

    feature_setting: str
    tag: str
    matched_keywords: list[str]
    outcome: RoutingOutcome
    gitlab_identifier: str
    classifier_identifier: str
    params: dict[str, Any]


def _sampling_params(metadata: TypeModelMetadata) -> dict[str, Any]:
    params = metadata.llm_definition.params
    return {
        name: getattr(params, name)
        for name in _REPORTED_PARAMS
        if getattr(params, name, None) is not None
    }


async def route_default_model(
    goal: str,
    http_client: GitlabHttpClient,
    project: Optional[Project],
    additional_context: Optional[list[AdditionalContext]],
    user: CloudConnectorUser,
    prompt_registry: BasePromptRegistry,
) -> Optional[RoutingDecision]:
    """Route once per request with the classifier. Any failure keeps the default.

    https://gitlab.com/gitlab-org/gitlab/-/work_items/630219
    """
    if not is_feature_enabled(FeatureFlag.DUO_DEVELOPER_MODEL_ROUTING):
        return None
    try:
        task = (
            await resource_task_text(http_client, project, additional_context) or goal
        )
        return await route_default_model_by_classifier(task, user, prompt_registry)
    except Exception as e:
        log.warning("Model routing failed; keeping the default", error=str(e))
        return None


async def resource_task_text(
    http_client: GitlabHttpClient,
    project: Optional[Project],
    additional_context: Optional[list[AdditionalContext]],
) -> str:
    """Title and description of the triggering issue or merge request, or "" when there is none."""
    envelope = next(
        (
            c
            for c in additional_context or []
            if c.category == "agent_platform_resource_context"
        ),
        None,
    )
    fields = json.loads(envelope.content) if envelope and envelope.content else {}
    path = RESOURCE_PATHS.get(fields.get("resource_type", ""))
    resource_id = str(fields.get("resource_id", ""))
    # The envelope comes from the client, so only an ASCII numeric IID may reach the API path.
    # `isdigit()` alone also accepts digits such as "²" or full-width "７".
    if not (project and path and resource_id.isascii() and resource_id.isdigit()):
        return ""
    response = await http_client.aget(
        path=f"/api/v4/projects/{project['id']}/{path}/{resource_id}",
        parse_json=False,
    )
    if not response.is_success():
        return ""
    body = (
        json.loads(response.body) if isinstance(response.body, str) else response.body
    )
    return f"{body.get('title') or ''}\n\n{body.get('description') or ''}".strip()


async def route_default_model_by_classifier(
    task: str, user: CloudConnectorUser, prompt_registry: BasePromptRegistry
) -> Optional[RoutingDecision]:
    """Ask the classifier (or small) tag's model for a tag.

    Unsure, a timeout or an error keeps the default.
    """
    metadata_by_tag = current_model_metadata_with_size_context.get()
    if metadata_by_tag is None or not metadata_by_tag.feature_setting:
        return None
    config = ModelSelectionConfig.instance()
    feature_setting = metadata_by_tag.feature_setting
    hint_tag, hint_keywords = config.resolve_tag_for_goal(feature_setting, task) or (
        None,
        [],
    )
    hint_keyword = hint_keywords[0] if hint_keywords else None
    # A "classifier" tag pins the routing model; otherwise the small tag decides.
    classifier_model = metadata_by_tag.by_tag.get(
        "classifier", metadata_by_tag.by_tag.get("small")
    )
    if classifier_model is None:
        return None

    started, answer = time.monotonic(), "error"
    try:
        prompt = prompt_registry.get_on_behalf(
            user,
            CLASSIFIER_PROMPT_ID,
            "^1.0.0",
            model_metadata=classifier_model,
            internal_event_extra={"is_routing_call": True},
        )
        message = await asyncio.wait_for(
            prompt.ainvoke(
                {
                    "goal": task[:CLASSIFIER_MAX_CHARS],
                    "hint_tag": hint_tag,
                    "hint_keyword": hint_keyword,
                }
            ),
            CLASSIFIER_TIMEOUT_S,
        )
        words = message.text().lower().split()
        answer = words[0].strip(".:*") if words else ""
    except asyncio.TimeoutError:
        answer = "timeout"

    routed = metadata_by_tag.by_tag.get(answer)
    log.info(
        "Classified goal tier",
        feature_setting=feature_setting,
        answer=answer,
        hint_tag=hint_tag,
        hint_keyword=hint_keyword,
        latency_ms=round((time.monotonic() - started) * 1000),
        from_model=metadata_by_tag.default.llm_definition.gitlab_identifier,
        to_model=routed.llm_definition.gitlab_identifier if routed else None,
    )
    # The answer is model output, so only a tag declared in the policy may reach the event.
    policy = config.get_resolved_unit_primitive_config_map().get(feature_setting)
    if policy is None or answer not in policy.models_for_tags:
        return None
    if routed is None:
        served, outcome = metadata_by_tag.default, RoutingOutcome.FALLBACK_DEFAULT
    else:
        current_model_metadata_with_size_context.set(
            metadata_by_tag.model_copy(update={"default": routed})
        )
        current_model_metadata_context.set(routed)
        served, outcome = routed, RoutingOutcome.ROUTED

    return RoutingDecision(
        feature_setting=feature_setting,
        tag=answer,
        matched_keywords=hint_keywords,
        outcome=outcome,
        gitlab_identifier=served.llm_definition.gitlab_identifier,
        classifier_identifier=classifier_model.llm_definition.gitlab_identifier,
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
