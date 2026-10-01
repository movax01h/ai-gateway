from unittest.mock import Mock

import pytest
from structlog.testing import capture_logs

from ai_gateway.model_metadata import ModelMetadata, ModelMetadataByTag
from ai_gateway.model_selection import ModelSelectionConfig
from duo_workflow_service.model_routing import (
    RoutingDecision,
    RoutingOutcome,
    route_default_model_by_goal,
    track_routing_decision,
)
from lib.context.model import (
    current_model_metadata_context,
    current_model_metadata_with_size_context,
)
from lib.feature_flags.context import current_feature_flag_context
from lib.internal_events.ai_context import AIContext
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import InternalEventAdditionalProperties
from lib.internal_events.event_enum import EventEnum


def _metadata(identifier: str) -> ModelMetadata:
    definition = ModelSelectionConfig.instance().get_model(identifier)
    return ModelMetadata(llm_definition=definition, name=identifier, provider="gitlab")


@pytest.fixture(name="routable_context")
def routable_context_fixture():
    large = _metadata("claude_sonnet_4_6_vertex")
    by_tag = ModelMetadataByTag(
        default=large,
        by_tag={"small": _metadata("claude_haiku_4_5_20251001_vertex"), "large": large},
        feature_setting="duo_developer",
    )
    current_model_metadata_with_size_context.set(by_tag)
    current_model_metadata_context.set(large)
    yield by_tag
    current_model_metadata_with_size_context.set(None)
    current_model_metadata_context.set(None)


@pytest.fixture(name="flag_on")
def flag_on_fixture():
    token = current_feature_flag_context.set({"duo_developer_model_routing"})
    yield
    current_feature_flag_context.reset(token)


@pytest.mark.usefixtures("flag_on")
def test_routes_default_to_the_matched_tag(routable_context):
    decision = route_default_model_by_goal("Fix the typo in the README")

    assert decision == RoutingDecision(
        feature_setting="duo_developer",
        tag="small",
        matched_keywords=["typo", "readme"],
        outcome=RoutingOutcome.ROUTED,
        gitlab_identifier="claude_haiku_4_5_20251001_vertex",
        params={"temperature": 0.0},
    )
    small = routable_context.by_tag["small"]
    assert current_model_metadata_with_size_context.get().default is small
    assert current_model_metadata_context.get() is small


@pytest.mark.usefixtures("flag_on")
def test_no_matching_tag_is_a_no_op(routable_context):
    assert route_default_model_by_goal("Add pagination to the issues list") is None
    assert current_model_metadata_with_size_context.get() is routable_context
    assert current_model_metadata_context.get() is routable_context.default


def test_flag_off_leaves_context_alone(routable_context):
    assert route_default_model_by_goal("Fix the typo") is None
    assert current_model_metadata_with_size_context.get() is routable_context


@pytest.mark.usefixtures("flag_on")
def test_no_metadata_context_is_a_no_op():
    current_model_metadata_with_size_context.set(None)

    assert route_default_model_by_goal("Fix the typo") is None
    assert current_model_metadata_with_size_context.get() is None


@pytest.mark.usefixtures("flag_on")
def test_pinned_model_is_never_routed(routable_context):
    pinned = routable_context.model_copy(update={"feature_setting": None})
    current_model_metadata_with_size_context.set(pinned)

    assert route_default_model_by_goal("Fix the typo") is None
    assert current_model_metadata_with_size_context.get() is pinned


@pytest.mark.usefixtures("flag_on")
def test_tag_without_a_model_falls_back_to_default(routable_context, monkeypatch):
    monkeypatch.setattr(
        ModelSelectionConfig.instance(),
        "resolve_tag_for_goal",
        lambda *_: ("reasoning", ["reason"]),
    )

    decision = route_default_model_by_goal("anything")

    assert decision is not None
    assert decision.tag == "reasoning"
    assert decision.outcome == RoutingOutcome.FALLBACK_DEFAULT
    assert decision.gitlab_identifier == "claude_sonnet_4_6_vertex"
    assert current_model_metadata_with_size_context.get() is routable_context
    assert current_model_metadata_context.get() is routable_context.default


@pytest.mark.parametrize(
    ("outcome", "log_level", "log_event"),
    [
        (RoutingOutcome.ROUTED, "info", "Routed model by goal"),
        (
            RoutingOutcome.FALLBACK_DEFAULT,
            "warning",
            "Routing tag has no resolvable model; keeping the default",
        ),
    ],
)
def test_track_routing_decision_emits_one_log_line_and_one_event(
    outcome, log_level, log_event
):
    decision = RoutingDecision(
        feature_setting="duo_developer",
        tag="small",
        matched_keywords=["typo", "readme"],
        outcome=outcome,
        gitlab_identifier="claude_haiku_4_5_20251001_vertex",
        params={"temperature": 0.0},
    )
    extra = {
        "feature_setting": "duo_developer",
        "matched_keywords": ["typo", "readme"],
        "gitlab_identifier": "claude_haiku_4_5_20251001_vertex",
        "params": {"temperature": 0.0},
    }
    internal_event_client = Mock(spec=InternalEventsClient)

    with capture_logs() as cap_logs:
        track_routing_decision(decision, "123", internal_event_client)

    assert cap_logs == [
        {
            "event": log_event,
            "log_level": log_level,
            "workflow_id": "123",
            "tag": "small",
            "outcome": outcome.value,
            **extra,
        }
    ]
    internal_event_client.track_event.assert_called_once_with(
        event_name=EventEnum.WORKFLOW_MODEL_ROUTING_DECISION.value,
        additional_properties=InternalEventAdditionalProperties(
            label="small",
            property=outcome.value,
            value=123,
            **extra,
        ),
        category="duo_workflow_service.model_routing",
        ai_context=AIContext(workflow_id="123"),
    )


@pytest.mark.usefixtures("flag_on")
def test_routing_error_keeps_default_and_logs(routable_context, monkeypatch):
    def _raise(*_):
        raise RuntimeError("boom")

    monkeypatch.setattr(ModelSelectionConfig.instance(), "resolve_tag_for_goal", _raise)

    with capture_logs() as cap_logs:
        assert route_default_model_by_goal("Fix the typo") is None

    assert current_model_metadata_with_size_context.get() is routable_context
    assert current_model_metadata_context.get() is routable_context.default
    failure_log = next(
        entry
        for entry in cap_logs
        if entry["event"] == "Model routing failed; keeping the default"
    )
    assert failure_log["log_level"] == "warning"
    assert failure_log["exc_info"] is True


def test_tracking_error_does_not_raise_and_logs():
    decision = RoutingDecision(
        feature_setting="duo_developer",
        tag="small",
        matched_keywords=["typo"],
        outcome=RoutingOutcome.ROUTED,
        gitlab_identifier="claude_haiku_4_5_20251001_vertex",
        params={},
    )
    internal_event_client = Mock(spec=InternalEventsClient)
    internal_event_client.track_event.side_effect = RuntimeError("boom")

    with capture_logs() as cap_logs:
        track_routing_decision(decision, "123", internal_event_client)

    failure_log = next(
        entry
        for entry in cap_logs
        if entry["event"] == "Model routing telemetry failed"
    )
    assert failure_log["log_level"] == "warning"
    assert failure_log["exc_info"] is True
