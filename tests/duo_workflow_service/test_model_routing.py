import asyncio
import json
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.messages import AIMessage
from structlog.testing import capture_logs

from ai_gateway.model_metadata import ModelMetadata, ModelMetadataByTag
from ai_gateway.model_selection import ModelSelectionConfig
from duo_workflow_service import model_routing
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.model_routing import (
    RoutingDecision,
    RoutingOutcome,
    resource_task_text,
    route_default_model,
    track_routing_decision,
)
from duo_workflow_service.workflows.type_definitions import AdditionalContext
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


def _registry(ainvoke):
    registry = Mock()
    registry.get_on_behalf.return_value.ainvoke = ainvoke
    return registry


async def _slow(*_args, **_kwargs):
    await asyncio.sleep(1)


@pytest.mark.asyncio
@pytest.mark.usefixtures("flag_on")
@pytest.mark.parametrize(
    ("ainvoke", "expected_tag"),
    [
        (AsyncMock(return_value=AIMessage(content="small")), "small"),
        (AsyncMock(return_value=AIMessage(content="Large.\nWhy")), "large"),
        (AsyncMock(return_value=AIMessage(content="unsure")), None),
        (AsyncMock(side_effect=_slow), None),
        (AsyncMock(side_effect=RuntimeError("boom")), None),
    ],
)
async def test_route_default_model(
    monkeypatch, routable_context, ainvoke, expected_tag
):
    monkeypatch.setattr(model_routing, "CLASSIFIER_TIMEOUT_S", 0.01)

    decision = await route_default_model(
        "Add pagination", Mock(), None, None, Mock(), _registry(ainvoke)
    )

    tag = decision.tag if decision else None
    assert tag == expected_tag
    expected = routable_context.by_tag[tag] if tag else routable_context.default
    assert current_model_metadata_context.get() is expected


@pytest.mark.asyncio
@pytest.mark.usefixtures("flag_on")
@pytest.mark.parametrize(
    ("goal", "hint_tag", "hint_keyword", "matched_keywords"),
    [
        ("Fix the typo in the README", "small", "typo", ["typo", "readme"]),
        ("Add pagination", None, None, []),
    ],
)
async def test_keywords_are_a_hint_to_the_classifier(
    routable_context, goal, hint_tag, hint_keyword, matched_keywords
):
    ainvoke = AsyncMock(return_value=AIMessage(content="large"))

    with capture_logs() as cap_logs:
        decision = await route_default_model(
            goal, Mock(), None, None, Mock(), _registry(ainvoke)
        )

    assert decision == RoutingDecision(
        feature_setting="duo_developer",
        tag="large",
        matched_keywords=matched_keywords,
        outcome=RoutingOutcome.ROUTED,
        gitlab_identifier="claude_sonnet_4_6_vertex",
        classifier_identifier="claude_haiku_4_5_20251001_vertex",
        params=decision.params,
    )
    inputs = ainvoke.call_args.args[0]
    assert (inputs["hint_tag"], inputs["hint_keyword"]) == (hint_tag, hint_keyword)
    classified = next(e for e in cap_logs if e["event"] == "Classified goal tier")
    assert classified["hint_keyword"] == hint_keyword
    assert classified["from_model"] == "claude_sonnet_4_6_vertex"


@pytest.mark.asyncio
async def test_flag_off_skips_the_classifier(routable_context):
    registry = _registry(AsyncMock())

    assert (
        await route_default_model("Fix the typo", Mock(), None, None, Mock(), registry)
        is None
    )
    assert not registry.get_on_behalf.called
    assert current_model_metadata_with_size_context.get() is routable_context


@pytest.mark.asyncio
@pytest.mark.usefixtures("flag_on")
async def test_pinned_model_is_never_routed(routable_context):
    pinned = routable_context.model_copy(update={"feature_setting": None})
    current_model_metadata_with_size_context.set(pinned)
    registry = _registry(AsyncMock())

    assert (
        await route_default_model("Fix the typo", Mock(), None, None, Mock(), registry)
        is None
    )
    assert not registry.get_on_behalf.called
    assert current_model_metadata_with_size_context.get() is pinned


@pytest.mark.asyncio
@pytest.mark.usefixtures("flag_on")
async def test_no_small_tag_skips_the_classifier(routable_context):
    without_small = routable_context.model_copy(
        update={"by_tag": {"large": routable_context.default}}
    )
    current_model_metadata_with_size_context.set(without_small)
    registry = _registry(AsyncMock())

    assert (
        await route_default_model("Fix the typo", Mock(), None, None, Mock(), registry)
        is None
    )
    assert not registry.get_on_behalf.called


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("envelope", "expected"),
    [
        ({"resource_type": "work_item", "resource_id": "7"}, "Fix typo\n\nIn README"),
        ({"resource_type": "pipeline", "resource_id": "7"}, ""),
        ({"resource_type": "work_item", "resource_id": "../../users"}, ""),
        ({"resource_type": "work_item", "resource_id": "²"}, ""),
        ({"resource_type": "work_item", "resource_id": "７"}, ""),
    ],
)
async def test_resource_task_text(envelope, expected):
    client = Mock()
    client.aget = AsyncMock(
        return_value=GitLabHttpResponse(
            200, json.dumps({"title": "Fix typo", "description": "In README"})
        )
    )
    context = [
        AdditionalContext(
            category="agent_platform_resource_context", content=json.dumps(envelope)
        )
    ]

    assert await resource_task_text(client, {"id": 3}, context) == expected


@pytest.mark.asyncio
async def test_resource_task_text_when_the_request_fails():
    client = Mock()
    client.aget = AsyncMock(return_value=GitLabHttpResponse(404, "{}"))
    context = [
        AdditionalContext(
            category="agent_platform_resource_context",
            content=json.dumps({"resource_type": "work_item", "resource_id": "7"}),
        )
    ]

    assert await resource_task_text(client, {"id": 3}, context) == ""


@pytest.mark.asyncio
@pytest.mark.usefixtures("flag_on")
async def test_classifier_tag_pins_the_routing_model(routable_context):
    classifier = _metadata("claude_haiku_4_5_20251001")
    routable_context.by_tag["classifier"] = classifier
    registry = _registry(AsyncMock(return_value=AIMessage(content="small")))

    decision = await route_default_model(
        "Add pagination", Mock(), None, None, Mock(), registry
    )

    assert registry.get_on_behalf.call_args.kwargs["model_metadata"] is classifier
    assert decision.classifier_identifier == "claude_haiku_4_5_20251001"


@pytest.mark.asyncio
@pytest.mark.usefixtures("flag_on")
async def test_tag_without_a_model_falls_back_to_default(routable_context):
    without_large = routable_context.model_copy(
        update={"by_tag": {"small": routable_context.by_tag["small"]}}
    )
    current_model_metadata_with_size_context.set(without_large)
    registry = _registry(AsyncMock(return_value=AIMessage(content="large")))

    decision = await route_default_model(
        "Add pagination", Mock(), None, None, Mock(), registry
    )

    assert decision.outcome == RoutingOutcome.FALLBACK_DEFAULT
    assert decision.gitlab_identifier == "claude_sonnet_4_6_vertex"
    assert current_model_metadata_with_size_context.get() is without_large


@pytest.mark.asyncio
@pytest.mark.usefixtures("flag_on")
@pytest.mark.parametrize("answer", ["acme", "password=hunter2"])
async def test_answer_outside_the_policy_has_no_decision(routable_context, answer):
    registry = _registry(AsyncMock(return_value=AIMessage(content=answer)))

    assert (
        await route_default_model(
            "Add pagination", Mock(), None, None, Mock(), registry
        )
        is None
    )
    assert current_model_metadata_with_size_context.get() is routable_context


@pytest.mark.asyncio
@pytest.mark.usefixtures("flag_on")
async def test_decision_carries_no_text_from_the_task(routable_context):
    # The task is user content; only policy values may reach the log and event.
    task = (
        "Acme Corp: refactor the migration, fix the typo in the README "
        "and drop password=hunter2 from security.md"
    )
    registry = _registry(AsyncMock(return_value=AIMessage(content="large")))
    policy = ModelSelectionConfig.instance().get_resolved_unit_primitive_config_map()[
        "duo_developer"
    ]
    keywords = {k for entry in policy.models_for_tags.values() for k in entry.keywords}

    decision = await route_default_model(task, Mock(), None, None, Mock(), registry)

    assert decision.matched_keywords
    assert set(decision.matched_keywords) <= keywords
    dumped = decision.model_dump_json().lower()
    for fragment in ("acme", "hunter2", "password", "security.md"):
        assert fragment not in dumped


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
        classifier_identifier="claude_haiku_4_5_20251001_vertex",
        params={"temperature": 0.0},
    )
    extra = {
        "feature_setting": "duo_developer",
        "matched_keywords": ["typo", "readme"],
        "gitlab_identifier": "claude_haiku_4_5_20251001_vertex",
        "classifier_identifier": "claude_haiku_4_5_20251001_vertex",
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


def test_tracking_error_does_not_raise_and_logs():
    decision = RoutingDecision(
        feature_setting="duo_developer",
        tag="small",
        matched_keywords=["typo"],
        outcome=RoutingOutcome.ROUTED,
        gitlab_identifier="claude_haiku_4_5_20251001_vertex",
        classifier_identifier="claude_haiku_4_5_20251001_vertex",
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
