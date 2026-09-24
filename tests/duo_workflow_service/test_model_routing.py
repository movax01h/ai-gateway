import pytest
from structlog.testing import capture_logs

from ai_gateway.model_metadata import ModelMetadata, ModelMetadataByTag
from ai_gateway.model_selection import ModelSelectionConfig
from duo_workflow_service.model_routing import route_default_model_by_goal
from lib.context.model import (
    current_model_metadata_context,
    current_model_metadata_with_size_context,
)
from lib.feature_flags.context import current_feature_flag_context


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
@pytest.mark.parametrize(
    ("goal", "expected_tag"),
    [
        ("Fix the typo in the README", "small"),
        ("Add pagination to the issues list", None),
    ],
)
def test_routes_default_by_goal(routable_context, goal, expected_tag):
    assert route_default_model_by_goal(goal) == expected_tag

    expected = (
        routable_context.by_tag[expected_tag]
        if expected_tag
        else routable_context.default
    )
    assert current_model_metadata_with_size_context.get().default is expected
    assert current_model_metadata_context.get() is expected


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
def test_tag_without_a_model_keeps_default(routable_context, monkeypatch):
    monkeypatch.setattr(
        ModelSelectionConfig.instance(),
        "resolve_tag_for_goal",
        lambda *_: ("reasoning", "reason"),
    )

    assert route_default_model_by_goal("anything") is None
    assert current_model_metadata_with_size_context.get() is routable_context


@pytest.mark.usefixtures("flag_on")
def test_logs_matched_keyword_and_from_to_models(routable_context):
    with capture_logs() as cap_logs:
        assert route_default_model_by_goal("Fix the typo in the README") == "small"

    routed_log = next(
        entry for entry in cap_logs if entry["event"] == "Routed default model by goal"
    )
    assert routed_log["tag"] == "small"
    assert routed_log["keyword"] == "typo"
    assert routed_log["from_model"] == "claude_sonnet_4_6_vertex"
    assert routed_log["to_model"] == "claude_haiku_4_5_20251001_vertex"
