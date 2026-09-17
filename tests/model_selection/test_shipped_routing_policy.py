# pylint: disable=file-naming-for-tests
"""Guards on the routing policy shipped in unit_primitives.yml.

Reads the real file so a keyword list spreading to another feature setting is caught.
"""

from pathlib import Path

import pytest
import yaml

from ai_gateway.model_selection import ModelSelectionConfig

_UNIT_PRIMITIVES = (
    Path(__file__).resolve().parents[2]
    / "ai_gateway/model_selection/unit_primitives.yml"
)

# Adding a feature setting here commits to an evaluation for it, see
# gitlab-org/gitlab#627660.
_ROUTED_FEATURE_SETTINGS = {"duo_developer"}

_DUO_DEVELOPER_TIERS = ("large", "small")


@pytest.fixture(name="shipped_tags", scope="module")
def shipped_tags_fixture() -> dict[str, dict]:
    """Every feature setting's raw `models_for_tags` block, straight from the YAML."""
    raw = yaml.safe_load(_UNIT_PRIMITIVES.read_text())
    return {
        entry["feature_setting"]: entry.get("models_for_tags") or {}
        for entry in raw["configurable_unit_primitives"]
    }


@pytest.fixture(name="duo_developer_tags", scope="module")
def duo_developer_tags_fixture(shipped_tags) -> dict[str, dict]:
    return shipped_tags["duo_developer"]


def test_only_duo_developer_carries_routing_keywords(shipped_tags):
    with_keywords = {
        feature_setting
        for feature_setting, tags in shipped_tags.items()
        for tag in tags.values()
        if isinstance(tag, dict) and tag.get("keywords")
    }

    assert with_keywords == _ROUTED_FEATURE_SETTINGS


def test_large_precedes_small_so_a_goal_matching_both_escalates(duo_developer_tags):
    assert list(duo_developer_tags) == list(_DUO_DEVELOPER_TIERS)


@pytest.mark.parametrize("tier", _DUO_DEVELOPER_TIERS)
def test_each_tier_can_be_selected_and_resolved(duo_developer_tags, tier):
    entry = duo_developer_tags[tier]

    assert entry["models"], "a tier with no models cannot resolve"
    assert entry["keywords"], "a tier with no keywords can never be selected"


@pytest.mark.parametrize("tier", _DUO_DEVELOPER_TIERS)
def test_keywords_need_no_cleanup_at_match_time(duo_developer_tags, tier):
    """Lowercase, deduplicated and unpadded, so the router can match them as written."""
    keywords = duo_developer_tags[tier]["keywords"]

    assert keywords == [word.lower() for word in keywords], "mixed case"
    assert len(keywords) == len(set(keywords)), "duplicates"
    assert all(word.strip() == word and word for word in keywords), "padding or empty"


@pytest.mark.parametrize(
    ("tier", "model_id", "keyword"),
    [
        pytest.param("large", "claude_sonnet_4_6_vertex", "refactor", id="large"),
        pytest.param("small", "claude_haiku_4_5_20251001_vertex", "typo", id="small"),
    ],
)
def test_shipped_policy_loads_into_tag_entries(tier, model_id, keyword):
    """The policy parses into ModelTagEntry objects, not just into valid YAML."""
    tags = (
        ModelSelectionConfig.instance()
        .get_unit_primitive_config_map()["duo_developer"]
        .models_for_tags
    )

    assert tags[tier].models == [model_id]
    assert keyword in tags[tier].keywords
