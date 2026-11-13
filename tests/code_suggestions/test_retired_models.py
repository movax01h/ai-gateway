import pytest
from structlog.testing import capture_logs

from ai_gateway.code_suggestions.retired_models import (
    remap_code_suggestion_metadata,
    remap_retired_identifier,
)


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("claude_sonnet_4_5_20250929", "claude_sonnet_4_5_20250929_bedrock"),
        ("claude_sonnet_4_5_20250929_vertex", "claude_sonnet_4_5_20250929_bedrock"),
        ("codestral_2508_vertex", "codestral_2508_vertex"),
        (None, None),
    ],
)
def test_remap_retired_identifier(identifier, expected):
    assert remap_retired_identifier(identifier) == expected


@pytest.mark.parametrize(
    ("identifier", "expected_logs"),
    [
        (
            "claude_sonnet_4_5_20250929",
            [
                {
                    "event": "Remapped a retired model",
                    "log_level": "info",
                    "requested": "claude_sonnet_4_5_20250929",
                    "served": "claude_sonnet_4_5_20250929_bedrock",
                }
            ],
        ),
        ("codestral_2508_vertex", []),
    ],
)
def test_remap_retired_identifier_logs_only_a_remap(identifier, expected_logs):
    with capture_logs() as logs:
        remap_retired_identifier(identifier)

    assert logs == expected_logs


@pytest.mark.parametrize(
    ("feature_setting", "identifier", "expected"),
    [
        (
            "code_completions",
            "claude_sonnet_4_5_20250929",
            "claude_sonnet_4_5_20250929_bedrock",
        ),
        (
            "code_generations",
            "claude_sonnet_4_5_20250929",
            "claude_sonnet_4_5_20250929_bedrock",
        ),
        (
            "code_generations",
            "claude_sonnet_4_5_20250929_vertex",
            "claude_sonnet_4_5_20250929_bedrock",
        ),
        ("code_completions", "claude_sonnet_5_5", "claude_sonnet_5_5"),
        (
            "review_merge_request",
            "claude_sonnet_4_5_20250929",
            "claude_sonnet_4_5_20250929",
        ),
        ("duo_chat", "claude_sonnet_4_5_20250929", "claude_sonnet_4_5_20250929"),
    ],
)
def test_remap_code_suggestion_metadata(feature_setting, identifier, expected):
    metadata = {"feature_setting": feature_setting, "identifier": identifier}

    assert remap_code_suggestion_metadata(metadata)["identifier"] == expected


@pytest.mark.parametrize(
    "metadata",
    [
        {
            "feature_setting": ["code_completions"],
            "identifier": "claude_sonnet_4_5_20250929",
        },
        {
            "feature_setting": "code_completions",
            "identifier": ["claude_sonnet_4_5_20250929"],
        },
        {"feature_setting": "code_completions"},
    ],
)
def test_remap_code_suggestion_metadata_ignores_unexpected_values(metadata):
    assert remap_code_suggestion_metadata(metadata) is metadata


def test_remap_reads_the_configured_aliases(monkeypatch):
    from ai_gateway.config import get_config

    monkeypatch.setattr(
        get_config().model_selection,
        "identifier_aliases",
        {"claude_sonnet_4_5_20250929": "claude_sonnet_4_6"},
    )

    assert remap_retired_identifier("claude_sonnet_4_5_20250929") == "claude_sonnet_4_6"
    assert (
        remap_retired_identifier("claude_sonnet_4_5_20250929_vertex")
        == "claude_sonnet_4_5_20250929_vertex"
    )


@pytest.mark.parametrize("feature_setting", ["code_completions", "code_generations"])
def test_default_alias_targets_are_selectable(feature_setting):
    from ai_gateway.config import get_config
    from ai_gateway.model_selection import ModelSelectionConfig

    allowed = ModelSelectionConfig.instance().allowed_models_for_feature(
        feature_setting
    )

    assert set(get_config().model_selection.identifier_aliases.values()) <= set(allowed)
