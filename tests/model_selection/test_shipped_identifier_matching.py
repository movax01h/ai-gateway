# pylint: disable=file-naming-for-tests
"""Check that self-hosted identifiers match the entries shipped in models.yml.

Reads the real models.yml so a renamed or removed entry is caught.
"""

import pytest

from ai_gateway.model_metadata import MATCHED_DEFINITION_FIELDS, MATCHED_PARAMS_FIELDS
from ai_gateway.model_selection import LLMDefinition, ModelSelectionConfig
from ai_gateway.model_selection.identifier_matching import (
    MatchKey,
    normalize_identifier,
)
from ai_gateway.model_selection.model_selection_config import _match_identifier


@pytest.fixture(name="shipped_config", scope="module")
def shipped_config_fixture() -> ModelSelectionConfig:
    return ModelSelectionConfig(default_models_override={})


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        ("bedrock/eu.anthropic.claude-sonnet-4-6", "claude_sonnet_4_6_bedrock"),
        ("bedrock/anthropic.claude-sonnet-4-6", "claude_sonnet_4_6_bedrock"),
        (
            "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            "claude_sonnet_4_5_20250929_bedrock",
        ),
        ("bedrock/us.anthropic.claude-sonnet-4-5-20250929", None),
        ("vertex_ai/claude-sonnet-4-6@default", "claude_sonnet_4_6_vertex"),
        ("vertex_ai/claude-sonnet-4-5@20250929", "claude_sonnet_4_5_20250929_vertex"),
        ("vertex_ai/gemini-3.5-flash", "gemini_3_5_flash_vertex"),
        ("vertex_ai/gemini-3.7-flash", "gemini_3_7_flash_vertex"),
        ("anthropic/claude-sonnet-4-6", "claude_sonnet_4_6"),
        ("anthropic/claude-haiku-4-5-20251001", "claude_haiku_4_5_20251001"),
        ("anthropic/claude-haiku-4-5", None),
        ("openai/gpt-5.5-2026-04-23", "gpt_5_5"),
        ("openai/gpt-5.5", None),
        ("openai/gpt-5", "gpt_5_alias"),
        ("azure/gpt-5", None),
    ],
)
def test_shipped_identifiers_match(shipped_config, identifier, expected):
    result = shipped_config.find_by_identifier(identifier)

    assert (result.gitlab_identifier if result else None) == expected


def test_entries_with_the_same_key_have_equal_copied_fields(shipped_config):
    """The first entry wins on a duplicate key, so duplicates must not disagree."""
    first_by_key: dict[MatchKey, LLMDefinition] = {}
    for definition in shipped_config.get_llm_definitions().values():
        identifier = _match_identifier(definition)
        key = normalize_identifier(identifier) if identifier else None
        if key is None:
            continue

        first = first_by_key.setdefault(key, definition)
        for field in MATCHED_DEFINITION_FIELDS:
            assert getattr(definition, field) == getattr(first, field), (
                f"{definition.gitlab_identifier} and {first.gitlab_identifier} "
                f"differ in {field}"
            )
        for field in MATCHED_PARAMS_FIELDS:
            assert getattr(definition.params, field) == getattr(first.params, field), (
                f"{definition.gitlab_identifier} and {first.gitlab_identifier} "
                f"differ in params.{field}"
            )
