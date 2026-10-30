# pylint: disable=file-naming-for-tests
"""Check the system message layout that the models shipped in models.yml get.

Reads the real models.yml, so a new Claude entry without ``uses_cache_breakpoints`` is caught here instead of
silently losing its cross-session prompt cache.
"""

import pytest

from ai_gateway.model_metadata import create_model_metadata
from ai_gateway.model_selection import ModelSelectionConfig
from ai_gateway.prompts.caching import merges_system_messages


@pytest.fixture(name="shipped_config", scope="module")
def shipped_config_fixture() -> ModelSelectionConfig:
    return ModelSelectionConfig(default_models_override={})


def _looks_like_claude(definition) -> bool:
    model = getattr(definition.params, "model", None) or ""
    names = [definition.gitlab_identifier, definition.name, model, *definition.family]
    return any("claude" in name.lower() for name in names)


def test_every_claude_entry_uses_cache_breakpoints(shipped_config):
    missing = [
        identifier
        for identifier, definition in shipped_config.get_llm_definitions().items()
        if _looks_like_claude(definition) and not definition.uses_cache_breakpoints
    ]

    assert not missing, (
        f"Set `uses_cache_breakpoints: true` on these Claude entries: {missing}"
    )


def test_only_claude_entries_use_cache_breakpoints(shipped_config):
    unexpected = [
        identifier
        for identifier, definition in shipped_config.get_llm_definitions().items()
        if definition.uses_cache_breakpoints and not _looks_like_claude(definition)
    ]

    assert not unexpected


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        # Self-hosted Claude, with and without an identifier that matches a shipped entry.
        (
            {
                "provider": "bedrock",
                "name": "claude_3",
                "identifier": "bedrock/eu.anthropic.claude-sonnet-4-6",
            },
            False,
        ),
        (
            {
                "provider": "bedrock",
                "name": "claude_3",
                "identifier": "bedrock/anthropic.claude-unreleased-9",
            },
            False,
        ),
        (
            {
                "provider": "vertex_ai",
                "name": "claude_3",
                "identifier": "vertex_ai/claude-sonnet-4-6@default",
            },
            False,
        ),
        # The general family picks up the flag from the matched Claude entry.
        (
            {
                "provider": "bedrock",
                "name": "general",
                "identifier": "bedrock/eu.anthropic.claude-sonnet-4-6",
            },
            False,
        ),
        # The general family with a model we don't know stays merged, as for other non-Claude models.
        (
            {
                "provider": "openai",
                "name": "general",
                "identifier": "custom_openai/my-model",
            },
            True,
        ),
        (
            {
                "provider": "openai",
                "name": "mistral",
                "identifier": "custom_openai/mistral-small",
            },
            True,
        ),
        (
            {"provider": "openai", "name": "qwen", "identifier": "custom_openai/qwen3"},
            True,
        ),
        # GitLab-managed models, as sent by any GitLab version.
        ({"provider": "gitlab", "identifier": "claude_opus_4_6_bedrock"}, False),
        ({"provider": "gitlab", "identifier": "claude_sonnet_4_6_vertex"}, False),
        ({"provider": "gitlab", "identifier": "gpt_5_5"}, True),
    ],
)
def test_layout_for_shipped_models(data, expected):
    metadata = create_model_metadata(data)
    llm_definition = metadata.llm_definition

    assert (
        merges_system_messages(llm_definition.model_class_provider, llm_definition)
        is expected
    )
