import pytest

from ai_gateway.model_selection.identifier_matching import (
    MatchKey,
    normalize_identifier,
)


@pytest.mark.parametrize(
    ("identifier", "expected"),
    [
        # Bedrock: the region prefix is removed
        (
            "bedrock/eu.anthropic.claude-sonnet-4-6",
            MatchKey("bedrock", "anthropic.claude-sonnet-4-6"),
        ),
        (
            "bedrock/global.anthropic.claude-sonnet-4-6",
            MatchKey("bedrock", "anthropic.claude-sonnet-4-6"),
        ),
        (
            "bedrock/us.anthropic.claude-sonnet-4-6",
            MatchKey("bedrock", "anthropic.claude-sonnet-4-6"),
        ),
        (
            "bedrock/apac.anthropic.claude-sonnet-4-6",
            MatchKey("bedrock", "anthropic.claude-sonnet-4-6"),
        ),
        (
            "bedrock/anthropic.claude-sonnet-4-6",
            MatchKey("bedrock", "anthropic.claude-sonnet-4-6"),
        ),
        # Bedrock: version suffixes are kept
        (
            "bedrock/us.anthropic.claude-sonnet-4-5-20250929-v1:0",
            MatchKey("bedrock", "anthropic.claude-sonnet-4-5-20250929-v1:0"),
        ),
        (
            "bedrock/us.anthropic.claude-sonnet-4-5-20250929",
            MatchKey("bedrock", "anthropic.claude-sonnet-4-5-20250929"),
        ),
        # Bedrock: the `converse/` route is removed
        (
            "bedrock/converse/eu.anthropic.claude-sonnet-4-6",
            MatchKey("bedrock", "anthropic.claude-sonnet-4-6"),
        ),
        (
            "bedrock/converse/anthropic.claude-sonnet-4-6",
            MatchKey("bedrock", "anthropic.claude-sonnet-4-6"),
        ),
        # Bedrock: ARNs and other routes cannot match
        (
            "bedrock/converse/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abcd1234",
            None,
        ),
        (
            "bedrock/arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/abcd1234",
            None,
        ),
        # Vertex: `@default` and `@latest` are removed
        (
            "vertex_ai/claude-sonnet-4-6@default",
            MatchKey("vertex_ai", "claude-sonnet-4-6"),
        ),
        (
            "vertex_ai/claude-sonnet-4-6@latest",
            MatchKey("vertex_ai", "claude-sonnet-4-6"),
        ),
        ("vertex_ai/claude-sonnet-4-6", MatchKey("vertex_ai", "claude-sonnet-4-6")),
        ("vertex_ai/gemini-3.5-flash", MatchKey("vertex_ai", "gemini-3.5-flash")),
        # Vertex: pinned versions are kept
        (
            "vertex_ai/claude-sonnet-4-5@20250929",
            MatchKey("vertex_ai", "claude-sonnet-4-5@20250929"),
        ),
        ("anthropic/claude-sonnet-4-6", MatchKey("anthropic", "claude-sonnet-4-6")),
        (
            "anthropic/claude-haiku-4-5-20251001",
            MatchKey("anthropic", "claude-haiku-4-5-20251001"),
        ),
        ("openai/gpt-5.5-2026-04-23", MatchKey("openai", "gpt-5.5-2026-04-23")),
        ("openai/gpt-5", MatchKey("openai", "gpt-5")),
        ("openai/responses/gpt-5", None),
        # Providers without a normalizer, and identifiers without a provider
        ("azure/my-deployment", None),
        ("custom_openai/Mixtral-8x7B-Instruct-v0.1", None),
        ("Mixtral-8x7B-Instruct-v0.1", None),
        ("bedrock/", None),
        ("bedrock/converse/", None),
        ("vertex_ai/@default", None),
        ("", None),
    ],
)
def test_normalize_identifier(identifier, expected):
    assert normalize_identifier(identifier) == expected


def test_bedrock_and_vertex_keys_differ_for_same_model():
    bedrock = normalize_identifier("bedrock/claude-sonnet-4-6")
    vertex = normalize_identifier("vertex_ai/claude-sonnet-4-6")

    assert bedrock != vertex
