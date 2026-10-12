"""Normalize model identifiers so a self-hosted identifier can match a models.yml entry.

A self-hosted identifier has the LiteLLM form ``<provider>/<model>``, for example
``bedrock/eu.anthropic.claude-sonnet-4-6``. Each provider has a normalizer that
removes the parts of the model ID that do not change the model, such as a Bedrock
region prefix. Two identifiers that give the same ``MatchKey`` refer to the same model.
"""

from typing import Callable, NamedTuple, Optional

__all__ = ["MatchKey", "normalize_identifier"]


class MatchKey(NamedTuple):
    """Key that identifies a model across identifiers with the same provider.

    Attributes:
        provider: The LiteLLM provider prefix, for example ``bedrock`` or ``vertex_ai``.
        model: The normalized model ID, for example ``anthropic.claude-sonnet-4-6`` for
            ``bedrock/eu.anthropic.claude-sonnet-4-6``.
    """

    provider: str
    model: str


_BEDROCK_REGION_PREFIXES = ("global.", "us.", "eu.", "apac.", "jp.", "au.")

_VERTEX_ALIAS_SUFFIXES = ("@default", "@latest")


def _normalize_bedrock(model: str) -> Optional[str]:
    # `converse/` selects the LiteLLM Converse API route. It does not change the model.
    model = model.removeprefix("converse/")

    # Application inference profile ARNs and other routes do not name a model,
    # so they cannot match.
    if "/" in model or model.startswith("arn:"):
        return None

    # Version suffixes such as `-v1:0` are kept and must match exactly.
    for prefix in _BEDROCK_REGION_PREFIXES:
        if model.startswith(prefix):
            return model.removeprefix(prefix)

    return model


def _normalize_vertex(model: str) -> Optional[str]:
    # `@default` and `@latest` are aliases for the current version. Any other
    # `@` suffix (such as `@20250929`) is a pinned version and must match exactly.
    for suffix in _VERTEX_ALIAS_SUFFIXES:
        if model.endswith(suffix):
            return model.removesuffix(suffix)

    return model


def _normalize_exact(model: str) -> Optional[str]:
    if "/" in model:
        return None

    return model


_NORMALIZERS: dict[str, Callable[[str], Optional[str]]] = {
    "anthropic": _normalize_exact,
    "bedrock": _normalize_bedrock,
    "openai": _normalize_exact,
    "vertex_ai": _normalize_vertex,
}


def normalize_identifier(identifier: str) -> Optional[MatchKey]:
    """Return the match key for an identifier, or ``None`` when it cannot be matched.

    Args:
        identifier: A LiteLLM-style identifier, for example ``bedrock/eu.anthropic.claude-sonnet-4-6``.

    Returns:
        The ``MatchKey`` for the identifier. ``None`` when the identifier has no provider
        prefix, the provider has no normalizer, or the model part cannot be matched.
    """
    provider, _, model = identifier.partition("/")
    normalizer = _NORMALIZERS.get(provider)
    if not model or normalizer is None:
        return None

    normalized = normalizer(model)
    if not normalized:
        return None

    return MatchKey(provider=provider, model=normalized)
