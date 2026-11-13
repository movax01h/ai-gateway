"""Serve pinned code suggestion models that their provider retires from a configured replacement."""

import structlog

from ai_gateway.config import get_config

log = structlog.stdlib.get_logger("retired_models")

# Chat and review keep the models they pinned.
_CODE_SUGGESTION_FEATURE_SETTINGS = frozenset({"code_completions", "code_generations"})


def remap_retired_identifier(identifier: str | None) -> str | None:
    """Return the configured replacement for a GitLab model identifier.

    Args:
        identifier: The model identifier from the request.

    Returns:
        The replacement from `AIGW_MODEL_SELECTION__IDENTIFIER_ALIASES`, or the input unchanged if it has none.
    """
    if identifier is None:
        return None
    replacement = get_config().model_selection.identifier_aliases.get(identifier)
    if replacement is None:
        return identifier
    log.info("Remapped a retired model", requested=identifier, served=replacement)
    return replacement


def remap_code_suggestion_metadata(metadata: dict) -> dict:
    """Remap an aliased identifier in a raw model_metadata block.

    The block is untyped JSON, so non-string values are returned unchanged.

    Args:
        metadata: The model_metadata block from the request body.

    Returns:
        The metadata with its identifier remapped, or the input unchanged.
    """
    feature_setting = metadata.get("feature_setting")
    identifier = metadata.get("identifier")
    if not isinstance(feature_setting, str) or not isinstance(identifier, str):
        return metadata
    if feature_setting not in _CODE_SUGGESTION_FEATURE_SETTINGS:
        return metadata
    return {**metadata, "identifier": remap_retired_identifier(identifier)}
