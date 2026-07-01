"""Load and register external LiteLLM model metadata.

This module provides a mechanism to register custom model metadata with LiteLLM
at AI Gateway startup, without requiring a LiteLLM proxy.

When LiteLLM has no built-in metadata for a model (e.g., a self-hosted or
third-party model on Fireworks AI), parameters like ``tool_choice`` may be
stripped or rejected. Operators can supply a JSON file declaring the model's
capabilities, which is then injected into LiteLLM's internal registry via
``litellm.register_model``.

The file path is read from the ``AIGW_LITELLM__MODEL_METADATA_FILE``
environment variable. If the variable is unset, no external metadata is
registered. If the file is missing or invalid, a warning is logged and the
application continues to start (graceful degradation).

Expected JSON structure::

    {
        "models": {
            "fireworks_ai/accounts/gitlab/deployments/my-model": {
                "litellm_provider": "fireworks_ai",
                "mode": "chat",
                "max_input_tokens": 262144,
                "max_output_tokens": 262144,
                "supports_function_calling": true,
                "supports_tool_choice": true,
                "supports_response_schema": true
            }
        }
    }
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import structlog
from litellm import register_model

__all__ = [
    "ENV_VAR_NAME",
    "load_external_model_metadata",
    "register_builtin_models",
    "register_external_models",
]

ENV_VAR_NAME = "AIGW_LITELLM__MODEL_METADATA_FILE"

log = structlog.stdlib.get_logger("litellm_model_registry")

# Models that ship in our model_selection registry but are absent from the
# pinned LiteLLM bundled `model_cost` registry. Anthropic/LiteLLM routinely
# announce "Day 0" support for new Claude releases (e.g.
# https://docs.litellm.ai/blog/claude_opus_4_8) well before that metadata
# lands in a stable, pinned PyPI release, so any model_selection entry that
# outpaces our LiteLLM pin needs to be registered here by hand. Doing so lets
# the AI Gateway route requests without waiting on a LiteLLM dependency bump.
# Capability flags and costs mirror the predecessor entry that already ships
# natively (Opus 4.6 for Opus 4.8, Sonnet 4.6 for Sonnet 5), since each was
# launched at parity. Remove the manual entry once LiteLLM ships a native one
# and the pin is bumped past it.
_OPUS_4_8_BEDROCK_METADATA: Dict[str, Any] = {
    "litellm_provider": "bedrock_converse",
    "mode": "chat",
    "max_input_tokens": 1_000_000,
    "max_output_tokens": 128_000,
    "max_tokens": 128_000,
    "input_cost_per_token": 5e-06,
    "output_cost_per_token": 2.5e-05,
    "cache_creation_input_token_cost": 6.25e-06,
    "cache_read_input_token_cost": 5e-07,
    "supports_function_calling": True,
    "supports_tool_choice": True,
    "supports_response_schema": True,
    "supports_prompt_caching": True,
    "supports_pdf_input": True,
    "supports_vision": True,
    "supports_reasoning": True,
    "supports_computer_use": True,
}

_SONNET_5_BEDROCK_METADATA: Dict[str, Any] = {
    "litellm_provider": "bedrock_converse",
    "mode": "chat",
    "max_input_tokens": 1_000_000,
    "max_output_tokens": 64_000,
    "max_tokens": 64_000,
    "input_cost_per_token": 3e-06,
    "output_cost_per_token": 1.5e-05,
    "cache_creation_input_token_cost": 3.75e-06,
    "cache_read_input_token_cost": 3e-07,
    "supports_function_calling": True,
    "supports_tool_choice": True,
    "supports_response_schema": True,
    "supports_prompt_caching": True,
    "supports_pdf_input": True,
    "supports_vision": True,
    "supports_reasoning": True,
    "supports_computer_use": True,
}

BUILTIN_MODEL_METADATA: Dict[str, Dict[str, Any]] = {
    # Bedrock cross-region inference profiles for Claude Opus 4.8.
    "global.anthropic.claude-opus-4-8-v1:0": _OPUS_4_8_BEDROCK_METADATA,
    "us.anthropic.claude-opus-4-8-v1:0": _OPUS_4_8_BEDROCK_METADATA,
    "eu.anthropic.claude-opus-4-8-v1:0": _OPUS_4_8_BEDROCK_METADATA,
    "bedrock/global.anthropic.claude-opus-4-8": _OPUS_4_8_BEDROCK_METADATA,
    # Bedrock cross-region inference profiles for Claude Sonnet 5.
    "global.anthropic.claude-sonnet-5-v1:0": _SONNET_5_BEDROCK_METADATA,
    "us.anthropic.claude-sonnet-5-v1:0": _SONNET_5_BEDROCK_METADATA,
    "eu.anthropic.claude-sonnet-5-v1:0": _SONNET_5_BEDROCK_METADATA,
    "bedrock/global.anthropic.claude-sonnet-5": _SONNET_5_BEDROCK_METADATA,
}


def register_builtin_models() -> None:
    """Register model metadata for entries that ship in `models.yml` but are missing from the pinned LiteLLM bundled
    registry.

    Safe to call multiple
    times; LiteLLM's `register_model` is idempotent for identical metadata.
    """
    try:
        register_model(BUILTIN_MODEL_METADATA)
    except Exception as exc:  # pylint: disable=broad-except
        log.warning(
            "Failed to register built-in LiteLLM model metadata",
            error=str(exc),
        )
        return

    log.info(
        "Registered built-in LiteLLM models",
        count=len(BUILTIN_MODEL_METADATA),
        models=list(BUILTIN_MODEL_METADATA.keys()),
    )


def load_external_model_metadata(file_path: str) -> Dict[str, Dict[str, Any]]:
    """Load model metadata from an external JSON file.

    Args:
        file_path: Path to the JSON file containing model metadata.

    Returns:
        A dict mapping model identifiers to their metadata dicts, suitable
        for passing to ``litellm.register_model``. Returns an empty dict if
        the file contains no models.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
        ValueError: If the JSON structure is not as expected.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"LiteLLM model metadata file not found: {file_path}")

    with open(path, "r", encoding="utf-8") as fp:
        contents = fp.read()

    # An empty file is treated as "no models to register" rather than a
    # JSON parse error, mirroring the previous YAML behaviour.
    if not contents.strip():
        return {}

    data = json.loads(contents)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object at the top level of {file_path}, "
            f"got {type(data).__name__}"
        )

    models = data.get("models")
    if models is None:
        # `models` key is required by convention; missing key yields no models.
        return {}

    if not isinstance(models, dict):
        raise ValueError(
            f"Expected `models` to be a JSON object in {file_path}, "
            f"got {type(models).__name__}"
        )

    # Validate that each entry is a mapping (LiteLLM expects dicts)
    for model_name, model_info in models.items():
        if not isinstance(model_info, dict):
            raise ValueError(
                f"Expected metadata for model `{model_name}` to be a JSON object, "
                f"got {type(model_info).__name__}"
            )

    return models


def _load_metadata_safely(
    file_path: str,
) -> Optional[Dict[str, Dict[str, Any]]]:
    """Load metadata from ``file_path``, logging and returning ``None`` on failure.

    Each known failure mode is logged with a tailored message so operators can diagnose the misconfiguration from the AI
    Gateway logs.
    """
    try:
        return load_external_model_metadata(file_path)
    except FileNotFoundError:
        log.warning(
            "External LiteLLM model metadata file not found; skipping registration",
            path=file_path,
        )
    except json.JSONDecodeError as exc:
        log.warning(
            "Invalid JSON in external LiteLLM model metadata file; skipping registration",
            path=file_path,
            error=str(exc),
        )
    except ValueError as exc:
        log.warning(
            "Invalid structure in external LiteLLM model metadata file; skipping registration",
            path=file_path,
            error=str(exc),
        )
    except OSError as exc:
        log.warning(
            "Failed to read external LiteLLM model metadata file; skipping registration",
            path=file_path,
            error=str(exc),
        )
    return None


def register_external_models(file_path: Optional[str] = None) -> None:
    """Register external model metadata with LiteLLM.

    Reads the file path from the ``AIGW_LITELLM__MODEL_METADATA_FILE``
    environment variable when ``file_path`` is not provided. Logs a warning
    and returns silently if the file is missing, empty, or invalid - this
    ensures the application can start even when external configuration is
    misconfigured.

    Args:
        file_path: Optional explicit path to the JSON file. When ``None``,
            the path is read from the environment variable.
    """
    if file_path is None:
        # pylint: disable=direct-environment-variable-reference
        file_path = os.getenv(ENV_VAR_NAME)
        # pylint: enable=direct-environment-variable-reference

    if not file_path:
        # No external file configured; this is the default state and not an error.
        return

    metadata = _load_metadata_safely(file_path)
    if metadata is None:
        return

    if not metadata:
        log.info(
            "External LiteLLM model metadata file contains no models",
            path=file_path,
        )
        return

    try:
        register_model(metadata)
    except Exception as exc:  # pylint: disable=broad-except
        # `register_model` raises generic exceptions; we don't want a bad
        # entry to prevent startup, so log and continue.
        log.warning(
            "Failed to register external LiteLLM models; skipping registration",
            path=file_path,
            error=str(exc),
        )
        return

    log.info(
        "Registered external LiteLLM models",
        path=file_path,
        count=len(metadata),
        models=list(metadata.keys()),
    )
