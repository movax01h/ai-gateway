"""Prompt template factory for completion models (FiM/text completion).

This module provides a passthrough prompt template that allows the CompletionLiteLLM model to receive prefix/suffix
inputs directly, handling the FiM formatting internally.
"""

from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from ai_gateway.model_selection.models import ModelClassProvider
from ai_gateway.prompts.config.base import PromptConfig

__all__ = ["completion_prompt_template_factory"]


def completion_prompt_template_factory(
    model_provider: ModelClassProvider,  # pylint: disable=unused-argument
    config: PromptConfig,  # pylint: disable=unused-argument
) -> Runnable[Any, dict[str, Any]]:
    """Create a passthrough prompt template for completion models.

    For completion models using CompletionLiteLLM, the FiM formatting is handled
    by the model itself based on the completion_type and fim_format configuration.
    This template simply passes through the inputs (prefix, suffix, etc.) to the model.

    Args:
        config: The prompt configuration (reserved for future template rendering)

    Returns:
        A Runnable that passes through the input dict to the model
    """

    def _passthrough(inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            "prefix": inputs.get("prefix", ""),
            "suffix": inputs.get("suffix", ""),
            **{k: v for k, v in inputs.items() if k not in ("prefix", "suffix")},
        }

    return RunnableLambda(_passthrough)
