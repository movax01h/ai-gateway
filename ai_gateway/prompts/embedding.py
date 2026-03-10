"""Prompt template factory for embedding models.

This module provides a passthrough prompt template that allows the EmbeddingLiteLLM model to receive `contents` input.
"""

from typing import Any

from langchain_core.runnables import Runnable, RunnableLambda

from ai_gateway.model_selection.models import ModelClassProvider
from ai_gateway.prompts.config.base import PromptConfig

__all__ = ["embedding_prompt_template_factory"]


def embedding_prompt_template_factory(
    model_provider: ModelClassProvider,  # pylint: disable=unused-argument
    config: PromptConfig,  # pylint: disable=unused-argument
) -> Runnable[Any, dict[str, Any]]:
    """Create a passthrough prompt template for embedding models.

    For embedding models using EmbeddingLiteLLM do not require a prompt.
    This template simply passes through the inputs (`contents`) to the model.

    Returns:
        A Runnable that passes through the input dict to the model
    """

    def _passthrough(inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            "contents": inputs.get("contents", []),
            **{k: v for k, v in inputs.items() if k not in ("contents")},
        }

    return RunnableLambda(_passthrough)
