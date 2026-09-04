"""Prompt template variables defaulted from the bound model, so callers need not declare an input for them.

A caller-supplied value still wins: LangChain merges ``partial_variables`` under the invocation kwargs.
"""

from functools import partial
from typing import Any, Callable, Optional

from langchain_core.prompts import BasePromptTemplate

from ai_gateway.model_metadata import TypeModelMetadata
from ai_gateway.model_selection.models import ModelClassProvider
from lib.prompts.caching import prompt_caching_enabled_in_current_request


def model_friendly_name(model_metadata: Optional[TypeModelMetadata]) -> str:
    """Return the model's friendly name, or ``"Unknown"`` when unavailable."""
    friendly_name = model_metadata.friendly_name if model_metadata else None

    return friendly_name or "Unknown"


def should_show_current_time(model_provider: Optional[ModelClassProvider]) -> bool:
    """A changing timestamp is what invalidates OpenAI's automatic prefix cache when a user opts out of caching."""
    user_opted_out_of_caching = prompt_caching_enabled_in_current_request() == "false"

    return model_provider == ModelClassProvider.OPENAI and user_opted_out_of_caching


# Names and their resolvers in one place so the two cannot drift.
_MODEL_VARIABLES: dict[
    str, Callable[[Optional[ModelClassProvider], Optional[TypeModelMetadata]], Any]
] = {
    "model_friendly_name": lambda _provider, metadata: model_friendly_name(metadata),
    "should_show_current_time": lambda provider, _metadata: should_show_current_time(
        provider
    ),
}

MODEL_TEMPLATE_VARIABLES = frozenset(_MODEL_VARIABLES)


def bind_model_variables(
    prompt_template: BasePromptTemplate,
    model_provider: Optional[ModelClassProvider],
    model_metadata: Optional[TypeModelMetadata],
) -> BasePromptTemplate:
    """Bind ``MODEL_TEMPLATE_VARIABLES`` as partial variables."""
    return prompt_template.partial(
        **{
            name: partial(resolve, model_provider, model_metadata)
            for name, resolve in _MODEL_VARIABLES.items()
        }
    )
