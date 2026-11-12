"""Prompt template for decision prompts.

The prompt's own template renders the input, and the questions the caller passes go alongside it.
"""

from operator import itemgetter
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel

from ai_gateway.model_selection.models import ModelClassProvider
from ai_gateway.prompts.config.base import PromptConfig
from lib.prompts.utilities import prompt_template_to_messages

__all__ = ["decision_prompt_template_factory"]


def decision_prompt_template_factory(
    model_provider: ModelClassProvider,  # pylint: disable=unused-argument
    config: PromptConfig,
) -> Runnable[dict[str, Any], dict[str, Any]]:
    """Build the template for a prompt with `operation_type: decision`.

    The registry calls every template factory with `(model_provider, config)`, so this one takes `model_provider`
    without using it.

    Args:
        model_provider: The provider of the prompt's model. Unused.
        config: The prompt definition. Its `prompt_template` renders the decision input.

    Returns:
        A runnable that turns `{**template_variables, "questions": ...}` into the `{"input", "questions"}` request
        that `OpenAIDecisionsModel` takes.
    """
    template = ChatPromptTemplate.from_messages(
        prompt_template_to_messages(config.prompt_template, tool_output_security=False),
        template_format="jinja2",
    )

    return RunnableParallel(
        input=template | RunnableLambda(lambda value: value.to_messages()),
        questions=itemgetter("questions"),
    )
