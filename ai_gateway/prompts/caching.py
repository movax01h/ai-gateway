from typing import Any, Optional

import structlog
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableConfig

from ai_gateway.prompts.config.models import ModelClassProvider

log = structlog.stdlib.get_logger("prompts")


class CacheControlInjectionPointsConverter(Runnable[PromptValue, PromptValue]):
    """Converter to modify the `cache_control_injection_points` LiteLLM param for non-LiteLLM model clients.

    https://docs.litellm.ai/docs/tutorials/prompt_caching
    """

    ANTHROPIC_DEFAULT_CACHE_CONTROL = {"type": "ephemeral", "ttl": "5m"}

    def invoke(
        self,
        input: PromptValue,
        config: Optional[RunnableConfig] = None,  # pylint: disable=unused-argument
        cache_control_injection_points: list[dict] | None = None,
        model_class_provider: str | None = None,
        **_kwargs: Any,
    ) -> PromptValue:
        if not cache_control_injection_points or not model_class_provider:
            return input

        log.info(
            "Converting cache control injection points",
            model_class_provider=model_class_provider,
            cache_control_injection_points=cache_control_injection_points,
        )

        match model_class_provider:
            case ModelClassProvider.ANTHROPIC:
                return self._convert_for_anthropic_client(
                    input=input,
                    cache_control_injection_points=cache_control_injection_points,
                )
            case _:
                raise NotImplementedError(
                    "cache_control_injection_points is specified but conversion method is not defined"
                )

        return input

    def _convert_for_anthropic_client(
        self, input: PromptValue, cache_control_injection_points: list[dict]
    ):
        if not hasattr(input, "messages"):
            return input

        for point in cache_control_injection_points:
            if point.get("location") == "message":
                target = input.messages[point["index"]]
                self._annotate_cache_control(target)

        return input

    def _annotate_cache_control(self, msg: BaseMessage):
        if isinstance(msg.content, str):
            msg.content = [
                {
                    "text": msg.content,
                    "type": "text",
                    "cache_control": self.ANTHROPIC_DEFAULT_CACHE_CONTROL,
                }
            ]
        elif isinstance(msg.content, list):
            last_content = msg.content[-1]

            if isinstance(last_content, str):
                msg.content[-1] = {
                    "text": last_content,
                    "type": "text",
                    "cache_control": self.ANTHROPIC_DEFAULT_CACHE_CONTROL,
                }
            elif isinstance(last_content, dict):
                last_content["cache_control"] = self.ANTHROPIC_DEFAULT_CACHE_CONTROL
