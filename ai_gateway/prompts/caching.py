from typing import Any, MutableMapping, Optional, override

import structlog
from langchain_core.messages import BaseMessage
from langchain_core.prompt_values import PromptValue
from langchain_core.runnables import Runnable, RunnableConfig

from ai_gateway.prompts.config.models import ModelClassProvider
from lib.prompts.caching import prompt_caching_enabled_in_current_request

log = structlog.stdlib.get_logger("prompts")

CACHE_CONTROL_INJECTION_POINTS_KEY = "cache_control_injection_points"
# Custom field of `cache_control_injection_points` to filter out points when prompt caching is disabled in a request.
REQUIRE_PROMPT_CACHING_ENABLED_IN_REQUEST = "require_prompt_caching_enabled_in_request"


def filter_cache_control_injection_points(model_kwargs: MutableMapping[str, Any]):
    if CACHE_CONTROL_INJECTION_POINTS_KEY not in model_kwargs:
        return

    def _check_valid_point(point: dict):
        if REQUIRE_PROMPT_CACHING_ENABLED_IN_REQUEST not in point:
            return True

        required_value = point.pop(REQUIRE_PROMPT_CACHING_ENABLED_IN_REQUEST)

        if not isinstance(required_value, str):
            return False

        return required_value == prompt_caching_enabled_in_current_request()

    model_kwargs[CACHE_CONTROL_INJECTION_POINTS_KEY] = [
        point
        for point in model_kwargs[CACHE_CONTROL_INJECTION_POINTS_KEY]
        if _check_valid_point(point)
    ]

    log.info(
        "Injected cache control points",
        cache_control_injection_points=model_kwargs[CACHE_CONTROL_INJECTION_POINTS_KEY],
    )


class CacheControlInjectionPointsConverter(Runnable[PromptValue, PromptValue]):
    """Converter to modify the `cache_control_injection_points` LiteLLM param for non-LiteLLM model clients.

    https://docs.litellm.ai/docs/tutorials/prompt_caching
    """

    ANTHROPIC_DEFAULT_CACHE_CONTROL = {"type": "ephemeral", "ttl": "5m"}

    @override
    def invoke(
        self,
        input: PromptValue,
        config: Optional[RunnableConfig] = None,
        cache_control_injection_points: list[dict] | None = None,
        model_class_provider: str | None = None,
        **_kwargs: Any,
    ) -> PromptValue:
        if (
            not cache_control_injection_points
            or not model_class_provider
            or not hasattr(input, "messages")
        ):
            return input

        log.info(
            "Converting cache control injection points",
            model_class_provider=model_class_provider,
            cache_control_injection_points=cache_control_injection_points,
        )

        # Create a deep copy of messages to avoid modifying the original list
        new_input = input.model_copy(deep=True)

        match model_class_provider:
            case ModelClassProvider.ANTHROPIC:
                self._convert_for_anthropic_client(
                    messages=new_input.messages,
                    cache_control_injection_points=cache_control_injection_points,
                )
            case _:
                raise NotImplementedError(
                    "cache_control_injection_points is specified but conversion method is not defined"
                )

        return new_input

    def _convert_for_anthropic_client(
        self,
        messages: list[BaseMessage],
        cache_control_injection_points: list[dict],
    ) -> None:
        for point in cache_control_injection_points:
            if point.get("location") == "message":
                target = messages[point["index"]]
                self._annotate_cache_control(target)

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
