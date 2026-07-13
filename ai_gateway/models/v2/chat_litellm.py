from collections.abc import Callable, Sequence
from typing import Any, Dict, List, Literal, Optional, Tuple, Union, override

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import _ChatModelBinding
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from litellm import AnthropicConfig, OpenAIGPT5Config
from pydantic import BaseModel

from ai_gateway.models.base import validate_custom_endpoint
from ai_gateway.models.v2 import (
    litellm_empty_text_patch,  # noqa: F401  (applies the monkey-patch)
)
from ai_gateway.models.v2._model_compat import (
    remove_trailing_assistant_message,
    supports_assistant_prefill,
)
from ai_gateway.models.v2.litellm_model_registry import (
    register_builtin_models,
    register_external_models,
)
from ai_gateway.vendor.langchain_litellm.litellm import ChatLiteLLM as _LChatLiteLLM

__all__ = ["ChatLiteLLM"]


def _force_gpt_5_max_completion_tokens(kwargs: Dict[str, Any]) -> None:
    """GPT-5 needs max_completion_tokens, but some providers (e.g. custom_openai, azure) send the deprecated max_tokens.
    Pass it via extra_body, which LiteLLM forwards as-is, regardless of provider.

    The GPT-5 model check below is the only guard: any provider hosting a GPT-5 model gets the rewrite.

    See https://github.com/karakeep-app/karakeep/issues/1969.
    """
    model = kwargs.get("model")
    if not model or not OpenAIGPT5Config.is_model_gpt_5_model(model):
        return
    if kwargs.get("max_tokens") is None:
        return
    extra_body = dict(kwargs.get("extra_body") or {})
    extra_body.setdefault("max_completion_tokens", kwargs.pop("max_tokens"))
    kwargs["extra_body"] = extra_body


def _remove_deprecated_temperature_parameters(kwargs: Dict[str, Any]) -> None:
    """Claude Opus 4.7/4.8, Sonnet 5, Fable 5, and Mythos 5 deprecated the temperature parameter. Passing it causes a
    Bad Request response, so we drop it to avoid that.

    https://gitlab.com/gitlab-org/gitlab/-/work_items/601614
    """
    model = kwargs.get("model")
    if not model or not _is_deprecated_temperature_model(model):
        return
    if kwargs.get("temperature") is None:
        return
    del kwargs["temperature"]


def _is_deprecated_temperature_model(model: str) -> bool:
    """Check whether the model deprecated the temperature parameter."""
    if AnthropicConfig._is_opus_4_7_model(model):
        return True
    model_lower = model.lower()
    return any(
        v in model_lower
        for v in (
            "opus-4-8",
            "opus_4_8",
            "opus-4.8",
            "opus_4.8",
            "sonnet-5",
            "sonnet_5",
            "fable-5",
            "fable_5",
            "mythos-5",
            "mythos_5",
        )
    )


# Claude families that still accept an assistant message as the final turn (prefill).
# Mirrors the `supports_assistant_prefill: true` entries in models.yml (Opus 4.1, Opus
# 4.5, Sonnet 4.5, Haiku 4.5) so self-hosted deployments behave the same as the
# GitLab-managed models. Anthropic removed prefill support starting with Claude 4.6 and
# states the removal is permanent, so any Claude model outside this allowlist no longer
# supports prefill. Keep in sync with models.yml.
#
# The datestamped entries are the only pre-4.6 legacy models still available for
# self-hosting; they support prefill but have no models.yml entry, so they are pinned
# here explicitly.
# https://platform.claude.com/docs/en/about-claude/models/migration-guide#breaking-changes
_ASSISTANT_PREFILL_SUPPORTED_MODEL_SUBSTRINGS = (
    "opus-4-1",
    "opus_4_1",
    "opus-4.1",
    "opus_4.1",
    "opus-4-5",
    "opus_4_5",
    "opus-4.5",
    "opus_4.5",
    "sonnet-4-5",
    "sonnet_4_5",
    "sonnet-4.5",
    "sonnet_4.5",
    "haiku-4-5",
    "haiku_4_5",
    "haiku-4.5",
    "haiku_4.5",
    # Legacy pre-4.6 models still available for self-hosting.
    "sonnet-4-20250514",
    "3-haiku-20240307",
    "3-sonnet-20240229",
)


def _model_supports_assistant_prefill(model: str) -> bool:
    """Name-based counterpart to `supports_assistant_prefill` for self-hosted models.

    True means the model accepts an assistant message as the final turn, so the request is left untouched. Non-Claude
    models always support prefill; a Claude model supports it only if it matches the allowlist above.
    """
    model_lower = model.lower()
    if "claude" not in model_lower:
        return True
    return any(v in model_lower for v in _ASSISTANT_PREFILL_SUPPORTED_MODEL_SUBSTRINGS)


def _rewrite_trailing_assistant_prefill(kwargs: Dict[str, Any]) -> None:
    """Re-role a trailing assistant (prefill) turn as a user turn for self-hosted Claude models that do not support
    assistant prefill.

    Claude 4.6+ reject any request whose conversation ends with an assistant turn; Anthropic removed prefill support and
    states the removal is permanent. We skip models that support prefill (`_model_supports_assistant_prefill`, the
    name-based equivalent of the `supports_assistant_prefill: true` allowlist) rather than gating on a "4.6+" version
    check, so self-hosted models mirror the GitLab-managed set exactly. Older Claude models outside the allowlist are
    also rewritten -- a benign re-role -- even though they technically still accept prefill.

    The `_create_message_dicts` gate only fixes models resolvable via `models.yml`; self-hosted models reach here with
    the real identifier in `kwargs["model"]`, so we detect and rewrite at the same point as the temperature strip.

    https://platform.claude.com/docs/en/about-claude/models/migration-guide#breaking-changes
    """
    model = kwargs.get("model")
    if not model or _model_supports_assistant_prefill(model):
        return
    messages = kwargs.get("messages")
    if not messages:
        return
    kwargs["messages"] = remove_trailing_assistant_message({"messages": messages})[
        "messages"
    ]


# Register built-in model metadata for models that ship in our model_selection
# registry but are absent from the pinned LiteLLM bundled `model_cost` registry
# (e.g., Claude Opus 4.8 Bedrock until the next LiteLLM bump).
register_builtin_models()

# Register any additional model metadata supplied by operators via an external
# JSON file (path set by AIGW_LITELLM__MODEL_METADATA_FILE). It allows operators
# to enable parameters like `tool_choice` for models that are missing from
# LiteLLM's built-in registry, without running a LiteLLM proxy.
register_external_models()


class ChatLiteLLM(_LChatLiteLLM):
    custom_models_enabled: bool = False
    allowed_api_bases: frozenset[str] = frozenset()

    @override
    def bind(self, **kwargs: Any) -> _ChatModelBinding:
        validate_custom_endpoint(
            self.custom_models_enabled,
            api_base=kwargs.get("api_base"),
            api_key=kwargs.get("api_key"),
            allowed_api_bases=self.allowed_api_bases,
            custom_llm_provider=kwargs.get("custom_llm_provider")
            or self.custom_llm_provider,
        )
        return super().bind(**kwargs)

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type[BaseModel] | Callable | BaseTool],
        tool_choice: Optional[
            Union[dict, str, Literal["auto", "none", "required", "any"], bool]
        ] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        kwargs.pop("web_search_options", None)  # Not yet supported for LiteLLM

        return super().bind_tools(tools, tool_choice=tool_choice, **kwargs)

    @override
    def _create_message_dicts(
        self, messages: List[BaseMessage], stop: Optional[List[str]]
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        message_dicts, params = super()._create_message_dicts(messages, stop)
        model_name = self.model_name or self.model
        if not supports_assistant_prefill(model_name):
            payload = remove_trailing_assistant_message({"messages": message_dicts})
            message_dicts = payload["messages"]
        return message_dicts, params

    @override
    async def acompletion_with_retry(
        self,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        # kwargs has the merged model, provider, and max_tokens by this point.
        _force_gpt_5_max_completion_tokens(kwargs)
        _remove_deprecated_temperature_parameters(kwargs)
        _rewrite_trailing_assistant_prefill(kwargs)
        return await super().acompletion_with_retry(run_manager=run_manager, **kwargs)

    @property
    @override
    def _client_params(self) -> dict[str, Any]:
        """Ensure api_key is passed as a kwarg to async litellm.acompletion.

        Upstream _client_params mutates self.client.api_key but does not include api_key in the returned kwargs. That
        works for the sync path (module-level litellm.api_key) but fails for async acompletion, which reads api_key from
        kwargs.
        """
        params = super()._client_params
        if self.api_key and "api_key" not in params:
            params["api_key"] = self.api_key
        return params
