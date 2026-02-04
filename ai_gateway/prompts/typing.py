from typing import Any, Mapping, Optional, Protocol, TypeAlias, runtime_checkable

from langchain_core.runnables import Runnable, RunnableBinding

from ai_gateway.prompts.config.base import PromptConfig


@runtime_checkable
class LLMModelProtocol(Protocol):
    """Protocol for LLM models used in prompts.

    This defines the common interface that both BaseChatModel and CompletionLiteLLM implement, allowing them to be used
    interchangeably in the prompt system.

    Both chat models (BaseChatModel) and completion models (CompletionLiteLLM) satisfy this protocol.
    """

    disable_streaming: bool

    @property
    def _identifying_params(self) -> Mapping[str, Any]: ...

    @property
    def _llm_type(self) -> str: ...

    def bind(self, **kwargs: Any) -> RunnableBinding: ...


# RunnableBinding is included for bound models (after .bind() is called).
# LLMModelProtocol covers both BaseChatModel and completion models like CompletionLiteLLM.
Model: TypeAlias = RunnableBinding | LLMModelProtocol


class TypeModelFactory(Protocol):
    def __call__(self, *, model: str, **kwargs: Optional[Any]) -> LLMModelProtocol: ...


class TypePromptTemplateFactory(Protocol):
    def __call__(self, config: PromptConfig) -> Runnable[Any, Any]: ...
