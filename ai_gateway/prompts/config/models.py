from enum import StrEnum
from typing import Annotated, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ModelClassProvider",
    "TypeModelParams",
    "BaseModelParams",
    "ChatLiteLLMParams",
    "ChatAnthropicParams",
    "ChatAmazonQParams",
    "ChatOpenAIParams",
    "CompletionLiteLLMParams",
    "CompletionType",
]


class ModelClassProvider(StrEnum):
    LITE_LLM = "litellm"
    LITE_LLM_COMPLETION = "litellm_completion"
    ANTHROPIC = "anthropic"
    AMAZON_Q = "amazon_q"
    OPENAI = "openai"


class CompletionType(StrEnum):
    FIM = "fim"
    TEXT = "text"


class BaseModelParams(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model: str | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    max_tokens: int | None = None
    max_retries: int | None = 1
    model_class_provider: str | None = None
    custom_llm_provider: str | None = None
    extra_headers: Mapping[str, str] | None = None


class ChatLiteLLMParams(BaseModelParams):
    model_class_provider: Literal[ModelClassProvider.LITE_LLM]
    custom_llm_provider: str | None = None
    """Easily switch to huggingface, replicate, together ai, sagemaker, etc.
    Example - https://litellm.vercel.app/docs/providers/vllm#batch-completion"""


class ChatAnthropicParams(BaseModelParams):
    model_class_provider: Literal[ModelClassProvider.ANTHROPIC]
    default_headers: Mapping[str, str] | None = None


class ChatAmazonQParams(BaseModelParams):
    model_class_provider: Literal[ModelClassProvider.AMAZON_Q]
    default_headers: Mapping[str, str] | None = None


class ChatOpenAIParams(BaseModelParams):
    model_class_provider: Literal[ModelClassProvider.OPENAI]


class CompletionLiteLLMParams(BaseModelParams):
    model_class_provider: Literal[ModelClassProvider.LITE_LLM_COMPLETION]
    completion_type: CompletionType
    fim_format: str | None = None
    custom_llm_provider: str | None = None

    @model_validator(mode="after")
    def validate_fim_format(self) -> "CompletionLiteLLMParams":
        if self.completion_type == CompletionType.FIM and not self.fim_format:
            raise ValueError("fim_format is required when completion_type is 'fim'")
        return self


TypeModelParams = Annotated[
    ChatLiteLLMParams
    | ChatAnthropicParams
    | ChatAmazonQParams
    | ChatOpenAIParams
    | CompletionLiteLLMParams,
    Field(discriminator="model_class_provider"),
]
