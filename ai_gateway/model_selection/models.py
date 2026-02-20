from enum import StrEnum
from typing import Mapping

from pydantic import BaseModel, ConfigDict, model_validator

__all__ = [
    "ModelClassProvider",
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
    GOOGLE_GENAI = "google_genai"


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
    extra_headers: Mapping[str, str] | None = None


class ChatLiteLLMParams(BaseModelParams):
    custom_llm_provider: str | None = None
    """Easily switch to huggingface, replicate, together ai, sagemaker, etc.
    Example - https://litellm.vercel.app/docs/providers/vllm#batch-completion"""


class ChatAnthropicParams(BaseModelParams):
    default_headers: Mapping[str, str] | None = None


class ChatAmazonQParams(BaseModelParams):
    default_headers: Mapping[str, str] | None = None


class ChatOpenAIParams(BaseModelParams):
    verbosity: str | None = None


class ChatGoogleGenAIParams(BaseModelParams):
    thinking_level: str = "low"
    streaming: bool = False


class CompletionLiteLLMParams(BaseModelParams):
    completion_type: CompletionType
    fim_format: str | None = None
    custom_llm_provider: str | None = None

    @model_validator(mode="after")
    def validate_fim_format(self) -> "CompletionLiteLLMParams":
        if self.completion_type == CompletionType.FIM and not self.fim_format:
            raise ValueError("fim_format is required when completion_type is 'fim'")
        return self
