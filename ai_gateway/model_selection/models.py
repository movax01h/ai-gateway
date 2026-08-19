from enum import StrEnum
from typing import Iterable, Mapping, Optional

from pydantic import BaseModel, ConfigDict, model_validator

__all__ = [
    "BaseModelParams",
    "ChatAmazonQParams",
    "ChatAnthropicParams",
    "ChatLiteLLMParams",
    "ChatOpenAIParams",
    "CompletionLiteLLMParams",
    "CompletionType",
    "ModelClassProvider",
    "OpenAIProviderParams",
    "OpenAIReasoningParams",
    "validate_client_headers",
]


_ALLOWED_CLIENT_HEADER_NAMES = frozenset(
    name.lower()
    for name in (
        "anthropic-beta",
        "anthropic-version",
        "x-trace-id",
    )
)


def validate_client_headers(
    headers: Optional[Mapping[str, str]],
    additional_allowed: Optional[Iterable[str]] = None,
) -> Optional[Mapping[str, str]]:
    """Enforce a fail-closed allowlist over a model request header map.

    Any header name not in ``_ALLOWED_CLIENT_HEADER_NAMES`` (optionally
    extended by ``additional_allowed``) is rejected, so the request never
    reaches a provider transport. Header names are compared case-insensitively
    and after stripping surrounding whitespace.

    Args:
        headers: The header map to validate. ``None`` or empty is allowed and
            returned unchanged.
        additional_allowed: Extra header names to permit for this call, on top
            of the static allowlist (e.g. operator-configured header names from
            trusted server config).

    Returns:
        The ``headers`` argument unchanged when every key is permitted.

    Raises:
        ValueError: If any header name is not on the effective allowlist.
    """
    if not headers:
        return headers

    allowed = _ALLOWED_CLIENT_HEADER_NAMES
    if additional_allowed:
        allowed = allowed | {name.strip().lower() for name in additional_allowed}

    for name in headers:
        if name.strip().lower() not in allowed:
            raise ValueError(
                f"Header '{name}' is not permitted. "
                f"Permitted headers: {sorted(allowed)}"
            )

    return headers


class ModelClassProvider(StrEnum):
    LITE_LLM = "litellm"
    LITE_LLM_COMPLETION = "litellm_completion"
    LITE_LLM_EMBEDDING = "litellm_embedding"
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
    identifier: str | None = None


class ChatAnthropicParams(BaseModelParams):
    default_headers: Mapping[str, str] | None = None

    # This allows us to override the API key per model via `AIGW_MODEL_SELECTION__MODEL_PARAMS`, which enable us to
    # switch between Anthropic organizations/workspaces when needed.
    api_key: str | None = None


class ChatAmazonQParams(BaseModelParams):
    default_headers: Mapping[str, str] | None = None


class OpenAIReasoningParams(BaseModel):
    """Reasoning config for the OpenAI Responses API; unset fields use API defaults.

    The int form of `effort` is deprecated: an account-specific scale kept
    only for Duo Chat's legacy latency tuning (8, below "low").
    """

    model_config = ConfigDict(extra="forbid")

    effort: str | int | None = None
    summary: str | None = None

    @model_validator(mode="after")
    def validate_not_empty(self) -> "OpenAIReasoningParams":
        if self.effort is None and self.summary is None:
            raise ValueError(
                "reasoning must set at least one of effort/summary; "
                "omit the block entirely to use API defaults"
            )
        return self


class OpenAIProviderParams(BaseModel):
    """OpenAI-only params, declared once so the models.yml layer (ChatOpenAIParams) and the prompt layer
    (PromptProviderParams.openai) cannot drift apart."""

    model_config = ConfigDict(extra="forbid")

    verbosity: str | None = None
    reasoning: OpenAIReasoningParams | None = None


class ChatOpenAIParams(BaseModelParams, OpenAIProviderParams):
    """Base model params plus the shared OpenAI provider mixin."""


class ChatGoogleGenAIParams(BaseModelParams):
    thinking_level: str = "low"
    streaming: bool = False


class CompletionLiteLLMParams(BaseModelParams):
    completion_type: CompletionType
    fim_format: str | None = None
    custom_llm_provider: str | None = None
    identifier: str | None = None

    @model_validator(mode="after")
    def validate_fim_format(self) -> "CompletionLiteLLMParams":
        if self.completion_type == CompletionType.FIM and not self.fim_format:
            raise ValueError("fim_format is required when completion_type is 'fim'")
        return self


class EmbeddingLiteLLMParams(BaseModelParams):
    custom_llm_provider: str | None = None
    vertex_ai_location: str | None = None
