from enum import StrEnum
from pathlib import Path
from typing import NamedTuple, Optional

from ai_gateway.code_suggestions.processing import LanguageId
from ai_gateway.code_suggestions.processing.base import LANGUAGE_COUNTER
from ai_gateway.code_suggestions.processing.ops import (
    lang_from_editor_lang,
    lang_from_filename,
)
from ai_gateway.experimentation import ExperimentTelemetry
from ai_gateway.models import (
    KindAmazonQModel,
    KindAnthropicModel,
    KindLiteLlmModel,
    KindModelProvider,
    KindVertexTextModel,
    ModelMetadata,
)
from ai_gateway.models.base import TokensConsumptionMetadata

__all__ = [
    "KindUseCase",
    "CodeSuggestionsOutput",
    "CodeSuggestionsChunk",
    "ModelProvider",
    "PROVIDERS_MODELS_MAP",
    "USE_CASES_MODELS_MAP",
    "SAAS_PROMPT_MODEL_MAP",
]


class ModelProvider(StrEnum):
    VERTEX_AI = "vertex-ai"
    ANTHROPIC = "anthropic"
    LITELLM = "litellm"


class KindUseCase(StrEnum):
    CODE_COMPLETIONS = "code completions"
    CODE_GENERATIONS = "code generations"


PROVIDERS_MODELS_MAP = {
    KindModelProvider.ANTHROPIC: set(KindAnthropicModel),
    KindModelProvider.VERTEX_AI: set(KindVertexTextModel),
    KindModelProvider.LITELLM: set(KindLiteLlmModel),
    KindModelProvider.MISTRALAI: set(KindLiteLlmModel),
    KindModelProvider.FIREWORKS: set(KindLiteLlmModel),
    KindModelProvider.AMAZON_Q: set(KindAmazonQModel),
}

USE_CASES_MODELS_MAP = {
    KindUseCase.CODE_COMPLETIONS: {
        KindAnthropicModel.CLAUDE_3_5_SONNET,
        KindAnthropicModel.CLAUDE_2_1,
        KindVertexTextModel.CODE_GECKO_002,
        KindLiteLlmModel.CODEGEMMA,
        KindLiteLlmModel.CODELLAMA,
        KindLiteLlmModel.CODESTRAL,
        KindLiteLlmModel.DEEPSEEKCODER,
        KindLiteLlmModel.MISTRAL,
        KindLiteLlmModel.MIXTRAL,
        KindLiteLlmModel.CLAUDE_3,
        KindLiteLlmModel.GPT,
        KindLiteLlmModel.QWEN_2_5,
        KindAmazonQModel.AMAZON_Q,
    },
    KindUseCase.CODE_GENERATIONS: {
        KindAnthropicModel.CLAUDE_2_0,
        KindAnthropicModel.CLAUDE_2_1,
        KindVertexTextModel.CODE_BISON_002,
        KindAnthropicModel.CLAUDE_3_SONNET,
        KindAnthropicModel.CLAUDE_3_5_SONNET,
        KindAnthropicModel.CLAUDE_3_HAIKU,
        KindAnthropicModel.CLAUDE_3_5_HAIKU,
        KindAnthropicModel.CLAUDE_3_5_SONNET_V2,
        KindLiteLlmModel.CODEGEMMA,
        KindLiteLlmModel.CODELLAMA,
        KindLiteLlmModel.CODESTRAL,
        KindLiteLlmModel.DEEPSEEKCODER,
        KindLiteLlmModel.MISTRAL,
        KindLiteLlmModel.MIXTRAL,
        KindLiteLlmModel.CLAUDE_3,
        KindLiteLlmModel.GPT,
        KindLiteLlmModel.CLAUDE_3_5,
    },
}

SAAS_PROMPT_MODEL_MAP = {
    "^1.0.0": {
        "model_provider": ModelProvider.ANTHROPIC,
        "model_version": KindAnthropicModel.CLAUDE_3_5_SONNET,
    },
    "1.0.0": {
        "model_provider": ModelProvider.ANTHROPIC,
        "model_version": KindAnthropicModel.CLAUDE_3_5_SONNET,
    },
    "1.0.1-dev": {
        "model_provider": ModelProvider.ANTHROPIC,
        "model_version": KindAnthropicModel.CLAUDE_3_5_SONNET_V2,
    },
    "2.0.0": {
        "model_provider": ModelProvider.VERTEX_AI,
        "model_version": KindAnthropicModel.CLAUDE_3_5_SONNET,
    },
    "2.0.1": {
        "model_provider": ModelProvider.VERTEX_AI,
        "model_version": KindAnthropicModel.CLAUDE_3_5_SONNET_V2,
    },
}


class CodeSuggestionsOutput(NamedTuple):
    class Metadata(NamedTuple):
        experiments: list[ExperimentTelemetry]
        tokens_consumption_metadata: Optional[TokensConsumptionMetadata] = None

    text: str
    score: float
    model: ModelMetadata
    lang_id: Optional[LanguageId] = None
    metadata: Optional[Metadata] = None

    @property
    def lang(self) -> str:
        return self.lang_id.name.lower() if self.lang_id else ""


class CodeSuggestionsChunk(NamedTuple):
    text: str


def resolve_lang_id(
    file_name: str, editor_lang: Optional[str] = None
) -> Optional[LanguageId]:
    lang_id = lang_from_filename(file_name)

    if lang_id is None and editor_lang:
        lang_id = lang_from_editor_lang(editor_lang)

    return lang_id


# TODO: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/292
def increment_lang_counter(
    filename: str,
    lang_id: Optional[LanguageId] = None,
    editor_lang_id: Optional[str] = None,
):
    labels = {"lang": None, "editor_lang": None}

    if lang_id:
        labels["lang"] = lang_id.name.lower()

    if editor_lang_id:
        labels["editor_lang"] = editor_lang_id

    labels["extension"] = Path(filename).suffix[1:]

    LANGUAGE_COUNTER.labels(**labels).inc()
