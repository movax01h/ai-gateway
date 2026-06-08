import random
from itertools import chain
from pathlib import Path
from typing import Annotated, Iterable, Literal, Optional

import yaml
from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from ai_gateway.config import get_config
from ai_gateway.model_selection.models import (
    BaseModelParams,
    ChatAmazonQParams,
    ChatAnthropicParams,
    ChatGoogleGenAIParams,
    ChatLiteLLMParams,
    ChatOpenAIParams,
    CompletionLiteLLMParams,
    EmbeddingLiteLLMParams,
    ModelClassProvider,
)
from ai_gateway.model_selection.types import DeprecationInfo, DevConfig
from lib.feature_flags import FeatureFlag, is_feature_enabled

BASE_PATH = Path(__file__).parent
MODELS_CONFIG_PATH = BASE_PATH / "models.yml"
UNIT_PRIMITIVE_CONFIG_PATH = BASE_PATH / "unit_primitives.yml"


class PromptParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stop: list[str] | None = None
    # NOTE: In langchain, some providers accept the timeout when initializing the client. However, support
    # and naming is inconsistent between them. Therefore, we bind the timeout to the prompt instead.
    # See https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/1035#note_2020952732 # pylint: disable=line-too-long
    timeout: float | None = None
    vertex_location: str | None = None
    cache_control_injection_points: list[dict] | None = None
    context_management: dict | None = None


class BaseLLMDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    gitlab_identifier: str
    prompt_params: PromptParams = PromptParams()
    max_context_tokens: int
    provider: Optional[str] = None
    description: str | None = None
    cost_indicator: Literal["$", "$$", "$$$", "$$$$"] | None = None
    params: BaseModelParams
    family: list[str] = []
    deprecation: Optional[DeprecationInfo] = None
    proxy_provider: Optional[str] = None
    # Claude 4.6+ rejects requests ending with an assistant turn (prefill).
    # Opt in by setting to true for models that still accept prefill.
    supports_assistant_prefill: bool = False


class ChatLiteLLMDefinition(BaseLLMDefinition):
    model_class_provider: Literal[ModelClassProvider.LITE_LLM] = (
        ModelClassProvider.LITE_LLM
    )
    params: ChatLiteLLMParams = ChatLiteLLMParams()


class ChatAnthropicDefinition(BaseLLMDefinition):
    model_class_provider: Literal[ModelClassProvider.ANTHROPIC] = (
        ModelClassProvider.ANTHROPIC
    )
    params: ChatAnthropicParams = ChatAnthropicParams()


class ChatAmazonQDefinition(BaseLLMDefinition):
    model_class_provider: Literal[ModelClassProvider.AMAZON_Q] = (
        ModelClassProvider.AMAZON_Q
    )
    params: ChatAmazonQParams = ChatAmazonQParams()


class ChatOpenAIDefinition(BaseLLMDefinition):
    model_class_provider: Literal[ModelClassProvider.OPENAI] = ModelClassProvider.OPENAI
    params: ChatOpenAIParams = ChatOpenAIParams()


class ChatGoogleGenAIDefinition(BaseLLMDefinition):
    model_class_provider: Literal[ModelClassProvider.GOOGLE_GENAI] = (
        ModelClassProvider.GOOGLE_GENAI
    )
    params: ChatGoogleGenAIParams = ChatGoogleGenAIParams()


class CompletionLiteLLMDefinition(BaseLLMDefinition):
    model_class_provider: Literal[ModelClassProvider.LITE_LLM_COMPLETION] = (
        ModelClassProvider.LITE_LLM_COMPLETION
    )
    params: CompletionLiteLLMParams


class EmbeddingLiteLLMDefinition(BaseLLMDefinition):
    model_class_provider: Literal[ModelClassProvider.LITE_LLM_EMBEDDING] = (
        ModelClassProvider.LITE_LLM_EMBEDDING
    )
    params: EmbeddingLiteLLMParams


LLMDefinition = Annotated[
    ChatLiteLLMDefinition
    | ChatAnthropicDefinition
    | ChatAmazonQDefinition
    | ChatOpenAIDefinition
    | ChatGoogleGenAIDefinition
    | CompletionLiteLLMDefinition
    | EmbeddingLiteLLMDefinition,
    Field(discriminator="model_class_provider"),
]


class UnitPrimitiveConfig(BaseModel):
    feature_setting: str
    unit_primitives: list[GitLabUnitPrimitive]
    default_models: list[str] = Field(min_length=1)
    models_for_size_preference: dict[Literal["small", "large"], str] = Field(
        default_factory=dict
    )
    selectable_models: list[str] = Field(default_factory=list)
    beta_models: list[str] = Field(default_factory=list)
    dev: DevConfig | None = None


class ModelSelectionConfig:
    _instance: Optional["ModelSelectionConfig"] = None

    def __init__(
        self,
        default_models_override: dict[str, list[str]],
        model_params_override: dict[str, dict] | None = None,
    ) -> None:
        self._llm_definitions: Optional[dict[str, LLMDefinition]] = None
        self._unit_primitive_configs: Optional[dict[str, UnitPrimitiveConfig]] = None
        self._default_models_override: dict[str, list[str]] = default_models_override
        self._model_params_override: dict[str, dict] = model_params_override or {}

    @classmethod
    def instance(cls) -> "ModelSelectionConfig":
        """Get the singleton instance of ModelSelectionConfig.

        Returns:
            The singleton ModelSelectionConfig instance.
        """
        if cls._instance is None:
            cfg = get_config()
            cls._instance = cls(
                default_models_override=cfg.model_selection.default_models,
                model_params_override=cfg.model_selection.model_params,
            )
        return cls._instance

    def get_llm_definitions(self) -> dict[str, LLMDefinition]:
        if not self._llm_definitions:

            with open(MODELS_CONFIG_PATH, "r") as f:
                config_data = yaml.safe_load(f)

            self._llm_definitions = {}
            for model_data in config_data["models"]:
                identifier = model_data["gitlab_identifier"]
                if identifier in self._model_params_override:
                    params_override = self._model_params_override[identifier]
                    model_data = {
                        **model_data,
                        "params": {**model_data.get("params", {}), **params_override},
                    }
                self._llm_definitions[identifier] = TypeAdapter(
                    LLMDefinition
                ).validate_python(model_data)

        return self._llm_definitions

    def get_unit_primitive_config_map(self) -> dict[str, UnitPrimitiveConfig]:
        if not self._unit_primitive_configs:
            with open(UNIT_PRIMITIVE_CONFIG_PATH, "r") as f:
                config_data = yaml.safe_load(f)

            self._unit_primitive_configs = {
                data["feature_setting"]: UnitPrimitiveConfig(**data)
                for data in config_data["configurable_unit_primitives"]
            }

            for feature_setting, models in self._default_models_override.items():
                if feature_setting in self._unit_primitive_configs:
                    self._unit_primitive_configs[feature_setting].default_models = (
                        models
                    )

        return self._unit_primitive_configs

    def get_unit_primitive_config(self) -> Iterable[UnitPrimitiveConfig]:
        return self.get_unit_primitive_config_map().values()

    def _validate_model_ids_exist(
        self,
        unit_primitive_configs: Iterable[UnitPrimitiveConfig],
        models_ids: set,
    ) -> list[str]:
        errors: set[str] = set()
        for unit_primitive_config in unit_primitive_configs:
            ids = chain(
                unit_primitive_config.default_models,
                unit_primitive_config.models_for_size_preference.values(),
                unit_primitive_config.selectable_models,
                unit_primitive_config.beta_models,
                (
                    unit_primitive_config.dev.selectable_models
                    if unit_primitive_config.dev
                    else []
                ),
            )
            errors.update(model_id for model_id in ids if model_id not in models_ids)
        if errors:
            return [
                f"The following models ids are used but are not defined in models.yml: {', '.join(errors)}"
            ]
        return []

    def _validate_default_models_are_selectable(
        self, unit_primitive_configs: Iterable[UnitPrimitiveConfig]
    ) -> list[str]:
        errors = [
            f"Feature '{upc.feature_setting}' has default model "
            f"'{default_model}' that is not in selectable_models."
            for upc in unit_primitive_configs
            for default_model in upc.default_models
            if upc.selectable_models and default_model not in upc.selectable_models
        ]
        if errors:
            return [
                "Default models must be included in selectable_models:\n"
                + "\n".join(f"  - {error}" for error in errors)
            ]
        return []

    def _validate_selectable_model_required_fields(
        self,
        unit_primitive_configs: Iterable[UnitPrimitiveConfig],
        models: dict,
        models_ids: set,
    ) -> list[str]:
        errors = []
        for upc in unit_primitive_configs:
            for model_id in upc.selectable_models:
                if model_id not in models_ids:
                    continue
                if models[model_id].cost_indicator is None:
                    errors.append(
                        f"Feature '{upc.feature_setting}' has selectable model "
                        f"'{model_id}' without a cost_indicator."
                    )
                if models[model_id].description is None:
                    errors.append(
                        f"Feature '{upc.feature_setting}' has selectable model "
                        f"'{model_id}' without a description."
                    )
        if errors:
            return [
                "Selectable models are missing required fields:\n"
                + "\n".join(f"  - {error}" for error in errors)
            ]
        return []

    def validate(self) -> None:
        unit_primitive_configs = list(self.get_unit_primitive_config())
        models = self.get_llm_definitions()
        models_ids = set(models.keys())

        error_messages = [
            *self._validate_model_ids_exist(unit_primitive_configs, models_ids),
            *self._validate_default_models_are_selectable(unit_primitive_configs),
            *self._validate_selectable_model_required_fields(
                unit_primitive_configs, models, models_ids
            ),
        ]

        if error_messages:
            raise ValueError("\n".join(error_messages))

    def refresh(self):
        """Refresh the configuration by reloading from source files."""
        self._llm_definitions = None
        self._unit_primitive_configs = None

    def get_proxy_models_for_provider(self, provider: str) -> list[str]:
        """Get list of allowed model names for a provider's proxy endpoint.

        Args:
            provider: The provider name (e.g., "anthropic", "openai")

        Returns:
            List of model names allowed for proxy
        """
        llm_definitions = self.get_llm_definitions()
        return [
            llm_def.params.model or ""
            for llm_def in llm_definitions.values()
            if llm_def.proxy_provider == provider and llm_def.params.model
        ]

    def get_model(self, model_id: str) -> LLMDefinition:
        if model := self.get_llm_definitions().get(model_id, None):
            return model
        raise ValueError(f"Invalid model identifier: {model_id}")

    def get_model_for_feature(self, feature_setting_name: str) -> LLMDefinition:
        if feature_setting := self.get_unit_primitive_config_map().get(
            feature_setting_name, None
        ):
            if is_feature_enabled(FeatureFlag.AI_GATEWAY_MULTI_DEFAULT_MODELS):
                return self.get_model(random.choice(feature_setting.default_models))
            return self.get_model(feature_setting.default_models[0])
        raise ValueError(f"Invalid feature setting: {feature_setting_name}")


def validate_model_selection_config():
    ModelSelectionConfig.instance().validate()
