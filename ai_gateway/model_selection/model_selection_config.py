import random
from itertools import chain
from pathlib import Path
from typing import Annotated, Iterable, Literal, Optional

import structlog
import yaml
from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from ai_gateway.config import (
    ModelReleaseFeatureAttachment,
    ModelReleasesPayload,
    get_config,
)
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
from ai_gateway.model_selection.types import (
    DeprecationInfo,
    DevConfig,
    FeatureDeprecatedModel,
)
from lib.feature_flags import FeatureFlag, is_feature_enabled

log = structlog.stdlib.get_logger(__name__)


def _safe_error_details(exc: Exception) -> list[dict]:
    """Build error details safe to log — no input values, only field paths and error types."""
    if isinstance(exc, ValidationError):
        return [
            {
                "loc": ".".join(str(p) for p in e["loc"]),
                "type": e["type"],
                "msg": e["msg"],
            }
            for e in exc.errors()
        ]
    return [{"type": type(exc).__name__, "msg": str(exc)}]


def _partition_known(ids: list[str], known: set[str]) -> tuple[list[str], list[str]]:
    """Split ids into (known, unknown) without mutating the input."""
    valid = [m for m in ids if m in known]
    dropped = [m for m in ids if m not in known]
    return valid, dropped


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
    # Bedrock expects the inference profile / model ARN to be passed at
    # invocation time via model_id, not at client initialization.
    # See https://docs.litellm.ai/docs/providers/bedrock#set-via-model_id
    model_id: str | None = None


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
    tags: list[str] = Field(
        default_factory=list,
        description="Semantic tags for this model (e.g. 'small', 'large', 'reasoning'). Used as metadata; resolution is driven by models_for_tags in unit_primitives.yml.",
    )
    deprecation: Optional[DeprecationInfo] = None
    proxy_provider: Optional[str] = None
    # Claude 4.6+ rejects requests ending with an assistant turn (prefill).
    # Opt in by setting to true for models that still accept prefill.
    supports_assistant_prefill: bool = False
    # Some reasoning models (e.g. Qwen) leak <think>...</think> reasoning into responses.
    # When true, the ReAct parser strips that block before it reaches the user.
    strip_reasoning: bool = False
    requires_single_system_message: bool = False
    # Some models return an empty 404 on streaming requests when max_tokens is at the
    # model's max. Opt in to use this model's max_tokens from models.yml
    use_model_max_tokens: bool = False


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
    models_for_tags: dict[str, str] = Field(default_factory=dict)
    selectable_models: list[str] = Field(default_factory=list)
    beta_models: list[str] = Field(default_factory=list)
    deprecated_models: list[FeatureDeprecatedModel] = Field(default_factory=list)
    dev: DevConfig | None = None


class ModelSelectionConfig:
    _instance: Optional["ModelSelectionConfig"] = None

    def __init__(
        self,
        default_models_override: dict[str, list[str]],
        model_params_override: dict[str, dict] | None = None,
        prompt_params_override: dict[str, dict] | None = None,
        model_releases: Optional[str] = None,
    ) -> None:
        self._llm_definitions: Optional[dict[str, LLMDefinition]] = None
        self._unit_primitive_configs: Optional[dict[str, UnitPrimitiveConfig]] = None
        self._default_models_override: dict[str, list[str]] = default_models_override
        self._model_params_override: dict[str, dict] = model_params_override or {}
        self._prompt_params_override: dict[str, dict] = prompt_params_override or {}
        self._env_llm_definitions: dict[str, LLMDefinition] = {}
        self._env_feature_attachments: dict[str, ModelReleaseFeatureAttachment] = {}
        if model_releases:
            self._load_env_releases(model_releases)

    @classmethod
    def instance(cls) -> "ModelSelectionConfig":
        """Get the singleton instance of ModelSelectionConfig.

        Returns:
            The singleton ModelSelectionConfig instance.
        """
        if cls._instance is None:
            cfg = get_config()
            model_releases = cfg.model_selection.model_releases
            cls._instance = cls(
                default_models_override=cfg.model_selection.default_models,
                model_params_override=cfg.model_selection.model_params,
                prompt_params_override=cfg.model_selection.prompt_params,
                model_releases=(
                    model_releases.get_secret_value() if model_releases else None
                ),
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
                if identifier in self._prompt_params_override:
                    prompt_params_override = self._prompt_params_override[identifier]
                    model_data = {
                        **model_data,
                        "prompt_params": {
                            **model_data.get("prompt_params", {}),
                            **prompt_params_override,
                        },
                    }
                self._llm_definitions[identifier] = TypeAdapter(
                    LLMDefinition
                ).validate_python(model_data)

        return self._llm_definitions

    def _load_env_releases(self, model_releases: str) -> None:
        try:
            payload = ModelReleasesPayload.model_validate_json(model_releases)
        except Exception as exc:  # catches json.JSONDecodeError, ValidationError, and anything else  # pylint: disable=broad-except
            log.error(
                "AIGW_MODEL_SELECTION__MODEL_RELEASES failed to parse; env-injected models will not be available",
                errors=_safe_error_details(exc),
            )
            return

        for model_data in payload.models:
            identifier = model_data.get("gitlab_identifier", "<unknown>")
            try:
                definition: LLMDefinition = TypeAdapter(LLMDefinition).validate_python(
                    model_data
                )
                self._env_llm_definitions[definition.gitlab_identifier] = definition
            except Exception as exc:  # catches all Pydantic validation errors to allow warm startup  # pylint: disable=broad-except
                log.error(
                    "Env-injected model definition failed validation; skipping",
                    gitlab_identifier=identifier,
                    validation_errors=_safe_error_details(exc),
                )
        self._env_feature_attachments = dict(payload.feature_attachments)

    def get_resolved_llm_definitions(self) -> dict[str, LLMDefinition]:
        """Get LLM definitions, merged with env-injected releases when enabled.

        Env-injected model definitions apply only when
        ``FeatureFlag.AI_MODEL_RELEASE`` is enabled for the current request,
        and take precedence over ``models.yml`` entries with the same
        ``gitlab_identifier``.

        Returns:
            Mapping of gitlab_identifier to LLMDefinition.
        """
        base = self.get_llm_definitions()
        if (
            not is_feature_enabled(FeatureFlag.AI_MODEL_RELEASE)
            or not self._env_llm_definitions
        ):
            return base
        return {**base, **self._env_llm_definitions}

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
                    self._unit_primitive_configs[
                        feature_setting
                    ].default_models = models

        return self._unit_primitive_configs

    def get_unit_primitive_config(self) -> Iterable[UnitPrimitiveConfig]:
        return self.get_unit_primitive_config_map().values()

    def get_resolved_unit_primitive_config_map(self) -> dict[str, UnitPrimitiveConfig]:
        """Get unit primitive configs, merged with env-injected attachments when enabled.

        Env-injected feature attachments (``selectable_models``,
        ``beta_models``, ``default_models``) apply only when
        ``FeatureFlag.AI_MODEL_RELEASE`` is enabled for the current request,
        and take precedence over ``unit_primitives.yml`` entries for the same
        feature_setting.

        Returns:
            Mapping of feature_setting to UnitPrimitiveConfig.
        """
        base = self.get_unit_primitive_config_map()
        if (
            not is_feature_enabled(FeatureFlag.AI_MODEL_RELEASE)
            or not self._env_feature_attachments
        ):
            return base
        result = dict(base)
        known_ids = set(self.get_resolved_llm_definitions().keys())
        for feature_setting, attachment in self._env_feature_attachments.items():
            if feature_setting not in result:
                continue
            upc = result[feature_setting]
            updates: dict = {}
            if attachment.selectable_models:
                valid, dropped = _partition_known(
                    attachment.selectable_models, known_ids
                )
                for m in dropped:
                    log.warning(
                        "Env attachment references unresolvable model; dropping",
                        gitlab_identifier=m,
                        feature_setting=feature_setting,
                        list="selectable_models",
                    )
                existing = set(upc.selectable_models)
                updates["selectable_models"] = [
                    *upc.selectable_models,
                    *[m for m in valid if m not in existing],
                ]
            if attachment.beta_models:
                valid, dropped = _partition_known(attachment.beta_models, known_ids)
                for m in dropped:
                    log.warning(
                        "Env attachment references unresolvable model; dropping",
                        gitlab_identifier=m,
                        feature_setting=feature_setting,
                        list="beta_models",
                    )
                existing_beta = set(upc.beta_models)
                updates["beta_models"] = [
                    *upc.beta_models,
                    *[m for m in valid if m not in existing_beta],
                ]
            if attachment.default_models:
                valid, dropped = _partition_known(attachment.default_models, known_ids)
                for m in dropped:
                    log.warning(
                        "Env attachment references unresolvable model; dropping",
                        gitlab_identifier=m,
                        feature_setting=feature_setting,
                        list="default_models",
                    )
                if valid:
                    updates["default_models"] = valid
            if updates:
                result[feature_setting] = upc.model_copy(update=updates)
        return result

    def _validate_model_ids_exist(
        self,
        unit_primitive_configs: Iterable[UnitPrimitiveConfig],
        models_ids: set,
    ) -> list[str]:
        errors: set[str] = set()
        for unit_primitive_config in unit_primitive_configs:
            ids = chain(
                unit_primitive_config.default_models,
                unit_primitive_config.models_for_tags.values(),
                unit_primitive_config.selectable_models,
                unit_primitive_config.beta_models,
                (dm.identifier for dm in unit_primitive_config.deprecated_models),
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

    def _validate_deprecated_models_are_selectable(
        self, unit_primitive_configs: Iterable[UnitPrimitiveConfig]
    ) -> list[str]:
        errors = [
            f"Feature '{upc.feature_setting}' has deprecated model "
            f"'{deprecated_model.identifier}' that is not in selectable_models."
            for upc in unit_primitive_configs
            for deprecated_model in upc.deprecated_models
            if deprecated_model.identifier not in upc.selectable_models
        ]
        if errors:
            return [
                "Feature-deprecated models must be included in selectable_models:\n"
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
            *self._validate_deprecated_models_are_selectable(unit_primitive_configs),
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
        if is_feature_enabled(FeatureFlag.AI_MODEL_RELEASE):
            if model := self._env_llm_definitions.get(model_id):
                return model
        if model := self.get_llm_definitions().get(model_id, None):
            return model
        raise ValueError(f"Invalid model identifier: {model_id}")

    def get_model_for_feature(self, feature_setting_name: str) -> LLMDefinition:
        if feature_setting := self.get_resolved_unit_primitive_config_map().get(
            feature_setting_name, None
        ):
            if is_feature_enabled(FeatureFlag.AI_GATEWAY_MULTI_DEFAULT_MODELS):
                return self.get_model(random.choice(feature_setting.default_models))
            return self.get_model(feature_setting.default_models[0])
        raise ValueError(f"Invalid feature setting: {feature_setting_name}")


def validate_model_selection_config():
    ModelSelectionConfig.instance().validate()
