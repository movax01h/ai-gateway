import functools
import random
import re
from itertools import chain
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal, Optional

import structlog
import yaml
from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    TypeAdapter,
    ValidationError,
    field_validator,
    model_validator,
)

from ai_gateway.config import (
    ModelReleaseFeatureAttachment,
    ModelReleasesPayload,
    get_config,
)
from ai_gateway.model_selection.keyword_bm25 import KeywordBM25
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
    DefaultModelEntry,
    DeprecationInfo,
    DevConfig,
    FeatureDeprecatedModel,
)
from lib.context.model import restricted_access_ctx
from lib.feature_flags import FeatureFlag, is_feature_enabled

log = structlog.stdlib.get_logger(__name__)


class RestrictedModelAccessError(Exception):
    """A restricted model was requested outside the flows it is restricted to.

    Deliberately not a ``ValueError``: callers that catch ``ValueError`` fall back to
    another model silently, and a denial must never be swallowed that way.
    """

    def __init__(self) -> None:
        super().__init__("Model not available for this flow")


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
    # Bedrock expects the inference profile / model ARN to be passed at
    # invocation time via model_id, not at client initialization.
    # See https://docs.litellm.ai/docs/providers/bedrock#set-via-model_id
    model_id: str | None = None

    # Anthropic-only params, stripped for other providers in Prompt.__init__
    # (see ANTHROPIC_ONLY_MODEL_KWARGS in ai_gateway/prompts/base.py).
    context_management: dict | None = None
    thinking: dict[str, Any] | None = None
    output_config: dict[str, Any] | None = None


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
    # Whether this deployment accepts image input; absent means ask litellm's registry.
    # Set per deployment, with a dated comment naming the evidence.
    supports_vision: Optional[bool] = None
    requires_single_system_message: bool = False
    # Some models are available only to accounts that have purchased GitLab credits.
    # When true, clients show a "Requires paid credits" indicator and block chat input.
    requires_paid_credits: bool = False
    # Some models return an empty 404 on streaming requests when max_tokens is at the
    # model's max. Opt in to use this model's max_tokens from models.yml
    use_model_max_tokens: bool = False
    # Flow config ids (e.g. "bl_security") allowed to use this model. When non-empty,
    # the model is usable only by a request authorized for one of these flows
    # (restricted_access_ctx) and is denied everywhere else; see
    # ensure_restricted_model_access.
    restricted_to_flows: list[str] = []


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


class ModelTagEntry(BaseModel):
    """Which models serve a tag, and the goal keywords that select it.

    `small: <id>` and `small: {models: [<id>], keywords: [...]}` build the same object.
    """

    model_config = ConfigDict(extra="forbid")

    models: list[str] = Field(min_length=1)
    keywords: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_model_id(cls, data: Any) -> Any:
        """Wrap the `tag: model_id` spelling into `{"models": [model_id]}`."""
        if isinstance(data, str):
            return {"models": [data]}
        return data


class UnitPrimitiveConfig(BaseModel):
    feature_setting: str
    unit_primitives: list[GitLabUnitPrimitive]
    default_models: list[DefaultModelEntry] = Field(min_length=1)
    models_for_tags: dict[str, ModelTagEntry] = Field(default_factory=dict)
    selectable_models: list[str] = Field(default_factory=list)
    beta_models: list[str] = Field(default_factory=list)
    deprecated_models: list[FeatureDeprecatedModel] = Field(default_factory=list)
    dev: DevConfig | None = None

    @field_validator("default_models", mode="before")
    @classmethod
    def coerce_default_models(cls, v: Any) -> Any:
        """Coerce plain strings in default_models to DefaultModelEntry dicts."""
        if not isinstance(v, list):
            return v

        return [
            {"identifier": entry} if isinstance(entry, str) else entry for entry in v
        ]

    @model_validator(mode="after")
    def validate_weights_all_or_none(self) -> "UnitPrimitiveConfig":
        """Weights must be specified for all entries or for no entries."""
        weights = [entry.weight for entry in self.default_models]
        has_weight = [w is not None for w in weights]
        if any(has_weight) and not all(has_weight):
            missing = [
                self.default_models[i].identifier
                for i, present in enumerate(has_weight)
                if not present
            ]
            raise ValueError(
                f"Feature '{self.feature_setting}': weight must be specified for all "
                f"default_models entries or none. Missing weight for: {missing}"
            )
        if all(has_weight) and sum(w for w in weights if w is not None) <= 0:
            raise ValueError(
                f"Feature '{self.feature_setting}': at least one default_models weight must be above 0"
            )
        return self

    @property
    def default_model_identifiers(self) -> list[str]:
        """Return the list of model identifiers from default_models."""
        return [entry.identifier for entry in self.default_models]

    @property
    def default_model_weights(self) -> list[float] | None:
        """Return weights if all entries have weights, otherwise None."""
        weights = [entry.weight for entry in self.default_models]
        if all(w is not None for w in weights):
            return weights  # type: ignore[return-value]
        return None


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
                    self._unit_primitive_configs[feature_setting].default_models = [
                        DefaultModelEntry(identifier=m) for m in models
                    ]

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
                    updates["default_models"] = [
                        DefaultModelEntry(identifier=m) for m in valid
                    ]
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
                unit_primitive_config.default_model_identifiers,
                chain.from_iterable(
                    entry.models
                    for entry in unit_primitive_config.models_for_tags.values()
                ),
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
            for default_model in upc.default_model_identifiers
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

    def _validate_restricted_models(
        self,
        unit_primitive_configs: Iterable[UnitPrimitiveConfig],
        models: dict[str, LLMDefinition],
    ) -> list[str]:
        errors = [
            f"Model '{model_id}': restricted_to_flows has empty or duplicate entries"
            for model_id, llm_def in models.items()
            if any(not flow.strip() for flow in llm_def.restricted_to_flows)
            or len(set(llm_def.restricted_to_flows)) != len(llm_def.restricted_to_flows)
        ]
        for config in unit_primitive_configs:
            # A feature's feature_setting names its flow by convention, so a restricted
            # model may be listed by the features of its own flows only.
            restricted = {
                i
                for i, d in models.items()
                if d.restricted_to_flows
                and config.feature_setting not in d.restricted_to_flows
            }
            for list_name, ids in {
                "default_models": config.default_model_identifiers,
                "selectable_models": config.selectable_models,
                "beta_models": config.beta_models,
                "dev.selectable_models": config.dev.selectable_models
                if config.dev
                else [],
                "models_for_tags": [
                    m for e in config.models_for_tags.values() for m in e.models
                ],
            }.items():
                errors.extend(
                    f"Restricted model '{model_id}' cannot be offered by "
                    f"'{config.feature_setting}': remove it from {list_name}"
                    for model_id in sorted(restricted.intersection(ids))
                )
        return errors

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
            *self._validate_restricted_models(unit_primitive_configs, models),
        ]

        if error_messages:
            raise ValueError("\n".join(error_messages))

    def refresh(self):
        """Refresh the configuration by reloading from source files."""
        self._llm_definitions = None
        self._unit_primitive_configs = None

    def get_proxy_models_for_provider(self, provider: str) -> list[str]:
        """Get list of allowed model names for a provider's proxy endpoint.

        Restricted models (``restricted_to_flows``) are never exposed on the proxy.

        Args:
            provider: The provider name (e.g., "anthropic", "openai")

        Returns:
            List of model names allowed for proxy
        """
        llm_definitions = self.get_llm_definitions()
        return [
            llm_def.params.model or ""
            for llm_def in llm_definitions.values()
            if llm_def.proxy_provider == provider
            and llm_def.params.model
            and self.restricted_flows_for(models=[llm_def.params.model]) is None
        ]

    def _restricted_definitions(self) -> list[LLMDefinition]:
        # Env-injected releases are included whether or not their feature flag is on:
        # a restriction must hold even where the definition itself is not served.
        return [
            llm_def
            for llm_def in chain(
                self.get_llm_definitions().values(),
                self._env_llm_definitions.values(),
            )
            if llm_def.restricted_to_flows
        ]

    def restricted_flows_for(
        self,
        identifiers: Iterable[Optional[str]] = (),
        models: Iterable[Optional[str]] = (),
    ) -> Optional[frozenset[str]]:
        """Return the flows a model may be used by, or None when it is not restricted.

        A model is restricted when one of ``identifiers`` is the ``gitlab_identifier`` of a
        restricted definition, or one of ``models`` (provider model strings) contains a
        restricted definition's ``params.model``:

        - preceded by the start of the string, ``/``, ``.``, ``:`` or ``@`` (so provider and
          region prefixes such as ``anthropic/``, ``vertex_ai/`` or ``us.anthropic.`` are
          caught), and
        - followed by the end of the string, ``@`` (``<model>@<date>``), ``:``
          (``<model>:0``), ``-v<digit>`` (``<model>-v1:0``) or an 8-digit date
          (``<model>-20261001``).

        Any other suffix is not matched: sibling models sharing the prefix (``<model>-mini``,
        ``<model>.1``), but also aliases such as ``<model>-2026-10-01``, ``<model>-latest``
        or ``<model>-preview``. A restricted definition's ``params.model`` must therefore be
        the base upstream model name, with no date or preview suffix.

        When several restricted definitions match, only flows allowed by all of them are
        returned.
        """
        ids = {i for i in identifiers if i}
        model_strings = [m.lower() for m in models if m]
        allowed: Optional[frozenset[str]] = None
        for llm_def in self._restricted_definitions():
            restricted_model = (llm_def.params.model or "").lower()
            matched = llm_def.gitlab_identifier in ids or (
                bool(restricted_model)
                and any(
                    re.search(
                        rf"(?:^|[/.:@]){re.escape(restricted_model)}"
                        r"(?=$|[@:]|-v\d|-\d{8})",
                        m,
                    )
                    for m in model_strings
                )
            )
            if matched:
                flows = frozenset(llm_def.restricted_to_flows)
                allowed = flows if allowed is None else allowed & flows
        return allowed

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
            identifiers = feature_setting.default_model_identifiers
            weights = feature_setting.default_model_weights
            chosen = random.choices(identifiers, weights=weights, k=1)[0]
            return self.get_model(chosen)
        raise ValueError(f"Invalid feature setting: {feature_setting_name}")

    def resolve_tag_for_goal(
        self, feature_setting_name: str, goal: str
    ) -> Optional[tuple[str, list[str]]]:
        """Return the first tag the goal matches and every one of its keywords found in the goal, or None.

        Tags are checked in declaration order. Keywords are scored with BM25 (see `keyword_bm25.py`).

        Design: https://gitlab.com/gitlab-org/gitlab/-/work_items/627658
        """
        unit_primitive_config = self.get_resolved_unit_primitive_config_map().get(
            feature_setting_name
        )
        if unit_primitive_config is None or not goal:
            return None

        tagged = [
            (tag, keyword)
            for tag, entry in unit_primitive_config.models_for_tags.items()
            for keyword in entry.keywords
        ]
        if not tagged:
            return None
        matches = _keyword_scorer(tuple(keyword for _, keyword in tagged)).matches(goal)
        hits = [pair for pair, hit in zip(tagged, matches) if hit]
        if not hits:
            return None
        tag = hits[0][0]
        return tag, [keyword for hit_tag, keyword in hits if hit_tag == tag]


@functools.lru_cache(maxsize=64)
def _keyword_scorer(keywords: tuple[str, ...]) -> KeywordBM25:
    # Keywords come from static YAML, so build the BM25 statistics once per keyword list.
    return KeywordBM25(list(keywords))


def ensure_restricted_model_access(
    identifiers: Iterable[Optional[str]] = (),
    models: Iterable[Optional[str]] = (),
) -> None:
    """Raise RestrictedModelAccessError unless the request may use the given model.

    Default-deny: a restricted model is usable only when ``restricted_access_ctx``
    names one of the flows it is restricted to.
    """
    identifiers = [i for i in identifiers if i]
    allowed = ModelSelectionConfig.instance().restricted_flows_for(identifiers, models)
    if allowed is None:
        return
    authorized_flow = restricted_access_ctx.get()
    if authorized_flow is None or authorized_flow not in allowed:
        log.warning(
            "Restricted model denied",
            identifiers=identifiers,
            authorized_flow=authorized_flow,
            allowed_flows=sorted(allowed),
        )
        raise RestrictedModelAccessError()


def validate_model_selection_config():
    ModelSelectionConfig.instance().validate()
