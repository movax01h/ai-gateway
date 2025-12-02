from itertools import chain
from pathlib import Path
from typing import Any, Iterable, Literal, Optional

import yaml
from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, ConfigDict

from ai_gateway.model_selection.types import DeprecationInfo, DevConfig

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


class LLMDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    gitlab_identifier: str
    prompt_params: PromptParams = PromptParams()
    provider: Optional[str] = None
    description: str | None = None
    cost_indicator: Literal["$", "$$", "$$$"] | None = None
    params: dict[str, Any] = {}
    family: list[str] = []
    deprecation: Optional[DeprecationInfo] = None


class UnitPrimitiveConfig(BaseModel):
    feature_setting: str
    unit_primitives: list[GitLabUnitPrimitive]
    default_model: str
    selectable_models: list[str] = []
    beta_models: list[str] = []
    dev: DevConfig | None = None


class ModelSelectionConfig:
    _instance: Optional["ModelSelectionConfig"] = None

    def __init__(self) -> None:
        self._llm_definitions: Optional[dict[str, LLMDefinition]] = None
        self._unit_primitive_configs: Optional[dict[str, UnitPrimitiveConfig]] = None

    @classmethod
    def instance(cls) -> "ModelSelectionConfig":
        """Get the singleton instance of ModelSelectionConfig.

        Returns:
            The singleton ModelSelectionConfig instance.
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def get_llm_definitions(self) -> dict[str, LLMDefinition]:
        if not self._llm_definitions:

            with open(MODELS_CONFIG_PATH, "r") as f:
                config_data = yaml.safe_load(f)

            self._llm_definitions = {
                model_data["gitlab_identifier"]: LLMDefinition(**model_data)
                for model_data in config_data["models"]
            }

        return self._llm_definitions

    def get_unit_primitive_config_map(self) -> dict[str, UnitPrimitiveConfig]:
        if not self._unit_primitive_configs:
            with open(UNIT_PRIMITIVE_CONFIG_PATH, "r") as f:
                config_data = yaml.safe_load(f)

            self._unit_primitive_configs = {
                data["feature_setting"]: UnitPrimitiveConfig(**data)
                for data in config_data["configurable_unit_primitives"]
            }

        return self._unit_primitive_configs

    def get_unit_primitive_config(self) -> Iterable[UnitPrimitiveConfig]:
        return self.get_unit_primitive_config_map().values()

    def validate(self) -> None:
        unit_primitive_configs = self.get_unit_primitive_config()
        models = self.get_llm_definitions()
        models_ids = models.keys()

        errors: set[str] = set()
        default_model_not_selectable_errors: list[str] = []
        dev_models_validation_errors: list[str] = []

        for unit_primitive_config in unit_primitive_configs:
            ids = chain(
                [unit_primitive_config.default_model],
                unit_primitive_config.selectable_models,
                unit_primitive_config.beta_models,
                (
                    unit_primitive_config.dev.selectable_models
                    if unit_primitive_config.dev
                    else []
                ),
            )

            errors.update(model_id for model_id in ids if model_id not in models_ids)

            # Validate that the default model is also included in selectable_models
            if (
                unit_primitive_config.default_model
                not in unit_primitive_config.selectable_models
            ):
                default_model_not_selectable_errors.append(
                    f"Feature '{unit_primitive_config.feature_setting}' has default model "
                    f"'{unit_primitive_config.default_model}' that is not in selectable_models."
                )

            if unit_primitive_config.dev:
                dev_selectable = unit_primitive_config.dev.selectable_models
                dev_groups = unit_primitive_config.dev.group_ids

                # Validate that dev_selectable_models has at least a dev group ID specified
                if dev_selectable and not dev_groups:
                    dev_models_validation_errors.append(
                        f"Feature '{unit_primitive_config.feature_setting}' has dev selectable_models "
                        f"but group_ids is empty. Specify at least one group ID."
                    )

        error_messages = []
        if errors:
            error_messages.append(
                f"The following models ids are used but are not defined in models.yml: {', '.join(errors)}"
            )

        if default_model_not_selectable_errors:
            error_messages.append(
                "Default models must be included in selectable_models:\n"
                + "\n".join(
                    f"  - {error}" for error in default_model_not_selectable_errors
                )
            )

        if dev_models_validation_errors:
            error_messages.append(
                "Developer-only models contain certain errors:\n"
                + "\n".join(f"  - {error}" for error in dev_models_validation_errors)
            )

        if error_messages:
            raise ValueError("\n".join(error_messages))

    def refresh(self):
        """Refresh the configuration by reloading from source files."""
        self._llm_definitions = None
        self._unit_primitive_configs = None

    def get_model(self, model_id: str) -> LLMDefinition:
        if model := self.get_llm_definitions().get(model_id, None):
            return model
        raise ValueError(f"Invalid model identifier: {model_id}")

    def get_model_for_feature(self, feature_setting_name: str) -> LLMDefinition:
        if feature_setting := self.get_unit_primitive_config_map().get(
            feature_setting_name, None
        ):
            return self.get_model(feature_setting.default_model)
        raise ValueError(f"Invalid feature setting: {feature_setting_name}")


def validate_model_selection_config():
    ModelSelectionConfig.instance().validate()
