from pathlib import Path
from typing import Any, Optional

import yaml
from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel

BASE_PATH = Path(__file__).parent
MODELS_CONFIG_PATH = BASE_PATH / "models.yml"
UNIT_PRIMITIVE_CONFIG_PATH = BASE_PATH / "unit_primitives.yml"


class LLMDefinition(BaseModel):
    name: str
    gitlab_identifier: str
    provider: str
    provider_identifier: str
    params: dict[str, Any] = {}
    family: Optional[str] = None


class UnitPrimitiveConfig(BaseModel):
    unit_primitives: list[GitLabUnitPrimitive]
    default_model: str
    selectable_models: list[str] = []
    beta_models: list[str] = []


class ModelSelectionConfig:
    __instance = None

    def __new__(cls):
        if cls.__instance is None:
            cls.__instance = super(ModelSelectionConfig, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        self._llm_definitions: Optional[dict[str, LLMDefinition]] = None
        self._unit_primitive_configs: Optional[list[UnitPrimitiveConfig]] = None

    def get_llm_definitions(self) -> dict[str, LLMDefinition]:
        if not self._llm_definitions:

            with open(MODELS_CONFIG_PATH, "r") as f:
                config_data = yaml.safe_load(f)

            self._llm_definitions = {
                model_data["gitlab_identifier"]: LLMDefinition(**model_data)
                for model_data in config_data["models"]
            }

        return self._llm_definitions

    def get_unit_primitive_config(self) -> list[UnitPrimitiveConfig]:
        if not self._unit_primitive_configs:
            with open(UNIT_PRIMITIVE_CONFIG_PATH, "r") as f:
                config_data = yaml.safe_load(f)

            self._unit_primitive_configs = [
                UnitPrimitiveConfig(**data)
                for data in config_data["configurable_unit_primitives"]
            ]

        return self._unit_primitive_configs

    def refresh(self):
        """Refresh the configuration by reloading from source files."""
        self._llm_definitions = None
        self._unit_primitive_configs = None

    def get_gitlab_model(self, gitlab_model_id: str) -> LLMDefinition:
        if gitlab_model := self.get_llm_definitions().get(gitlab_model_id, None):
            return gitlab_model
        raise ValueError(f"Invalid model identifier: {gitlab_model_id}")
