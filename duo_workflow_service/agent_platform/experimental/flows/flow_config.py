import inspect
from pathlib import Path
from typing import ClassVar, Self

import yaml
from pydantic import BaseModel

from duo_workflow_service.agent_platform.experimental import components

__all__ = ["FlowConfig", "load_component_class"]


class FlowConfig(BaseModel):
    DIRECTORY_PATH: ClassVar[Path] = Path(__file__).resolve().parent / "configs"
    flow: dict
    components: list[dict]
    routers: list[dict]
    environment: str
    version: int

    @classmethod
    def from_yaml_config(cls, path: str) -> Self:
        try:
            yaml_path = cls.DIRECTORY_PATH / path

            with open(yaml_path, "r", encoding="utf-8") as file:
                yaml_content = yaml.safe_load(file)

            return cls(**yaml_content)
        except FileNotFoundError:
            raise FileNotFoundError(f"{path} file not found in {cls.DIRECTORY_PATH}")
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Error parsing YAML file: {e}") from e


def load_component_class(cls_name: str) -> type:
    # Check if the class exists in the module
    if not hasattr(components, cls_name):
        raise AttributeError(f"Component class '{cls_name}' not found")

    component_class = getattr(components, cls_name)

    if not inspect.isclass(component_class):
        raise TypeError(f"'{cls_name}' is not a class")

    return component_class
