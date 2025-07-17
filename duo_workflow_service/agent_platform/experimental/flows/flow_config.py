import inspect
from pathlib import Path
from typing import ClassVar, Self

import yaml
from pydantic import BaseModel

from duo_workflow_service.agent_platform.experimental import components

__all__ = ["FlowConfig", "load_component_class"]


_PREFIX_BLOCLIST = (
    "..",
    "/.../",
    r"\…..\\",
    "%00../../../../../",
    "%2e%2e%2f",
    "%252e%252e%252f",
    "%c0%ae%c0%ae%c0%af",
    "%uff0e%uff0e%u2215",
    "%uff0e%uff0e%u2216",
)


class FlowConfig(BaseModel):
    DIRECTORY_PATH: ClassVar[Path] = Path(__file__).resolve().parent / "configs"
    flow: dict
    components: list[dict]
    routers: list[dict]
    environment: str
    version: str

    @classmethod
    def from_yaml_config(cls, path: str) -> Self:
        try:
            # Validate path before resolving to prevent directory traversal
            if any(prefix in path for prefix in _PREFIX_BLOCLIST) or path.startswith(
                "/"
            ):
                raise ValueError(f"Path traversal detected: {path}")

            base_path = cls.DIRECTORY_PATH.resolve()
            yaml_path = (base_path / f"{path}.yml").resolve()

            if not yaml_path.is_relative_to(base_path):
                raise ValueError(f"Path traversal detected: {path}")

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
