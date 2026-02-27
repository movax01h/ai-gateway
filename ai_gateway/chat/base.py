from abc import ABC, abstractmethod
from typing import Optional

from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, ConfigDict

from ai_gateway.chat.tools import BaseTool
from lib.context import StarletteUser

__all__ = [
    "UnitPrimitiveToolset",
    "BaseToolsRegistry",
]


class UnitPrimitiveToolset(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: GitLabUnitPrimitive
    tools: list[BaseTool]

    # Minimum required GitLab version to use the tools.
    # If it's not set, the tools are available for all GitLab versions.
    min_required_gl_version: Optional[str] = None


class BaseToolsRegistry(ABC):
    @abstractmethod
    def get_on_behalf(self, user: StarletteUser, gl_version: str) -> list[BaseTool]:
        pass

    @abstractmethod
    def get_all(self) -> list[BaseTool]:
        pass
