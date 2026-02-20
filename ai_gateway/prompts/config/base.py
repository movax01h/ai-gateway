from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, ConfigDict

from ai_gateway.model_selection import PromptParams
from ai_gateway.model_selection.models import BaseModelParams

__all__ = ["PromptConfig", "ModelConfig"]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    params: BaseModelParams = BaseModelParams()


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: ModelConfig = ModelConfig()
    unit_primitives: list[GitLabUnitPrimitive]
    prompt_template: dict[str, str]
    params: PromptParams | None = None


class InMemoryPromptConfig(BaseModel):
    prompt_id: str

    model_config = ConfigDict(extra="forbid")

    name: str
    model: ModelConfig | None = None
    unit_primitives: list[GitLabUnitPrimitive]
    prompt_template: dict[str, str]
    params: PromptParams | None = None

    def to_prompt_data(self) -> dict:
        params = self.model_dump()
        params.pop("prompt_id")
        return params
