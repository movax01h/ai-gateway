from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, ConfigDict

from ai_gateway.model_selection import PromptParams
from ai_gateway.model_selection.models import BaseModelParams
from lib.billing_events.service import LLMOperationType

__all__ = ["PromptConfig", "ModelConfig"]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    params: BaseModelParams = BaseModelParams()


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: ModelConfig = ModelConfig()
    unit_primitive: GitLabUnitPrimitive
    prompt_template: dict[str, str]
    params: PromptParams | None = None
    operation_type: LLMOperationType = "standard"


class InMemoryPromptConfig(BaseModel):
    prompt_id: str

    model_config = ConfigDict(extra="forbid")

    name: str
    model: ModelConfig | None = None
    unit_primitives: list[GitLabUnitPrimitive]
    prompt_template: dict[str, str]
    params: PromptParams | None = None
    operation_type: LLMOperationType = "standard"

    def to_prompt_data(self) -> dict:
        params = self.model_dump(exclude={"prompt_id", "unit_primitives"})

        # Transform `unit_primitives` (kept for backwards compatibility) into a single value, with a default
        params["unit_primitive"] = next(
            iter(self.unit_primitives), GitLabUnitPrimitive.DUO_AGENT_PLATFORM
        )

        return params
