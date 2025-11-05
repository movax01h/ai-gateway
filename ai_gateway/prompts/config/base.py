from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, ConfigDict

from ai_gateway.prompts.config.models import BaseModelParams, TypeModelParams

__all__ = ["PromptConfig", "ModelConfig", "BaseModelConfig"]


class BaseModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    params: BaseModelParams


class ModelConfig(BaseModelConfig):
    params: TypeModelParams


class PromptParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stop: list[str] | None = None
    # NOTE: In langchain, some providers accept the timeout when initializing the client. However, support
    # and naming is inconsistent between them. Therefore, we bind the timeout to the prompt instead.
    # See https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/1035#note_2020952732 # pylint: disable=line-too-long
    timeout: float | None = None
    vertex_location: str | None = None
    cache_control_injection_points: list[dict] | None = None


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: ModelConfig
    unit_primitives: list[GitLabUnitPrimitive] = []
    prompt_template: dict[str, str]
    params: PromptParams | None = None


class InMemoryPromptConfig(PromptConfig):
    prompt_id: str

    def to_prompt_config(self) -> PromptConfig:
        params = self.model_dump()
        params.pop("prompt_id")
        return PromptConfig(**params)
