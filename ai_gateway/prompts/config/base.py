from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, ConfigDict, model_validator

from ai_gateway.model_selection import PromptParams
from ai_gateway.model_selection.models import BaseModelParams, OpenAIProviderParams
from lib.billing_events.service import LLMOperationType

__all__ = [
    "InMemoryPromptConfig",
    "ModelConfig",
    "PromptConfig",
    "PromptProviderParams",
]


# Field names must exactly match ModelClassProvider values (enforced by test)
class PromptProviderParams(BaseModel):
    """Provider-conditional model params in a prompt definition.

    Placement rule: set model params in models.yml unless the value must
    differ per feature; only then use this block.

    A block applies only when the prompt resolves to that provider; otherwise
    it is skipped, never an error (models served through LiteLLM never match
    `openai`). Each set field replaces its models.yml counterpart whole —
    unset fields are left alone, nested objects are not deep-merged, and a
    value cannot be cleared back to unset.
    """

    model_config = ConfigDict(extra="forbid")

    openai: OpenAIProviderParams | None = None

    @model_validator(mode="after")
    def validate_blocks_not_empty(self) -> "PromptProviderParams":
        blocks = self.model_dump(exclude_none=True)

        if not blocks:
            raise ValueError("provider_params must set at least one provider block")

        empty = [name for name, block in blocks.items() if not block]
        if empty:
            raise ValueError(
                f"provider_params blocks must not be empty: {', '.join(empty)}"
            )

        return self


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    params: BaseModelParams = BaseModelParams()
    provider_params: PromptProviderParams | None = None


class PromptConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    model: ModelConfig = ModelConfig()
    unit_primitive: GitLabUnitPrimitive
    prompt_template: dict[str, str | list[str]]
    params: PromptParams | None = None
    operation_type: LLMOperationType = "standard"


class InMemoryPromptConfig(BaseModel):
    prompt_id: str

    model_config = ConfigDict(extra="forbid")

    name: str
    model: ModelConfig | None = None
    unit_primitives: list[GitLabUnitPrimitive]
    prompt_template: dict[str, str | list[str]]
    params: PromptParams | None = None
    operation_type: LLMOperationType = "standard"

    def to_prompt_data(self) -> dict:
        params = self.model_dump(exclude={"prompt_id", "unit_primitives"})

        # Transform `unit_primitives` (kept for backwards compatibility) into a single value, with a default
        params["unit_primitive"] = next(
            iter(self.unit_primitives), GitLabUnitPrimitive.DUO_AGENT_PLATFORM
        )

        return params
