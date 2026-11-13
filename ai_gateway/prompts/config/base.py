from typing import Optional

from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, ConfigDict, model_validator

from ai_gateway.model_selection import ModelSelectionConfig, PromptParams
from ai_gateway.model_selection.models import (
    BaseModelParams,
    LiteLLMProviderParams,
    OpenAIProviderParams,
)
from lib.billing_events.service import LLMOperationType
from lib.prompts.utilities import (
    PromptSource,
    ResolvedPromptTemplate,
    prompt_sources,
    resolve_prompt_template,
)

__all__ = [
    "InMemoryPromptConfig",
    "ModelConfig",
    "PromptConfig",
    "PromptFormat",
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
    litellm: LiteLLMProviderParams | None = None

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


class PromptFormat(BaseModel):
    """What a prompt declares to render, shared by the registry's and the inline configs.

    `Prompt` builds a model call from `resolve()`, and the registry collects an author's variables from `sources()`.
    Other consumers still read `prompt_template` directly.
    """

    name: str
    prompt_template: dict[str, str | list[str]]
    tool_output_security: bool = True

    def sources(self) -> tuple[PromptSource, ...]:
        """Every template string this prompt renders, with the role it was written in, without rendering anything."""
        return prompt_sources(self.prompt_template)

    def resolve(
        self, tool_output_security: Optional[bool] = None
    ) -> ResolvedPromptTemplate:
        """Resolve this prompt's declared format into what it renders.

        Args:
            tool_output_security: Overrides the prompt's own setting, for a caller that binds tools.

        Returns:
            The resolved prompt.
        """
        return resolve_prompt_template(
            self.prompt_template,
            tool_output_security=(
                self.tool_output_security
                if tool_output_security is None
                else tool_output_security
            ),
        )


class PromptConfig(PromptFormat):
    model_config = ConfigDict(extra="forbid")

    model: ModelConfig = ModelConfig()
    unit_primitive: GitLabUnitPrimitive
    # The feature setting in unit_primitives.yml that lists the models of this prompt version. Only for prompts that the
    # AI Gateway serves over HTTP. A prompt that only the Duo Workflow Service runs does not set it: the flow selects
    # the models of its prompts.
    models_from: str | None = None
    params: PromptParams | None = None
    operation_type: LLMOperationType = LLMOperationType.STANDARD

    @model_validator(mode="after")
    def validate_models_from(self) -> "PromptConfig":
        if self.models_from is None:
            return self

        unit_primitive_config = (
            ModelSelectionConfig.instance()
            .get_unit_primitive_config_map()
            .get(self.models_from)
        )
        if unit_primitive_config is None:
            raise ValueError(
                f"models_from '{self.models_from}' is not a feature setting in unit_primitives.yml"
            )
        if self.unit_primitive not in unit_primitive_config.unit_primitives:
            raise ValueError(
                f"The '{self.models_from}' entry in unit_primitives.yml does not list the unit primitive "
                f"'{self.unit_primitive}'"
            )

        return self


class InMemoryPromptConfig(PromptFormat):
    prompt_id: str

    model_config = ConfigDict(extra="forbid")

    model: ModelConfig | None = None
    unit_primitives: list[GitLabUnitPrimitive]
    params: PromptParams | None = None
    operation_type: LLMOperationType = LLMOperationType.STANDARD

    def to_prompt_data(self) -> dict:
        params = self.model_dump(exclude={"prompt_id", "unit_primitives"})

        # Transform `unit_primitives` (kept for backwards compatibility) into a single value, with a default
        params["unit_primitive"] = next(
            iter(self.unit_primitives), GitLabUnitPrimitive.DUO_AGENT_PLATFORM
        )

        return params
