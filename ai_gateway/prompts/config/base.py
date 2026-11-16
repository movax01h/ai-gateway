from typing import Self

from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, ConfigDict, model_validator

from ai_gateway.model_selection import ModelSelectionConfig, PromptParams
from ai_gateway.model_selection.models import (
    BaseModelParams,
    LiteLLMProviderParams,
    OpenAIProviderParams,
)
from ai_gateway.prompts.config.formats import (
    LegacyTemplate,
    PromptTemplateFormat,
    parse_format,
)
from ai_gateway.prompts.config.template import Template
from lib.billing_events.service import LLMOperationType

__all__ = [
    "InMemoryPromptConfig",
    "ModelConfig",
    "PromptConfig",
    "PromptProviderParams",
    "TemplateDeclaration",
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


class TemplateDeclaration(BaseModel):
    """The fields under which a prompt declares its template, shared by the registry's and the inline configs.

    A prompt is written as `prompt_template` or `template`, never both and never neither. The keys stay as the
    published schema has them; `prompt_format` is the one object every consumer reads instead of either key, so
    nothing downstream branches on the syntax. Consumers that only understand the old format read
    `legacy_prompt_template`, which refuses a `template` prompt.
    """

    # The one-format rule, for the published schema that authors validate against in the AI Catalog and their editor.
    # A branch matches when its key holds an object, so an explicit `null` counts as not declared, as in Pydantic.
    model_config = ConfigDict(
        json_schema_extra={
            "oneOf": [
                {"required": [key], "properties": {key: {"type": "object"}}}
                for key in ("prompt_template", "template")
            ]
        }
    )

    name: str
    prompt_template: dict[str, str | list[str]] | None = None
    template: Template | None = None
    tool_output_security: bool = True

    @model_validator(mode="after")
    def validate_one_format(self) -> Self:
        parse_format(self.prompt_template, self.template)
        return self

    @property
    def prompt_format(self) -> PromptTemplateFormat:
        """The one format this prompt declares, resolved from whichever key it was written under.

        Read from the fields on every access rather than stored when the config validates, so a copy made with
        `model_copy(update=...)`, which skips validation, can't carry the format of the original.
        """
        return parse_format(self.prompt_template, self.template)

    @property
    def legacy_prompt_template(self) -> dict[str, str | list[str]]:
        """The old format's roles, for consumers that read only it.

        Returns:
            The declared `prompt_template`.

        Raises:
            ValueError: When the prompt declares `template`, which only agent flows render. An empty prompt would
                otherwise reach the model with no error.
        """
        prompt_format = self.prompt_format
        if not isinstance(prompt_format, LegacyTemplate):
            raise ValueError(
                f"Prompt '{self.name}' declares 'template', which only agent flows render. "
                "This consumer reads 'prompt_template'."
            )

        return prompt_format.roles


class PromptConfig(TemplateDeclaration):
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


class InMemoryPromptConfig(TemplateDeclaration):
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
