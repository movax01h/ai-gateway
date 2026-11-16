"""The prompt format a flow version declares with a top-level `template` key.

`system` renders on every model call and is never stored. A block that renders once per message a component
receives joins it later.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lib.prompts.utilities import (
    PromptSource,
    ResolvedPromptTemplate,
    prompt_template_to_messages,
)

__all__ = [
    "SystemPart",
    "SystemPrompt",
    "Template",
]


def _has_text(text: str) -> str:
    # A blank part renders an empty message, which a model either rejects or answers with nothing.
    if not text.strip():
        raise ValueError("A template part needs text")
    return text


# The published schema refuses blank text too, so the AI Catalog and an editor catch it before the flow loads.
_NonBlankText = Annotated[str, Field(json_schema_extra={"pattern": r"\S"})]


class SystemPart(BaseModel):
    """One part of the system prompt, in the advanced form."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    # Options on one part, such as where caching ends, join as new keys: unknown keys are refused today, so adding
    # one later breaks no config.
    text: _NonBlankText

    @field_validator("text")
    @classmethod
    def _text_has_content(cls, text: str) -> str:
        return _has_text(text)


class SystemPrompt(BaseModel):
    """The system prompt in the advanced form: one system message, written as parts in order.

    `type` names what kind of system content this is, and it's required, so a later kind joins with a name of its
    own and nothing has to be inferred. The parts are content blocks of the one system message. Claude receives them
    as separate blocks, with the cache breakpoint after the first, so the first part holds what stays the same across
    sessions and is read from the prompt cache. Other models get the parts joined into one system prompt, as today.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["system_message"]
    parts: list[SystemPart] = Field(min_length=1)


class Template(BaseModel):
    """A prompt in the format that flow versions on the chat engine declare.

    `system` renders on every model call and is never stored. It's a string, or a `SystemPrompt` when it has more
    than one part. The keys fix the roles, so a prompt can't put a user or assistant turn where the platform doesn't
    expect one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    system: Union[_NonBlankText, SystemPrompt]

    @field_validator("system")
    @classmethod
    def _system_has_text(
        cls, system: Union[str, SystemPrompt]
    ) -> Union[str, SystemPrompt]:
        if isinstance(system, str):
            _has_text(system)
        return system

    @property
    def renders_prompt(self) -> bool:
        # `system` is required and never blank, so a template always renders a prompt of its own.
        return True

    @property
    def system_texts(self) -> tuple[str, ...]:
        """The system prompt's parts, in order; a string is one part."""
        if isinstance(self.system, str):
            return (self.system,)
        return tuple(part.text for part in self.system.parts)

    def resolve(self, tool_output_security: bool) -> ResolvedPromptTemplate:
        """What this prompt renders on every model call: the system prompt, with its parts in order.

        The parts are built exactly as a `prompt_template.system` list is, one LangChain system message each, followed
        by the security block and the history placeholder, so each model gets them as it does today: Claude as
        separate blocks with the cache breakpoint after the first, other models joined into one system prompt.
        Rendering them through the same function makes that true by construction, which keeps a prompt's cached
        prefix unchanged when it moves to this format. An option on a part, once there is one, is applied here.
        """
        return ResolvedPromptTemplate(
            prefix=prompt_template_to_messages(
                {"system": list(self.system_texts)},
                tool_output_security=tool_output_security,
            )
        )

    def sources(self) -> tuple[PromptSource, ...]:
        """Every template string this prompt renders, with its block, without rendering anything."""
        return tuple(PromptSource("system", text) for text in self.system_texts)
