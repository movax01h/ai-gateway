"""The prompt format a flow version declares with a top-level `template` key.

`system` renders on every model call and is never stored. `message` renders once for each batch of messages a
component claims together, into one user message stored in that component's history, so a template change never
rewrites what the model already saw.
"""

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from lib.prompts.utilities import (
    PromptSource,
    ResolvedPromptTemplate,
    prompt_template_to_messages,
)

__all__ = [
    "DEFAULT_MESSAGE_TEMPLATE",
    "INBOUND_MESSAGES_VARIABLE",
    "SystemPart",
    "SystemPrompt",
    "Template",
]


# The messages a `message` block renders. The engine passes them to that block only, so an author never provides
# them, and no other block can render them. `cancelled_turn` (ADR-004 §6) joins here, as a second entry in the
# `message` source's `provided_by_engine`, in the change that makes entry dispatch pass it; marking it earlier would
# turn today's build-time refusal into a missing-variable error at the first render.
INBOUND_MESSAGES_VARIABLE = "inbound_messages"

# What `message` renders when an author leaves it out: the messages as they were sent.
DEFAULT_MESSAGE_TEMPLATE = "{{ " + INBOUND_MESSAGES_VARIABLE + ".text }}"


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
    than one part. `message` renders once for each batch of messages a component claims together, into one user
    message stored in that component's history, so a template change never rewrites what the model already saw.
    `{{ inbound_messages.text }}` joins the batch's texts; `{% for message in inbound_messages %}` frames each message
    on its own. It's one string for now; a structured form can follow the way `system` has one, and a string stays
    valid then. The keys fix the roles, so a prompt can't put a user or assistant turn where the platform doesn't
    expect one.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    system: Union[_NonBlankText, SystemPrompt]
    # Left out, the platform renders the messages as they were sent.
    message: Optional[_NonBlankText] = None

    @field_validator("system")
    @classmethod
    def _system_has_text(
        cls, system: Union[str, SystemPrompt]
    ) -> Union[str, SystemPrompt]:
        if isinstance(system, str):
            _has_text(system)
        return system

    @field_validator("message")
    @classmethod
    def _message_has_text(cls, message: Optional[str]) -> Optional[str]:
        return None if message is None else _has_text(message)

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
        """What this prompt renders: the system prompt on every call, and one user message per claimed batch.

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
            ),
            # The one user message, rendered once per claimed batch and appended to the component's history, so it
            # carries no security block and no history placeholder.
            tail=("user", self.message_text),
        )

    @property
    def message_text(self) -> str:
        """The `message` block as it renders: the author's string, or the platform default when it's left out."""
        return self.message or DEFAULT_MESSAGE_TEMPLATE

    def sources(self) -> tuple[PromptSource, ...]:
        """Every template string this prompt renders, with its block, without rendering anything.

        The `message` block is always a source, the platform default included, so the variable check and the
        security scan read exactly what `resolve` renders. Only that block is handed the messages it renders, so only
        its source marks `inbound_messages` as provided by the engine. In `system` the variable stays required, and
        the variable check refuses it.
        """
        return tuple(PromptSource("system", text) for text in self.system_texts) + (
            PromptSource(
                "message",
                self.message_text,
                provided_by_engine=frozenset({INBOUND_MESSAGES_VARIABLE}),
            ),
        )
