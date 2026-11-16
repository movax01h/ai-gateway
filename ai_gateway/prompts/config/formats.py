"""The two syntaxes a prompt may be written in, as one object each.

A prompt config declares `prompt_template` (the old roles mapping) or `template` (the format flow versions on the
chat engine use). Whichever it is, `parse_format` turns it into one object that answers the two questions every
consumer has: what does this prompt render (`resolve`), and which strings does it render (`sources`). Nothing
downstream branches on the syntax.
"""

from dataclasses import dataclass
from typing import Any, Optional, Protocol

from ai_gateway.prompts.config.template import Template
from lib.prompts.utilities import (
    PromptSource,
    ResolvedPromptTemplate,
    prompt_template_to_messages,
)

__all__ = [
    "LegacyTemplate",
    "PromptTemplateFormat",
    "parse_format",
]


class PromptTemplateFormat(Protocol):
    """What a prompt format provides, whichever syntax it was written in."""

    @property
    def renders_prompt(self) -> bool:
        """Whether the prompt renders anything of its own, which is what gets the default cache breakpoint."""

    def resolve(self, tool_output_security: bool) -> ResolvedPromptTemplate:
        """What the prompt renders on every model call."""

    def sources(self) -> tuple[PromptSource, ...]:
        """Every template string the prompt renders, without rendering anything."""


@dataclass(frozen=True)
class LegacyTemplate:
    """The old format: a mapping of roles to template strings, rendered whole on every model call."""

    roles: dict[str, str | list[str]]

    @property
    def renders_prompt(self) -> bool:
        # An empty `prompt_template: {}` renders nothing of its own, as on `main`.
        return bool(self.roles)

    def resolve(self, tool_output_security: bool) -> ResolvedPromptTemplate:
        return ResolvedPromptTemplate(
            prefix=prompt_template_to_messages(
                self.roles, tool_output_security=tool_output_security
            )
        )

    def sources(self) -> tuple[PromptSource, ...]:
        # A list expands into one source per entry; roles that aren't text, like `placeholder`, are the caller's to skip.
        return tuple(
            PromptSource(role, text)
            for role, value in self.roles.items()
            for text in ([value] if isinstance(value, str) else value)
            if text
        )


def parse_format(
    prompt_template: Optional[dict[str, str | list[str]]],
    template: Optional[Template | dict[str, Any]],
) -> PromptTemplateFormat:
    """The one format a prompt declares, from the two keys it may be written under.

    Args:
        prompt_template: The old format's roles mapping, if declared.
        template: The new format, as the validated model or as the mapping it was written as, if declared.

    Returns:
        The declared format.

    Raises:
        ValueError: When the prompt declares both formats or neither. `is None`, not emptiness: an empty
            `prompt_template` is a declared format, as it is on `main`.
    """
    if prompt_template is not None and template is not None:
        raise ValueError(
            "A prompt declares either 'template' or 'prompt_template', not both"
        )
    if template is not None:
        return (
            template
            if isinstance(template, Template)
            else Template.model_validate(template)
        )
    if prompt_template is None:
        raise ValueError("A prompt declares 'template' or 'prompt_template'")

    return LegacyTemplate(prompt_template)
