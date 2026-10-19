"""The workspace agent kind: an agent template a customer authored in their workspace.

Items arrive in full with the request, so every field is here rather than an id to resolve. A kind sourced from the AI
Catalog would be its own module beside this one, referencing its item by id and version.

This model validates what an item is. Whether it fits the flow it attaches to is
:mod:`~duo_workflow_service.agent_platform.v1.catalog.binding`'s job.
"""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, Field, StringConstraints

__all__ = [
    "MAX_DESCRIPTION_LENGTH",
    "MAX_NAME_LENGTH",
    "MAX_PROMPT_LENGTH",
    "WorkspaceAgent",
]

# Ceilings on client-supplied text. Every value reaches the model, so these bound both
# the prompt-injection surface and the context one request can spend. Applied to what
# the client sent, before namespacing, so a name may exceed MAX_NAME_LENGTH once
# prefixed.
MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1_024
MAX_PROMPT_LENGTH = 10_000


# Required and stripped: each value reaches the graph and the `delegate_task` enum
# verbatim. Tool names are identifiers, so they share the name ceiling.
_Name = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_NAME_LENGTH),
]
_Description = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=MAX_DESCRIPTION_LENGTH
    ),
]
_Prompt = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True, min_length=1, max_length=MAX_PROMPT_LENGTH
    ),
]

# Namespaces client names by source, so an item can never take a name a flow author
# used, and two sources can each supply an agent of the same name.
_NAME_PREFIX = "workspace/agents/"


def _namespaced(name: str) -> str:
    """Prefix a name, tolerating one that already carries the prefix."""
    return name if name.startswith(_NAME_PREFIX) else f"{_NAME_PREFIX}{name}"


_ItemName = Annotated[_Name, AfterValidator(_namespaced)]


class WorkspaceAgent(BaseModel):
    """An agent template a customer authored in their workspace, attached as a subagent.

    Attributes:
        name: Component name in the flow graph, namespaced with ``_NAME_PREFIX``. At most
            ``MAX_NAME_LENGTH`` characters as sent.
        description: What the coordinator sees in ``delegate_task``. At most ``MAX_DESCRIPTION_LENGTH``
            characters.
        toolset: Tools the agent may use. May be empty: an agent that only reasons is a valid delegation
            target.
        prompt: The agent's system prompt, passed to the flow's prompt as a value. Required, because that
            prompt is only a placeholder for this one. At most ``MAX_PROMPT_LENGTH`` characters.
    """

    name: _ItemName
    description: _Description
    toolset: list[_Name] = Field(default_factory=list)
    prompt: _Prompt
