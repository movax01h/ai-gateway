"""The workspace agent kind: an agent template a customer authored in their workspace.

Items arrive in full with the request, so every field is here rather than an id to resolve. A kind sourced from the AI
Catalog would be its own module beside this one, referencing its item by id and version.

This model validates what an item is. Whether it fits the flow it attaches to is
:mod:`~duo_workflow_service.agent_platform.v1.catalog.binding`'s job.
"""

from typing import Annotated, Any, Mapping

from pydantic import AfterValidator, BaseModel, Field, StringConstraints
from pydantic_core import ErrorDetails

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

# Workhorse forwards a rejection in a WebSocket close frame, whose reason is cut at 123
# bytes. The longest message leaves this much room for the name it quotes.
_MAX_DISPLAY_NAME_BYTES = 40

# Pydantic error types, by what the user has to fix. Proto3 drops empty strings, so a
# blank field arrives as a missing one.
_PROBLEMS = {
    "missing": "missing",
    "string_too_short": "missing",
    "string_too_long": "too_long",
}

# What the user reads for each field and problem, worded as in the agent file, where
# the request's `toolset` is `tools`. A pair not listed reads "<agent> is not valid."
_MESSAGES = {
    ("name", "missing"): "the name is missing",
    ("name", "too_long"): "the name has {length} characters, the limit is {limit}",
    ("name", "invalid"): "the name is not valid",
    ("description", "missing"): "the description is missing",
    ("description", "too_long"): (
        "the description has {length} characters, the limit is {limit}"
    ),
    ("description", "invalid"): "the description is not valid",
    ("prompt", "missing"): "the prompt is missing",
    ("prompt", "too_long"): "the prompt has {length} characters, the limit is {limit}",
    ("prompt", "invalid"): "the prompt is not valid",
    ("tools", "invalid"): "the tools are not valid",
    ("tool", "missing"): "tool {tool} has no name",
    ("tool", "too_long"): "tool {tool} has {length} characters, the limit is {limit}",
    ("tool", "invalid"): "tool {tool} is not valid",
}


def _display_name(name: str) -> str:
    """The name as the user declared it, short enough to quote in an error.

    A long name loses its middle, keeping the start and the end that tell similar names apart.
    """
    name = name.strip().removeprefix(_NAME_PREFIX)
    # Checked on characters first, so an over-limit name is never encoded whole.
    if (
        len(name) <= _MAX_DISPLAY_NAME_BYTES
        and len(name.encode()) <= _MAX_DISPLAY_NAME_BYTES
    ):
        return name
    keep = (_MAX_DISPLAY_NAME_BYTES - len("…".encode())) // 2
    # Cut on bytes; a character split at either cut is dropped. Every character is at
    # least one byte, so encoding `keep` characters from each end covers the cut.
    head = name[:keep].encode()[:keep].decode(errors="ignore")
    tail = name[-keep:].encode()[-keep:].decode(errors="ignore")
    return f"{head}…{tail}"


def _subject(agent: Any, position: int) -> str:
    """Name an agent by its declared name, or by its position when it has no name to quote.

    A name over the limit is still quoted, shortened, so the user can tell which agent to fix.
    """
    name = agent.get("name") if isinstance(agent, Mapping) else None
    if isinstance(name, str) and name.strip():
        return f"Workspace agent '{_display_name(name)}'"
    return f"Workspace agent {position}"


def _field_and_tool(field_loc: tuple[int | str, ...]) -> tuple[str, int | None]:
    """The field an error is about, as the message table names it, and the tool's number for an error on one tool.

    Pydantic places a list entry's 0-based index after the field.

    Examples:
        ``("prompt",)`` -> ``("prompt", None)``
        ``("toolset",)`` -> ``("tools", None)``
        ``("toolset", 1)`` -> ``("tool", 2)``: the second tool.
    """
    if field_loc[0] != "toolset":
        return str(field_loc[0]), None
    if len(field_loc) == 1 or not isinstance(field_loc[1], int):
        return "tools", None
    return "tool", field_loc[1] + 1


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

    @property
    def display_name(self) -> str:
        """The name as the user declared it, short enough to quote in an error."""
        return _display_name(self.name)

    @classmethod
    def describe_error(
        cls,
        err: ErrorDetails,
        field_loc: tuple[int | str, ...],
        agent: Any,
        position: int,
    ) -> str:
        """Rewrite one pydantic error about an agent in the user's terms.

        This runs while a rejection is handled, so it reads the error defensively: a crash here would replace the
        user's message.

        Args:
            err: The error.
            field_loc: Its location inside the agent, such as ``("toolset", 1)``; empty for the agent itself.
            agent: The agent as sent, for its name.
            position: The agent's 1-based position in the request.

        Returns:
            The agent, by name or by position when it has no name to quote, and its problem.
        """
        subject = _subject(agent, position)

        if not field_loc:
            return f"{subject} is not valid."

        field, tool = _field_and_tool(field_loc)

        value, ctx = err["input"], err.get("ctx", {})
        problem = _PROBLEMS.get(err["type"], "invalid")
        if problem == "too_long" and not (
            isinstance(value, str) and "max_length" in ctx
        ):
            problem = "invalid"

        message = _MESSAGES.get((field, problem))
        if message is None:
            return f"{subject} is not valid."

        # Counted as the constraint counts it, after stripping.
        length = len(value.strip()) if isinstance(value, str) else None
        details = message.format(tool=tool, length=length, limit=ctx.get("max_length"))
        return f"{subject}: {details}."
