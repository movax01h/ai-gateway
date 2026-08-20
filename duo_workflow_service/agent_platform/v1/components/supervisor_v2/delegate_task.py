from enum import StrEnum
from typing import ClassVar, Self, TypedDict

from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict, Field, create_model

__all__ = [
    "DelegateTask",
    "SubagentDescriptor",
    "build_delegate_task_model",
]


class SubagentDescriptor(TypedDict):
    """Name and description of a managed subagent, used to build the delegate_task tool."""

    name: str
    description: str


def build_delegate_task_model(
    subagents: list[SubagentDescriptor],
) -> type["DelegateTask"]:
    """Build a DelegateTask Pydantic model with a dynamically generated SubagentEnum.

    The SubagentEnum is generated from the subagents list, constraining
    the LLM at the tool-calling level to only valid subagent names.  Each enum
    member's description is embedded in the field description so the LLM knows
    what each subagent specialises in.

    Args:
        subagents: List of dicts with ``name`` and ``description``
            for each managed subagent.

    Returns:
        A DelegateTask Pydantic model class with the SubagentEnum type for
        subagent_name.
    """
    subagent_enum = StrEnum(  # type: ignore[misc]
        "SubagentEnum", {cfg["name"]: cfg["name"] for cfg in subagents}
    )

    enum_values = [member.value for member in subagent_enum]
    agent_descriptions = "\n".join(
        f"- {cfg['name']}: {cfg['description']}" for cfg in subagents
    )

    dynamic_model = create_model(
        "DynamicDelegateTask",
        __base__=DelegateTask,
        # pydantic's create_model does NOT inherit __doc__ from __base__ — pass
        # it explicitly, since this docstring is what becomes the tool's
        # description shown to the LLM (see DelegateTask's docstring).
        __doc__=DelegateTask.__doc__,
        subagent_name=(
            subagent_enum,
            Field(
                description=(
                    f"The specialist agent to delegate to. Available agents:\n{agent_descriptions}"
                ),
                json_schema_extra={"enum": enum_values},  # type: ignore[dict-item]
            ),
        ),
    )

    dynamic_model.model_rebuild(force=True)

    return dynamic_model


class DelegateTask(BaseModel):
    """Delegate a task to a specialist subagent.

    **Rules — violating these will produce an error:**

    1. Calling this tool multiple times in the same turn is expected and
        encouraged whenever you have several independent tasks to delegate:
        each call spawns its own subsession and they all run concurrently. Do
        NOT delegate one task, wait, then delegate the next — batch every
        independent delegation into the same turn instead.
    2. However, in a turn where you call this tool, it must be the ONLY tool
        you call — never mix delegate_task with any other tool (e.g.
        run_command, read_file) in the same message. If you also need to run
        other tools, do that in a separate turn, before or after delegating.
    3. A subagent keeps nothing between delegations. Every call starts one
        from scratch, so give it the full context it needs; you cannot send a
        follow-up to a subagent you delegated to earlier. If a result comes
        back incomplete, delegate again and restate everything the subagent
        needs in the new prompt.
    """

    model_config = ConfigDict(title="delegate_task", frozen=True)

    tool_title: ClassVar[str] = "delegate_task"

    subagent_name: str = Field(description="The specialist agent to delegate to.")
    description: str = Field(
        description=(
            "A short (3-5 words) description of the task, used to label it "
            "in the UI while the subagent is running (e.g. 'Investigate "
            "flaky test', 'Refactor auth module')."
        )
    )
    prompt: str = Field(
        description="Detailed instructions and context for the subagent."
    )

    @classmethod
    def from_ai_message(cls, ai_message: AIMessage) -> Self:
        """Extract a DelegateTask from an AI message's tool calls."""
        delegate_call = next(
            (tc for tc in ai_message.tool_calls if tc["name"] == cls.tool_title),
            None,
        )
        if delegate_call is None:
            raise ValueError(f"No {cls.tool_title} tool call found in AI message")
        return cls(**delegate_call["args"])
