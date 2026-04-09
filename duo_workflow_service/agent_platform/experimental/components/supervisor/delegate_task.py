from enum import StrEnum
from typing import ClassVar, Optional, Self, TypedDict

from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict, Field, create_model

__all__ = ["DelegateTask", "ManagedAgentConfig", "build_delegate_task_model"]


class ManagedAgentConfig(TypedDict):
    """Name and description of a managed subagent, used to build the delegate_task tool."""

    name: str
    description: str


def build_delegate_task_model(
    managed_agents_config: list[ManagedAgentConfig],
) -> type["DelegateTask"]:
    """Build a DelegateTask Pydantic model with a dynamically generated SubagentEnum.

    The SubagentEnum is generated from the managed_agents_config list, constraining
    the LLM at the tool-calling level to only valid subagent names.  Each enum
    member's description is embedded in the field description so the LLM knows
    what each subagent specialises in.

    Args:
        managed_agents_config: List of dicts with ``name`` and ``description``
            for each managed subagent.

    Returns:
        A DelegateTask Pydantic model class with the SubagentEnum type for
        subagent_name.
    """
    subagent_enum = StrEnum(  # type: ignore[misc]
        "SubagentEnum", {cfg["name"]: cfg["name"] for cfg in managed_agents_config}
    )

    enum_values = [member.value for member in subagent_enum]
    agent_descriptions = "\n".join(
        f"- {cfg['name']}: {cfg['description']}" for cfg in managed_agents_config
    )

    dynamic_model = create_model(
        "DynamicDelegateTask",
        __base__=DelegateTask,
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
    """Base class for delegate_task tool.

    Delegates a task to a specialist subagent.

    **Important constraints — failure to follow these will result in an error:**

    - You may only call this tool **once per turn**. Parallel delegation is not
        supported. To involve multiple subagents, delegate to them sequentially
        across separate turns.
    - This must be the **only** tool call in the turn. Do not mix delegate_task
        with any other tool calls in the same message.

    The actual model used at runtime is built dynamically by build_delegate_task_model() with a SubagentEnum generated
    from the managed_agents list.
    """

    model_config = ConfigDict(title="delegate_task", frozen=True)

    tool_title: ClassVar[str] = "delegate_task"

    subagent_name: str = Field(description="The specialist agent to delegate to.")
    subsession_id: Optional[int] = Field(
        default=None,
        description=(
            "Set to the ID of an existing subsession to resume it with "
            "additional instructions. Leave empty to spawn a new subsession. "
            "Subsession IDs are returned in the tool response when a subsession completes."
        ),
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
