"""Shared machinery for sources that attach their items as subagents of the claiming component."""

from typing import Any, Literal, Optional, Protocol, Self, Union

from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, ConfigDict

from ai_gateway.prompts.config.base import InMemoryPromptConfig, PromptParams
from duo_workflow_service.agent_platform.v1.components.agent.component import (
    AgentComponent,
)
from duo_workflow_service.agent_platform.v1.components.supervisor.ui_log import (
    UILogEventsSupervisor,
)

__all__ = [
    "ITEM_PROMPT_VARIABLE",
    "AgentComponentTemplate",
    "SubagentItem",
    "item_prompt_config",
    "literal_input",
    "promote_claimant",
    "synthesized_agent_config",
]

ITEM_PROMPT_VARIABLE = "item_prompt"

_DELEGATION_UI_LOG_EVENTS = (
    UILogEventsSupervisor.ON_DELEGATION,
    UILogEventsSupervisor.ON_DELEGATION_RETURNS,
    UILogEventsSupervisor.ON_DELEGATION_ERROR,
)


class SubagentItem(Protocol):
    """What an item must carry to be built into a subagent."""

    @property
    def name(self) -> str: ...

    description: str
    toolset: list[str]
    prompt: str


def literal_input(value: str, alias: str, *, optional: bool = False) -> dict[str, Any]:
    """Build a ``literal`` component input.

    Args:
        value: The literal value the input carries.
        alias: The prompt variable it arrives as.
        optional: Whether a prompt that never reads the variable is still valid.

    Returns:
        The input config.
    """
    key: dict[str, Any] = {"from": value, "as": alias, "literal": True}
    if optional:
        key["optional"] = True

    return key


class AgentComponentTemplate(BaseModel):
    """Component config built once per included item.

    Attributes:
        type: Always ``AgentComponent``.
        prompt_id: The prompt these items are built against.
        inputs: Component inputs, appended to per item.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["AgentComponent"]
    prompt_id: str
    inputs: Optional[list[Union[str, dict[str, Any]]]] = None

    @classmethod
    def default(cls, prompt_id: str) -> Self:
        """Return the template an agent item is built from.

        Args:
            prompt_id: The prompt the source ships for its items.

        Returns:
            A fresh instance per call.
        """
        return cls.model_validate(
            {
                "type": AgentComponent.__name__,
                "prompt_id": prompt_id,
                "inputs": [{"from": "context:goal", "as": "goal"}],
                "ui_log_events": [
                    "on_agent_reasoning",
                    "on_tool_execution_success",
                    "on_tool_execution_failed",
                    "on_agent_final_answer",
                ],
            }
        )


def promote_claimant(
    rewritten: dict, claimant_config: dict, agent_names: list[str]
) -> dict:
    """Make a claim-stripped claimant a supervisor over the named agents.

    Args:
        rewritten: The claimant's config with its claim already removed. Mutated in place and returned.
        claimant_config: The claiming component's authored config, read for the events it declared.
        agent_names: The synthesized components it gains, one ``subagents`` entry each.

    Returns:
        The rewritten config, with a subagent entry per name and the delegation UI log events merged in.
    """
    rewritten["subagents"] = list(rewritten.get("subagents") or []) + [
        {"name": name} for name in agent_names
    ]

    declared = list(claimant_config.get("ui_log_events") or [])
    rewritten["ui_log_events"] = declared + [
        str(event) for event in _DELEGATION_UI_LOG_EVENTS if str(event) not in declared
    ]

    return rewritten


def synthesized_agent_config(
    template: AgentComponentTemplate, item: SubagentItem, claimant_config: dict
) -> dict:
    """Synthesize a component config for one agent item.

    Args:
        template: The template the item is built from.
        item: The item to build.
        claimant_config: The claiming component's config, read for flags a subagent inherits.

    Returns:
        The template with what the item owns filled in.
    """
    config = template.model_dump(exclude_unset=True)
    config["name"] = item.name
    config["description"] = item.description
    config["toolset"] = list(item.toolset)
    config["inputs"] = list(config.get("inputs", [])) + [
        literal_input(item.prompt, ITEM_PROMPT_VARIABLE)
    ]
    if claimant_config.get("strict_validation"):
        config["strict_validation"] = True

    return config


def item_prompt_config(prompt_id: str, name: str) -> InMemoryPromptConfig:
    """Return the prompt a source's agent items are built against.

    Args:
        prompt_id: The id the source's template names.
        name: The prompt's display name.

    Returns:
        A fresh instance per call. The item's own prompt arrives as the ``item_prompt`` value, and ``goal`` is the
        task the coordinator delegates.
    """
    return InMemoryPromptConfig(
        prompt_id=prompt_id,
        name=name,
        unit_primitives=[GitLabUnitPrimitive.DUO_AGENT_PLATFORM],
        prompt_template={
            "system": f"{{{{ {ITEM_PROMPT_VARIABLE} }}}}",
            "user": "{{ goal }}",
        },
        params=PromptParams(timeout=120),
    )
