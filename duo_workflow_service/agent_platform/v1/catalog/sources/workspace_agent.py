"""The workspace agent source: agent templates a customer authored in their repository.

Items arrive in full with the request rather than being fetched, so there is nothing to version and no id to address one
by: a flow references them all at once, with :data:`WILDCARD_ITEM_ID`.

Binding promotes the claiming component to a supervisor and synthesizes one ``AgentComponent`` per item beneath it. Item
prompts travel as ``literal`` inputs, never spliced into Jinja source, so client text is a template value rather than
code.
"""

from typing import Any, Literal, Optional, Self, Union, override

from gitlab_cloud_connector import GitLabUnitPrimitive
from pydantic import BaseModel, ConfigDict

from ai_gateway.prompts.config.base import InMemoryPromptConfig, PromptParams
from duo_workflow_service.agent_platform.v1.catalog.errors import (
    CatalogItemConfigError,
    CatalogItemsError,
)
from duo_workflow_service.agent_platform.v1.catalog.items import (
    WorkspaceAgent,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.base import (
    BindRequest,
    CatalogSource,
)
from duo_workflow_service.agent_platform.v1.catalog.sources.reference import (
    CatalogItemRef,
    CatalogItemSource,
    CatalogItemType,
    looks_like_ref,
)
from duo_workflow_service.agent_platform.v1.components.agent.component import (
    AgentComponent,
)
from duo_workflow_service.agent_platform.v1.components.supervisor.ui_log import (
    UILogEventsSupervisor,
)

__all__ = [
    "WILDCARD_ITEM_ID",
    "AgentComponentTemplate",
    "WorkspaceAgentSource",
    "agent_template_prompt",
]

# Reference every item of a kind, rather than one by id.
WILDCARD_ITEM_ID = "*"

# Named after the source and kind, so two sources offering the same kind never share a
# prompt.
_PROMPT_ID = f"{CatalogItemSource.WORKSPACE}_{CatalogItemType.AGENT_TEMPLATE}_prompt"

# Prompt variables injected as literal inputs: the item's own prompt, and whether the
# request carried any items for the claimant to coordinate.
_ITEM_PROMPT_VARIABLE = "item_prompt"
_HAS_AGENTS_VARIABLE = "has_workspace_agents"

# Merged into the claimant's `ui_log_events` on promotion. Not declarable per flow,
# because a plain `AgentComponent` rejects them.
_DELEGATION_UI_LOG_EVENTS = (
    UILogEventsSupervisor.ON_DELEGATION,
    UILogEventsSupervisor.ON_DELEGATION_RETURNS,
    UILogEventsSupervisor.ON_DELEGATION_ERROR,
)


def _literal_input(value: str, alias: str, *, optional: bool = False) -> dict[str, Any]:
    """Build a ``literal`` component input.

    Args:
        value: The literal value the input carries.
        alias: The prompt variable it arrives as.
        optional: Whether a prompt that never reads the variable is still valid. No runtime effect, so it is
            opt-in per input.

    Returns:
        The input config.
    """
    key: dict[str, Any] = {"from": value, "as": alias, "literal": True}
    if optional:
        key["optional"] = True

    return key


class AgentComponentTemplate(BaseModel):
    """Component config built once per included item.

    Declares only the keys binding reasons about; ``extra="allow"`` forwards the rest to ``AgentComponent``, which
    stays the single definition of its own config. ``name``, ``description`` and ``toolset`` are set per item and
    would be overwritten if declared here.

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
    def default(cls) -> Self:
        """Return the template a workspace agent item is built from.

        Returns:
            A fresh instance per call, so nothing accumulates across items or runs. Built from a dict, like a
            YAML-supplied template would be, so every forwarded key takes the same ``extra="allow"`` path.
        """
        return cls.model_validate(
            {
                "type": AgentComponent.__name__,
                # Left unversioned, so it resolves to `agent_template_prompt`
                # below rather than to a file-based definition.
                "prompt_id": _PROMPT_ID,
                # The delegated task, and the only thing a subagent reads from state.
                # `bind_to_supervisor` swaps this for a subsession-scoped key at attach
                # time; declared anyway, because prompt-variable coverage is checked at
                # construction, before that swap.
                "inputs": [{"from": "context:goal", "as": "goal"}],
                "ui_log_events": [
                    "on_agent_reasoning",
                    "on_tool_execution_success",
                    "on_tool_execution_failed",
                    "on_agent_final_answer",
                ],
            }
        )


def _validate_agents(request: BindRequest) -> None:
    """Check the items against this flow: a free name, and tools the registry can resolve.

    Args:
        request: The claim being bound.

    Raises:
        CatalogItemsError: If an item's name is taken, or it declares an unusable tool. Namespacing clears ordinary
            component names, leaving one reachable collision: a component the author declared inside the namespace.
    """
    taken = {comp_config.get("name") for comp_config in request.components_config}

    for agent in request.items.workspace_agents:
        if agent.name in taken:
            raise CatalogItemsError(
                f"Workspace agent name '{agent.name}' collides with a component "
                f"name used by this flow."
            )

        try:
            request.tools_registry.toolset(list(agent.toolset))
        except ValueError as exc:
            # The registry raises ValueError; re-raised naming the item, which it
            # cannot see.
            raise CatalogItemsError(
                f"Workspace agent '{agent.name}' declares an unusable tool: {exc}"
            ) from exc


def _rewrite_claimant(claimant_config: dict, agents: list[WorkspaceAgent]) -> dict:
    """Rewrite the claiming component for the items it resolved to.

    The catalog reference goes, statically named subagents stay, and ``has_workspace_agents`` is injected so the prompt
    can gate its delegation guidance. With items the claimant also gains a subagent entry per item and the delegation
    UI log events.

    Args:
        claimant_config: The claiming component's authored config.
        agents: The items it resolved to, possibly none.

    Returns:
        A new config; the authored one is left untouched.
    """
    rewritten = dict(claimant_config)
    # Optional, because every claimant gets the flag, including prompts with no
    # delegation guidance to gate.
    rewritten["inputs"] = list(claimant_config.get("inputs", [])) + [
        _literal_input(
            "true" if agents else "",
            _HAS_AGENTS_VARIABLE,
            optional=True,
        )
    ]

    static_entries = [
        entry
        for entry in claimant_config.get("subagents") or []
        if not looks_like_ref(entry)
    ]
    subagents = static_entries + [{"name": agent.name} for agent in agents]

    if subagents:
        rewritten["subagents"] = subagents
    else:
        # Nothing to supervise: drop the key so the factory builds a plain agent.
        rewritten.pop("subagents", None)

    if not agents:
        return rewritten

    # str() keeps the config dict plain: YAML-declared events arrive as strings, and
    # the component re-validates either form.
    declared = list(claimant_config.get("ui_log_events") or [])
    rewritten["ui_log_events"] = declared + [
        str(event) for event in _DELEGATION_UI_LOG_EVENTS if str(event) not in declared
    ]

    return rewritten


def _to_component_config(agent: WorkspaceAgent, claimant_config: dict) -> dict:
    """Synthesize a component config for one agent item.

    Args:
        agent: The item to build.
        claimant_config: The claiming component's config, read for flags a subagent inherits.

    Returns:
        The default template with what the item owns filled in. ``exclude_unset`` leaves the component's own
        defaults to apply.
    """
    config = AgentComponentTemplate.default().model_dump(exclude_unset=True)
    # These belong to the item, whatever the template says.
    config["name"] = agent.name
    config["description"] = agent.description
    config["toolset"] = list(agent.toolset)
    # Required rather than optional: the prompt is ours, and always reads it.
    config["inputs"] = list(config.get("inputs", [])) + [
        _literal_input(agent.prompt, _ITEM_PROMPT_VARIABLE)
    ]
    # A synthesized subagent is validated as strictly as the component claiming it.
    if claimant_config.get("strict_validation"):
        config["strict_validation"] = True

    return config


def agent_template_prompt() -> InMemoryPromptConfig:
    """Return the prompt every workspace agent item is built against.

    Shipped here rather than declared per flow: a flow includes items without writing anything for them,
    and every item of this kind is built the same way. Letting a flow override it is a later phase.

    Returns:
        A fresh instance per call, so nothing accumulates across flows. The item's own prompt arrives as
        a value in ``item_prompt``, never spliced into the template source, so client text is data rather
        than template code. ``goal`` is the task the coordinator delegates.
    """
    return InMemoryPromptConfig(
        prompt_id=_PROMPT_ID,
        name="Workspace Agent Template",
        unit_primitives=[GitLabUnitPrimitive.DUO_AGENT_PLATFORM],
        prompt_template={
            "system": f"{{{{ {_ITEM_PROMPT_VARIABLE} }}}}",
            "user": "{{ goal }}",
        },
        params=PromptParams(timeout=120),
    )


class WorkspaceAgentSource(CatalogSource):
    """Builds workspace agent templates into subagents of the claiming component."""

    SOURCE = CatalogItemSource.WORKSPACE
    ITEM_TYPE = CatalogItemType.AGENT_TEMPLATE

    @override
    def validate_ref(self, ref: CatalogItemRef) -> None:
        """Reject a reference this source cannot serve.

        Args:
            ref: A reference the flow declared.

        Raises:
            CatalogItemConfigError: If a concrete id or a version is declared. Items arrive with the request, so
                there is neither to select by.
        """
        if ref.item_id != WILDCARD_ITEM_ID:
            raise CatalogItemConfigError(
                f"include entry '{ref}' must use item_id '{WILDCARD_ITEM_ID}': "
                f"workspace items arrive with the request, so there is no id to "
                f"address one by."
            )

        if ref.version is not None:
            raise CatalogItemConfigError(
                f"include entry '{ref}' must not declare a version: workspace items "
                f"arrive with the request, so there is no version to select."
            )

    @override
    def bind(self, request: BindRequest) -> list[dict]:
        """Promote the claimant to a supervisor over one component per item.

        Args:
            request: The reference, the claiming component, and the items to bind.

        Returns:
            The flow's components, with the claimant rewritten and one component appended per item. With no items
            the claimant builds as authored, so a flow that accepts items is unchanged by a request that sends none.
        """
        agents = request.items.workspace_agents

        _validate_agents(request)

        claimant_config = request.claimant_config
        bound = [
            (
                _rewrite_claimant(comp_config, agents)
                if comp_config is claimant_config
                else comp_config
            )
            for comp_config in request.components_config
        ]
        bound.extend(_to_component_config(agent, claimant_config) for agent in agents)

        return bound
