"""Build a LangGraph ``StateGraph`` from a flow config."""

from typing import Any

from gitlab_cloud_connector import CloudConnectorUser
from langgraph.graph import StateGraph

from ai_gateway.prompts import BasePromptRegistry
from ai_gateway.response_schemas.base import BaseResponseSchemaRegistry
from duo_workflow_service.agent_platform.utils.flow import (
    strip_ask_listed_pre_approvals,
)
from duo_workflow_service.agent_platform.v1.components.base import (
    AbortComponent,
    BaseComponent,
    EndComponent,
)
from duo_workflow_service.agent_platform.v1.components.supervisor.component import (
    extract_subagent_names,
)
from duo_workflow_service.agent_platform.v1.flows.flow_config import (
    FlowConfig,
    load_component_class,
)
from duo_workflow_service.agent_platform.v1.routers import Router
from duo_workflow_service.agent_platform.v1.state import FlowState
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.tools import Toolset
from lib.events import GLReportingEventContext
from lib.internal_events.client import InternalEventsClient

__all__ = ["FlowGraphBuilder"]


class FlowGraphBuilder:
    """Builds an uncompiled ``StateGraph`` from a flow config.

    Per-run dependencies (registries, workflow identity, user) are held on the
    builder; the config is an argument to ``build``. One builder can therefore
    build more than one graph.

    The builder is intentionally compile-agnostic: it returns the ``StateGraph``
    rather than a ``CompiledStateGraph``, because the caller owns the checkpointer
    decision.

    ``agent_platform/experimental/flows/base.py`` still carries its own near-copy of
    this algorithm.  It cannot adopt this builder as-is because it binds the
    experimental component, router and flow-config namespaces rather than the v1
    ones.  Converging them is tracked in
    https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/2759;
    until that lands, a fix made here very likely needs applying there too.  The copies
    have already drifted: this one accepts a mapping-shaped router condition input,
    that one does not.
    """

    def __init__(
        self,
        *,
        tools_registry: ToolsRegistry,
        prompt_registry: BasePromptRegistry,
        schema_registry: BaseResponseSchemaRegistry,
        workflow_id: str,
        workflow_type: GLReportingEventContext,
        user: CloudConnectorUser,
        internal_event_client: InternalEventsClient,
    ):
        self._tools_registry = tools_registry
        self._flow_prompt_registry = prompt_registry
        self._flow_schema_registry = schema_registry
        self._workflow_id = workflow_id
        self._workflow_type = workflow_type
        self._user = user
        self._internal_event_client = internal_event_client

    def build(self, flow_config: FlowConfig) -> StateGraph:
        """Build the graph for ``flow_config``, entry point set, ready to compile."""
        if flow_config.flow.entry_point is None:
            raise ValueError(
                "Can not build flow graph: entry_point is not defined in the flow config."
            )

        graph = StateGraph(FlowState)
        components = self._build_components(flow_config, graph)
        self._build_routers(flow_config, components, graph)

        entry_component = components[flow_config.flow.entry_point]
        graph.set_entry_point(entry_component.__entry_hook__())

        return graph

    def _build_components(
        self, flow_config: FlowConfig, graph: StateGraph
    ) -> dict[str, BaseComponent]:
        end_component = EndComponent(
            name="end",
            flow_id=self._workflow_id,
            flow_type=self._workflow_type,
            user=self._user,
        )
        end_component.attach(graph)

        abort_component = AbortComponent(
            name="abort",
            flow_id=self._workflow_id,
            flow_type=self._workflow_type,
            user=self._user,
        )
        abort_component.attach(graph)

        components: dict[str, BaseComponent] = {
            "end": end_component,
            "abort": abort_component,
        }

        # Single-pass construction with deferred queue for components
        # that depend on other components (e.g. supervisors need subagents).
        deferred: list[dict] = []

        for comp_config in flow_config.components:
            comp_params = self._prepare_component_params(comp_config, flow_config)

            if self._has_unresolved_dependencies(comp_config, components):
                deferred.append(comp_config)
                continue

            self._instantiate_component(comp_config, comp_params, components)

        # Build deferred components — their dependencies are now available
        for comp_config in deferred:
            comp_params = self._prepare_component_params(comp_config, flow_config)
            self._instantiate_component(comp_config, comp_params, components)

        return components

    def _prepare_component_params(
        self, comp_config: dict, flow_config: FlowConfig
    ) -> dict:
        """Prepare constructor parameters from a component config dict."""
        comp_params = {k: v for k, v in comp_config.items() if k != "type"}

        comp_params.update(
            {
                "prompt_registry": self._flow_prompt_registry,
                "schema_registry": self._flow_schema_registry,
                "flow_id": self._workflow_id,
                "flow_type": self._workflow_type,
                "user": self._user,
                "environment": flow_config.environment,
            }
        )

        if "pre_approved_tools" in comp_params:
            comp_params["pre_approved_tools"] = strip_ask_listed_pre_approvals(
                comp_params["pre_approved_tools"], self._tools_registry
            )

        if "toolset" in comp_params:
            comp_params["toolset"] = self._parse_toolset(
                comp_params["toolset"], flow_config
            )
        elif "tool_name" in comp_params:
            comp_params["toolset"] = self._tools_registry.toolset(
                [comp_params["tool_name"]]
            )

        return comp_params

    def _has_unresolved_dependencies(
        self,
        comp_config: dict,
        components: dict[str, BaseComponent],
    ) -> bool:
        """Check if a component has dependencies that haven't been built yet.

        A component has unresolved dependencies when it declares ``subagents``
        and at least one of those agents has not yet been built.  This applies to
        ``SupervisorAgentComponent`` configs that include ``subagents``.
        """
        subagents = comp_config.get("subagents", [])
        if not subagents:
            return False

        try:
            subagent_names = extract_subagent_names(subagents)
        except ValueError as exc:
            comp_name = comp_config.get("name", "<unknown>")
            raise ValueError(
                f"Component '{comp_name}' has a malformed subagents entry: {exc}"
            ) from exc
        return any(name not in components for name in subagent_names)

    def _instantiate_component(
        self,
        comp_config: dict,
        comp_params: dict,
        components: dict[str, BaseComponent],
    ) -> None:
        """Instantiate a single component and add it to the components dict.

        The shared components dict is injected as ``_built_components`` for
        ``AgentComponent`` configs only.  That type is registered in the v1
        :class:`ComponentRegistry` as a factory which dispatches to
        :class:`SupervisorAgentComponent` when the config declares ``subagents``,
        and it needs the pool to resolve those references.  A plain
        :class:`AgentComponent` pops and discards the key.

        After the component is created, the builder inspects its
        ``subagent_components`` attribute (present on
        :class:`SupervisorAgentComponent`) and removes the consumed subagents
        from the shared dict.  This keeps the mutation explicit and owned by the
        builder rather than hidden inside the factory.
        """
        comp_name = comp_config["name"]
        comp_type = comp_config["type"]
        comp_class = load_component_class(comp_type)

        if comp_name in components:
            raise ValueError(
                f"Duplicate component name: '{comp_name}'. Component names must be unique."
            )

        # AgentComponent configs are handled by a factory that needs the shared
        # components dict to resolve subagent references (for supervisor dispatch).
        if comp_type == "AgentComponent":
            comp_params["_built_components"] = components

        component = comp_class(**comp_params)
        components[comp_name] = component

        # If the newly created component consumed subagents (i.e. it is a
        # SupervisorAgentComponent), remove those subagents from the shared dict
        # so they are not exposed as top-level components (entry points, routers,
        # etc.).
        if hasattr(component, "subagent_components"):
            for consumed_name in component.subagent_components:
                components.pop(consumed_name, None)

    def _build_routers(
        self,
        flow_config: FlowConfig,
        components: dict[str, BaseComponent],
        graph: StateGraph,
    ) -> None:
        """Build and attach routers to the graph based on configuration.

        Creates routers that orders components in the flow graph.
        Supports conditional routing based on component outputs.

        Args:
            flow_config: The flow config whose ``routers`` to attach
            components: Dictionary of component instances keyed by name
            graph: The StateGraph instance to attach routers to

        Example conditional router configuration:

        - from: "human_input"
            condition:
                input: "status"
                routes:
                    "Execution": "agent"
                    "default_route": "end"
        """
        for router_config in flow_config.routers:
            from_comp = components[router_config["from"]]

            if "condition" in router_config:
                to_components = {}
                for route_key, comp_name in router_config["condition"][
                    "routes"
                ].items():
                    to_components[route_key] = components[comp_name]

                # A condition input is either a plain state-path string or, like
                # component inputs, a mapping ({from: ..., optional: true}) so a
                # router can branch on a key that may be absent from the state.
                # BaseRouter parses both forms via IOKey.parse_key.
                input_field = router_config["condition"]["input"]
                if not isinstance(input_field, (str, dict)):
                    raise ValueError("Router input must be a string or a mapping.")

                router = Router(
                    from_component=from_comp,
                    input=router_config["condition"]["input"],
                    to_component=to_components,
                    flow_id=self._workflow_id,
                    flow_type=self._workflow_type,
                    internal_event_client=self._internal_event_client,
                )
            else:
                to_comp = components[router_config["to"]]
                router = Router(from_component=from_comp, to_component=to_comp)

            router.attach(graph)

    def _parse_toolset(self, toolset_config: list, flow_config: FlowConfig) -> Toolset:
        """Parse toolset configuration and extract tool options.

        Supports two formats:
        1. Simple string: "tool_name"
        2. Dict with options: {"tool_name": {"option": "value"}}

        For flows with ``environment: chat``, all MCP tools connected to the
        session are automatically appended to the toolset, preserving the
        pre-Phase-1 behaviour for interactive chat assistants (e.g. Duo CLI,
        Interactive Developer, Software Development).  Flows with
        ``environment: ambient`` receive only the tools explicitly declared in
        their YAML config.

        Returns a Toolset with the appropriate tool options applied.
        """
        tool_names: list[str] = []
        tool_options: dict[str, dict[str, Any]] = {}

        for item in toolset_config:
            if isinstance(item, str):
                tool_names.append(item)
            elif isinstance(item, dict):
                for tool_name, options in item.items():
                    tool_names.append(tool_name)
                    if options:
                        tool_options[tool_name] = options

        if flow_config.should_auto_inject_mcp_tools():
            tool_names += self._tools_registry.mcp_tool_names()

        return self._tools_registry.toolset(tool_names, tool_options=tool_options)
