import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Dict, Optional, override

import jsonschema
from dependency_injector.wiring import Provide, inject
from gitlab_cloud_connector import CloudConnectorUser
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.types import Command

from ai_gateway.container import ContainerApplication
from ai_gateway.prompts import BasePromptRegistry, InMemoryPromptRegistry
from contract import contract_pb2
from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
    EndComponent,
)
from duo_workflow_service.agent_platform.experimental.components.supervisor.component import (
    SupervisorAgentComponent,
)
from duo_workflow_service.agent_platform.experimental.flows.flow_config import (
    FlowConfig,
    load_component_class,
)
from duo_workflow_service.agent_platform.experimental.routers import Router
from duo_workflow_service.agent_platform.experimental.state import FlowState
from duo_workflow_service.agent_platform.experimental.state.base import (
    FlowEvent,
    FlowEventType,
)
from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    WorkflowStatusEventEnum,
)
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.entities.agent_user_environment import _AGENT_SKILLS_ID
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    ToolStatus,
    UiChatLog,
    WorkflowStatusEnum,
)
from duo_workflow_service.interceptors.route import support_self_hosted_billing
from duo_workflow_service.tracking.errors import log_exception
from duo_workflow_service.workflows.abstract_workflow import (
    AbstractWorkflow,
)
from duo_workflow_service.workflows.type_definitions import AdditionalContext
from lib.events import GLReportingEventContext
from lib.internal_events.client import InternalEventsClient

__all__ = ["Flow"]

_EXECUTOR_CONTEXT = [
    "os_information",
    "shell_information",
    "agent_user_environment",
    "user_rule",
]


class UserDecision(StrEnum):
    APPROVE = "approval"
    REJECT = "rejection"


@support_self_hosted_billing(class_schema="flow/experimental")
class Flow(AbstractWorkflow):
    _config: FlowConfig
    _flow_prompt_registry: BasePromptRegistry

    # pylint: disable=dangerous-default-value
    @inject
    def __init__(
        self,
        workflow_id: str,
        workflow_metadata: Dict[str, Any],
        workflow_type: GLReportingEventContext,
        config: FlowConfig,
        mcp_tools: list[contract_pb2.McpTool] = [],
        user: Optional[CloudConnectorUser] = None,
        additional_context: Optional[list[AdditionalContext]] = None,
        approval: Optional[contract_pb2.Approval] = None,
        prompt_registry: BasePromptRegistry = Provide[
            ContainerApplication.pkg_prompts.prompt_registry
        ],
        internal_event_client: InternalEventsClient = Provide[
            ContainerApplication.internal_event.client
        ],
        **kwargs,
    ):
        super().__init__(
            workflow_id=workflow_id,
            workflow_metadata=workflow_metadata,
            workflow_type=workflow_type,
            mcp_tools=mcp_tools,
            user=user,
            additional_context=additional_context,
            approval=approval,
            prompt_registry=prompt_registry,
            internal_event_client=internal_event_client,
            **kwargs,
        )
        self._config = config

        self._flow_prompt_registry = InMemoryPromptRegistry(prompt_registry)
        if self._config.prompts:
            for prompt_config in self._config.prompts:
                prompt_id = prompt_config["prompt_id"]
                self._flow_prompt_registry.register_prompt(
                    prompt_id=prompt_id,
                    prompt_data=prompt_config,
                )

    # pylint: enable=dangerous-default-value

    @override
    def get_workflow_state(self, goal: str) -> FlowState:  # type: ignore[override]
        initial_ui_chat_log = UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            content=f"Starting Flow: {goal}",
            message_id=None,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.SUCCESS,
            correlation_id=None,
            tool_info=None,
            additional_context=None,
            message_sub_type=None,
        )

        return FlowState(
            status=WorkflowStatusEnum.NOT_STARTED,
            conversation_history={},
            ui_chat_log=[initial_ui_chat_log],
            context={
                "project_id": self._project.get("id"),  # type: ignore[union-attr]
                "project_http_url_to_repo": self._project.get("http_url_to_repo"),  # type: ignore[union-attr]
                "project_default_branch": self._project.get("default_branch"),  # type: ignore[union-attr]
                "goal": goal,
                "current_date": datetime.now().strftime("%Y-%m-%d"),
                "workflow_id": self._workflow_id,
                "inputs": self._process_additional_context(
                    self._additional_context or []
                ),
            },
        )

    def _process_additional_context(
        self, additional_context: list[AdditionalContext]
    ) -> Dict:
        processed_additional_context = {}

        jsonschemas_by_category = self._config.input_json_schemas_by_category()
        for item in additional_context:
            if (
                item.category in _EXECUTOR_CONTEXT
            ):  # This category is passed in from the executor
                if item.id == _AGENT_SKILLS_ID:
                    processed_additional_context["workspace_agent_skills"] = (
                        item.content
                    )
                else:
                    processed_additional_context[item.category] = item.content
                continue

            if item.category not in jsonschemas_by_category.keys():
                self.log.warn(
                    f"Unknown additional context envelope {item.category} has been skipped",
                    additional_context_category=item.category,
                )
                continue
            try:
                schema = jsonschemas_by_category.get(item.category)
                if not item.content:
                    raise ValueError(
                        f"content must be specified for input '{item.category}'."
                    )

                content_object = json.loads(item.content)
                jsonschema.validate(content_object, schema)
                processed_additional_context[item.category] = content_object
            except jsonschema.ValidationError as e:
                raise ValueError(
                    f"input '{item.category}' does not match specified schema: {e}"
                )
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in input item, {e}")

        return processed_additional_context

    def _resume_command(self, goal: str) -> Command:
        event = FlowEvent(event_type=FlowEventType.RESPONSE, message=goal)
        if not self._approval:
            # Handle case where approval is None
            return Command(resume=event)

        match self._approval.WhichOneof("user_decision"):
            case UserDecision.APPROVE:
                event = FlowEvent(event_type=FlowEventType.APPROVE)
            case UserDecision.REJECT:
                event = FlowEvent(
                    event_type=FlowEventType.REJECT,
                    message=self._approval.rejection.message,
                )
            case _:
                # This should never happen according to contract.proto
                raise ValueError(
                    f"Unexpected approval decision: {self._approval.WhichOneof('user_decision')}"
                )

        return Command(resume=event)

    @override
    async def get_graph_input(
        self, goal: str, status_event: str, checkpoint_tuple: Any
    ) -> Any:
        match status_event:
            case WorkflowStatusEventEnum.START:
                return self.get_workflow_state(goal)
            case WorkflowStatusEventEnum.RESUME:
                return self._resume_command(goal)
            case WorkflowStatusEventEnum.RETRY:
                if checkpoint_tuple is None:
                    return self.get_workflow_state(
                        goal
                    )  # no saved checkpoints from last run
                return None  # retry from last checkpoint
            case _:
                return None

    def _build_components(
        self, tools_registry: ToolsRegistry, graph: StateGraph
    ) -> dict[str, BaseComponent]:
        end_component = EndComponent(
            name="end",
            flow_id=self._workflow_id,
            flow_type=self._workflow_type,
            user=self._user,
        )
        end_component.attach(graph)

        components: dict[str, BaseComponent] = {"end": end_component}

        # Single-pass construction with deferred queue for components
        # that depend on other components (e.g. supervisors need subagents).
        deferred: list[dict] = []

        for comp_config in self._config.components:
            comp_params = self._prepare_component_params(comp_config, tools_registry)

            if self._has_unresolved_dependencies(comp_config, components):
                deferred.append(comp_config)
                continue

            self._instantiate_component(comp_config, comp_params, components)

        # Build deferred components — their dependencies are now available
        for comp_config in deferred:
            comp_params = self._prepare_component_params(comp_config, tools_registry)
            self._instantiate_component(comp_config, comp_params, components)

        return components

    def _prepare_component_params(
        self, comp_config: dict, tools_registry: ToolsRegistry
    ) -> dict:
        """Prepare constructor parameters from a component config dict."""
        comp_params = {k: v for k, v in comp_config.items() if k != "type"}

        comp_params.update(
            {
                "prompt_registry": self._flow_prompt_registry,
                "flow_id": self._workflow_id,
                "flow_type": self._workflow_type,
                "user": self._user,
            }
        )

        if "toolset" in comp_params:
            comp_params["toolset"] = self._parse_toolset(
                tools_registry, comp_params["toolset"]
            )
        elif "tool_name" in comp_params:
            comp_params["toolset"] = tools_registry.toolset([comp_params["tool_name"]])

        return comp_params

    def _has_unresolved_dependencies(
        self,
        comp_config: dict,
        components: dict[str, BaseComponent],
    ) -> bool:
        """Check if a component has dependencies that haven't been built yet.

        Currently only SupervisorAgentComponent has dependencies (managed_agents).
        """
        if comp_config["type"] != SupervisorAgentComponent.__name__:
            return False

        managed_agent_names = comp_config.get("managed_agents", [])
        return any(name not in components for name in managed_agent_names)

    def _instantiate_component(
        self,
        comp_config: dict,
        comp_params: dict,
        components: dict[str, BaseComponent],
    ) -> None:
        """Instantiate a single component and add it to the components dict."""
        comp_name = comp_config["name"]
        comp_class = load_component_class(comp_config["type"])

        if comp_name in components:
            raise ValueError(
                f"Duplicate component name: '{comp_name}'. Component names must be unique."
            )

        # For supervisors, resolve subagent references and inject them
        if comp_config["type"] == SupervisorAgentComponent.__name__:
            subagents = self._resolve_subagents(comp_config, components)
            comp_params["subagent_components"] = subagents

        components[comp_name] = comp_class(**comp_params)

    def _resolve_subagents(
        self,
        supervisor_config: dict,
        components: dict[str, BaseComponent],
    ) -> dict[str, BaseComponent]:
        """Resolve managed subagent references for a supervisor.

        Pops each managed_agents name from the components dict, transferring
        ownership to the supervisor. This prevents subagents from appearing
        in routers, being shared across supervisors, or being used as entry
        points. Type validation is handled by SupervisorAgentComponent's
        model validator.

        Args:
            supervisor_config: The supervisor's component config dict.
            components: Already-built component instances (mutated — subagents
                are removed).

        Returns:
            Dictionary mapping subagent names to their component instances.

        Raises:
            ValueError: If a managed agent name is not found in components.
        """
        supervisor_name = supervisor_config["name"]
        managed_agent_names = supervisor_config.get("managed_agents", [])
        subagents: dict[str, BaseComponent] = {}

        for agent_name in managed_agent_names:
            if agent_name not in components:
                raise ValueError(
                    f"Supervisor '{supervisor_name}' references managed agent "
                    f"'{agent_name}' which is not defined in components."
                )

            subagents[agent_name] = components.pop(agent_name)

        return subagents

    def _build_routers(
        self, components: dict[str, BaseComponent], graph: StateGraph
    ) -> None:
        """Build and attach routers to the graph based on configuration.

        Creates routers that orders components in the flow graph.
        Supports conditional routing based on component outputs.

        Args:
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
        for router_config in self._config.routers:
            from_comp = components[router_config["from"]]

            if "condition" in router_config:
                to_components = {}
                for route_key, comp_name in router_config["condition"][
                    "routes"
                ].items():
                    to_components[route_key] = components[comp_name]

                input_field = router_config["condition"]["input"]
                if not isinstance(input_field, str):
                    raise ValueError("Router input must be a string.")

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

    @override
    def _compile(
        self,
        goal: str,
        tools_registry: ToolsRegistry,
        checkpointer: BaseCheckpointSaver,
    ) -> Any:
        if self._config.flow.entry_point is None:
            raise ValueError(
                "Can not compile flow: entry_point is not defined in the flow config."
            )

        graph = StateGraph(FlowState)
        components = self._build_components(tools_registry, graph)
        self._build_routers(components, graph)

        entry_component = components[self._config.flow.entry_point]
        graph.set_entry_point(entry_component.__entry_hook__())

        return graph.compile(checkpointer=checkpointer)

    def _parse_toolset(
        self, tools_registry: ToolsRegistry, toolset_config: list
    ) -> Any:
        """Parse toolset configuration and extract tool options.

        Supports two formats:
        1. Simple string: "tool_name"
        2. Dict with options: {"tool_name": {"option": "value"}}

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

        return tools_registry.toolset(tool_names, tool_options=tool_options)

    @override
    async def _handle_workflow_failure(
        self, error: BaseException, compiled_graph: Any, graph_config: Any
    ):
        log_exception(
            error, extra={"workflow_id": self._workflow_id, "source": __name__}
        )
