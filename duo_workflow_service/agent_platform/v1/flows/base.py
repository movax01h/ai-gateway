import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Dict, Optional, override
from uuid import uuid4

import jsonschema
from dependency_injector.wiring import Provide, inject
from gitlab_cloud_connector import CloudConnectorUser
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import StateGraph
from langgraph.types import Command, Overwrite

from ai_gateway.container import ContainerApplication
from ai_gateway.prompts import BasePromptRegistry, InMemoryPromptRegistry
from ai_gateway.response_schemas import InlineResponseSchemaRegistry
from ai_gateway.response_schemas.base import BaseResponseSchemaRegistry
from contract import contract_pb2
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
from duo_workflow_service.agent_platform.v1.state.base import FlowEvent, FlowEventType
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
from duo_workflow_service.errors.typing import GENERIC_WORKFLOW_ERROR_MESSAGE
from duo_workflow_service.interceptors.route import support_self_hosted_billing
from duo_workflow_service.tracking.errors import log_exception
from duo_workflow_service.workflows.abstract_workflow import (
    AbstractWorkflow,
)
from duo_workflow_service.workflows.type_definitions import AdditionalContext
from lib.events import GLReportingEventContext
from lib.internal_events.client import InternalEventsClient

__all__ = ["Flow", "persist_error_to_ui_chat_log"]

_EXECUTOR_CONTEXT = [
    "os_information",
    "shell_information",
    "agent_user_environment",
    "user_rule",
]


class UserDecision(StrEnum):
    APPROVE = "approval"
    REJECT = "rejection"


async def persist_error_to_ui_chat_log(
    compiled_graph: Any,
    graph_config: Any,
    existing_logs: list,
    log: Any,
    workflow_id: str,
) -> None:
    """Persist a generic error entry to the UI chat log via aupdate_state.

    Appends a ``WORKFLOW_END`` / ``FAILURE`` log entry to *existing_logs* and
    writes the combined list to the graph checkpoint using ``Overwrite`` so
    that the reducer is bypassed entirely.  If ``aupdate_state`` raises, the
    exception is caught and a warning is logged instead of propagating.

    Args:
        compiled_graph: The compiled LangGraph instance (must not be ``None``).
        graph_config: The graph configuration dict passed to ``aupdate_state``.
        existing_logs: The current ``ui_chat_log`` entries to preserve.
        log: A structlog logger (or any object with a ``warning`` method).
        workflow_id: The workflow identifier used in the warning log.
    """
    error_message = GENERIC_WORKFLOW_ERROR_MESSAGE

    error_log = UiChatLog(
        message_type=MessageTypeEnum.AGENT,
        message_sub_type=None,
        content=error_message,
        timestamp=datetime.now(timezone.utc).isoformat(),
        message_id=f"error-{str(uuid4())}",
        status=ToolStatus.FAILURE,
        correlation_id=None,
        tool_info=None,
        additional_context=None,
    )

    try:
        # We pass the full ui_chat_log list (existing + error)
        # because aupdate_state does not reliably apply the
        # _ui_chat_log_reducer after a graph failure.
        await compiled_graph.aupdate_state(
            graph_config,
            {"ui_chat_log": Overwrite(value=existing_logs + [error_log])},
        )
    except Exception as exc:
        log.warning(
            "Failed to persist error ui_chat_log to checkpoint",
            workflow_id=workflow_id,
            exc_info=exc,
        )


@support_self_hosted_billing(class_schema="flow/v1")
class Flow(AbstractWorkflow):
    _config: FlowConfig
    _flow_prompt_registry: BasePromptRegistry
    _flow_schema_registry: BaseResponseSchemaRegistry

    # pylint: disable=dangerous-default-value
    @inject
    def __init__(
        self,
        workflow_id: str,
        workflow_metadata: Dict[str, Any],
        workflow_type: GLReportingEventContext,
        user: CloudConnectorUser,
        config: FlowConfig,
        mcp_tools: list[contract_pb2.McpTool] = [],
        additional_context: Optional[list[AdditionalContext]] = None,
        approval: Optional[contract_pb2.Approval] = None,
        prompt_registry: BasePromptRegistry = Provide[
            ContainerApplication.pkg_prompts.prompt_registry
        ],
        schema_registry: BaseResponseSchemaRegistry = Provide[
            ContainerApplication.pkg_schemas.schema_registry
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
                prompt_id = prompt_config.prompt_id
                self._flow_prompt_registry.register_prompt(
                    prompt_id=prompt_id,
                    prompt_data=prompt_config.to_prompt_data(),
                )

        self._flow_schema_registry = InlineResponseSchemaRegistry(schema_registry)
        if self._config.response_schemas:
            for schema_config in self._config.response_schemas:
                self._flow_schema_registry.register_schema(
                    schema_config.schema_id, schema_config.to_schema_dict()
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
                "session_url": self._session_url,
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
        if not self._approval or self._approval.WhichOneof("user_decision") is None:
            # Handle case where approval is None
            return Command(resume=event)

        ui_chat_log_update: list[UiChatLog] = []

        match self._approval.WhichOneof("user_decision"):
            case UserDecision.APPROVE:
                event = FlowEvent(event_type=FlowEventType.APPROVE)
            case UserDecision.REJECT:
                if message := self._approval.rejection.message:
                    event = FlowEvent(
                        event_type=FlowEventType.MODIFY,
                        message=message,
                    )
                    # Emit the user's feedback as a UI chat log entry, mirroring
                    # the chat workflow pattern. Uses Command(update=...) so the
                    # entry is appended alongside prior approval logs.
                    ui_chat_log_update = [
                        UiChatLog(
                            message_type=MessageTypeEnum.USER,
                            message_sub_type=None,
                            content=message,
                            message_id=f"user-{str(uuid4())}",
                            timestamp=datetime.now(timezone.utc).isoformat(),
                            status=ToolStatus.SUCCESS,
                            correlation_id=None,
                            tool_info=None,
                            additional_context=None,
                        )
                    ]
                else:
                    event = FlowEvent(
                        event_type=FlowEventType.REJECT,
                    )
            case _:
                # This should never happen according to contract.proto
                raise ValueError(
                    f"Unexpected approval decision: {self._approval.WhichOneof('user_decision')}"
                )

        if ui_chat_log_update:
            return Command(resume=event, update={"ui_chat_log": ui_chat_log_update})
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
                "schema_registry": self._flow_schema_registry,
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

        The shared ``_built_components`` dict is always injected into the params
        so that factory callables (e.g. the ``AgentComponent`` factory registered
        in the v1 :class:`ComponentRegistry`) can resolve subagent references when
        needed.  Factories that do not require it (plain :class:`AgentComponent`)
        simply pop and discard the key.

        After the component is created, ``Flow`` inspects its
        ``subagent_components`` attribute (present on
        :class:`SupervisorAgentComponent`) and removes the consumed subagents
        from the shared dict.  This keeps the mutation explicit and owned by
        ``Flow`` rather than hidden inside the factory.
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

        Raises:
            ValueError: If a tool name does not exist in the registry. Note that
                MCP tools are supplied at runtime and are not present in the
                registry at validation time; configs using MCP tool names in
                non-chat-partial environments will fail validation.
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

        known_names = tools_registry.known_tool_names
        unknown_names = [name for name in tool_names if name not in known_names]
        if unknown_names:
            raise ValueError(
                f"Unknown tool name(s) in toolset: {unknown_names}. "
                f"Verify that the tool names are spelled correctly. "
                f"Note: MCP tools are supplied at runtime and cannot be validated statically."
            )

        return tools_registry.toolset(tool_names, tool_options=tool_options)

    @override
    async def _handle_workflow_failure(
        self, error: BaseException, compiled_graph: Any, graph_config: Any
    ):
        log_exception(
            error, extra={"workflow_id": self._workflow_id, "source": __name__}
        )

        if compiled_graph is not None:
            existing_logs = (
                list(self.checkpoint_notifier.ui_chat_log)
                if self.checkpoint_notifier
                else []
            )
            await persist_error_to_ui_chat_log(
                compiled_graph=compiled_graph,
                graph_config=graph_config,
                existing_logs=existing_logs,
                log=self.log,
                workflow_id=self._workflow_id,
            )
