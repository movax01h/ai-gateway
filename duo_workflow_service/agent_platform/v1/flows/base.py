import json
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Dict, NoReturn, Optional, override
from uuid import uuid4

import jsonschema
from dependency_injector.wiring import Provide, inject
from gitlab_cloud_connector import CloudConnectorUser
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph
from langgraph.types import Command, Overwrite

from ai_gateway.container import ContainerApplication
from ai_gateway.prompts import BasePromptRegistry, InMemoryPromptRegistry
from ai_gateway.response_schemas import InlineResponseSchemaRegistry
from ai_gateway.response_schemas.base import BaseResponseSchemaRegistry
from contract import contract_pb2
from duo_workflow_service.agent_platform.utils.exceptions import (
    NotifiableAgentException,
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
from duo_workflow_service.errors.typing import (
    GENERIC_WORKFLOW_ERROR_MESSAGE,
    EnvelopeVersionMismatchException,
)
from duo_workflow_service.gitlab.gitlab_service_context import GitLabServiceContext
from duo_workflow_service.interceptors.route import support_self_hosted_billing
from duo_workflow_service.tracking.errors import log_exception
from duo_workflow_service.workflows.abstract_workflow import (
    AbstractWorkflow,
)
from duo_workflow_service.workflows.type_definitions import AdditionalContext
from lib.events import GLReportingEventContext
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import (
    merge_request_url_context,
    pipeline_source_context,
)
from lib.version import resolve_version

__all__ = ["Flow", "persist_error_to_ui_chat_log"]

_ENVELOPE_DEFAULT_VERSION = "1.0.0"
_ENVELOPE_DEFAULT_CONSTRAINT = "^1.0.0"

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
    error: BaseException | None = None,
) -> None:
    """Persist an error entry to the UI chat log via aupdate_state.

    Appends a ``WORKFLOW_END`` / ``FAILURE`` log entry to *existing_logs* and
    writes the combined list to the graph checkpoint using ``Overwrite`` so
    that the reducer is bypassed entirely.  If ``aupdate_state`` raises, the
    exception is caught and a warning is logged instead of propagating.

    When *error* is a :class:`NotifiableAgentException`, the safe ``ui_message``
    is surfaced to the UI. Any other exception (or ``None``) falls back to the
    generic catch-all message to avoid leaking internal details.

    Args:
        compiled_graph: The compiled LangGraph instance (must not be ``None``).
        graph_config: The graph configuration dict passed to ``aupdate_state``.
        existing_logs: The current ``ui_chat_log`` entries to preserve.
        log: A structlog logger (or any object with a ``warning`` method).
        workflow_id: The workflow identifier used in the warning log.
        error: The exception that caused the failure, used to extract a safe
            user-facing message when available.
    """
    if isinstance(error, NotifiableAgentException):
        error_message = error.ui_message
    else:
        error_message = GENERIC_WORKFLOW_ERROR_MESSAGE

    error_log = UiChatLog(
        message_type=MessageTypeEnum.AGENT,
        message_sub_type=None,
        content=error_message,
        timestamp=datetime.now(timezone.utc).isoformat(),
        message_id=f"error-{uuid4()!s}",
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
        streaming: bool = False,
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
        self._stream = streaming

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

        self._set_tracking_context_from_additional_context()

    # pylint: enable=dangerous-default-value

    def _set_tracking_context_from_additional_context(self) -> None:
        """Set merge_request_url and pipeline_source ContextVars from additional context."""
        context_mappings = [
            ("merge_request", "url", merge_request_url_context),
            ("pipeline", "source", pipeline_source_context),
        ]
        for item in self._additional_context or []:
            for category, key, context_var in context_mappings:
                if item.category == category and item.content:
                    try:
                        value = json.loads(item.content).get(key)
                        if value:
                            context_var.set(value)
                    except (json.JSONDecodeError, AttributeError) as e:
                        self.log.warning(
                            "Failed to parse additional context for tracking",
                            category=category,
                            error=str(e),
                        )

    @override
    def get_workflow_state(self, goal: str) -> FlowState:  # type: ignore[override]
        initial_ui_chat_log = UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            content=f"Starting Flow: {goal}",
            message_id=f"tool-{uuid4()!s}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.SUCCESS,
            correlation_id=None,
            tool_info=None,
            additional_context=None,
            message_sub_type=None,
        )

        if self._config.environment == "chat":
            initial_ui_chat_log["message_type"] = MessageTypeEnum.USER
            initial_ui_chat_log["content"] = goal

        gitlab_instance_info = GitLabServiceContext.get_current_instance_info()
        gitlab_service_context = {
            "gitlab_instance_type": (
                gitlab_instance_info.instance_type
                if gitlab_instance_info
                else "Unknown"
            ),
            "gitlab_instance_url": (
                gitlab_instance_info.instance_url if gitlab_instance_info else "Unknown"
            ),
            "gitlab_instance_version": (
                gitlab_instance_info.instance_version
                if gitlab_instance_info
                else "Unknown"
            ),
        }

        return FlowState(
            status=WorkflowStatusEnum.NOT_STARTED,
            conversation_history={},
            ui_chat_log=[initial_ui_chat_log],
            context={
                "project_id": self._project.get("id") if self._project else None,
                "project_http_url_to_repo": (
                    self._project.get("http_url_to_repo") if self._project else None
                ),
                "project_default_branch": (
                    self._project.get("default_branch") if self._project else None
                ),
                "namespace": self._namespace,
                "goal": goal,
                "current_date": datetime.now().strftime("%Y-%m-%d"),
                "workflow_id": self._workflow_id,
                "session_url": self._session_url,
                "inputs": self._process_additional_context(
                    self._additional_context or []
                ),
                **gitlab_service_context,
            },
            agent_context_limits={},
        )

    @override
    def _support_namespace_level_workflow(self) -> bool:
        return True

    def _resolve_envelope_content(
        self,
        category: str,
        envelopes: list[AdditionalContext],
        schema: Any,
        constraint: Optional[str],
    ) -> Any:
        # Flows without an explicit constraint implicitly require ^1.0.0: they
        # were authored against the 1.x series and must not silently receive a
        # breaking v2.x envelope.
        if constraint is None:
            self.log.warning(
                "No version_constraint declared for envelope category; falling back to default constraint",
                category=category,
                default_constraint=_ENVELOPE_DEFAULT_CONSTRAINT,
            )
        effective_constraint = (
            constraint if constraint is not None else _ENVELOPE_DEFAULT_CONSTRAINT
        )
        try:
            versioned: list[tuple[str, Any]] = []
            for item in envelopes:
                if not item.content:
                    raise ValueError(
                        f"content must be specified for input '{category}'."
                    )
                obj = json.loads(item.content)
                raw = (item.metadata or {}).get("version")
                if not isinstance(raw, str):
                    self.log.warning(
                        "Envelope payload does not declare a version; falling back to default version",
                        category=category,
                        default_version=_ENVELOPE_DEFAULT_VERSION,
                    )
                versioned.append(
                    (raw if isinstance(raw, str) else _ENVELOPE_DEFAULT_VERSION, obj)
                )

            versions = [v for v, _ in versioned]
            try:
                best = resolve_version(versions, effective_constraint)
            except ValueError:
                detail = (
                    f"No envelope for '{category}' satisfies constraint "
                    f"'{effective_constraint}'. Available versions: {versions}."
                )
                raise NotifiableAgentException(
                    "Your GitLab instance sent additional context in an incompatible "
                    "version. Please upgrade your GitLab instance to a compatible version "
                    "and try again.",
                    internal_detail=detail,
                ) from EnvelopeVersionMismatchException(detail)

            # resolve_version returns a string from its input list, so this
            # lookup is always safe. On duplicate version strings, last wins.
            content = dict(versioned)[best]

            jsonschema.validate(content, schema)
            return content
        except jsonschema.ValidationError as e:
            raise ValueError(f"input '{category}' does not match specified schema: {e}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in input item, {e}")

    def _process_additional_context(
        self, additional_context: list[AdditionalContext]
    ) -> Dict:
        processed_additional_context = {}

        jsonschemas_by_category = self._config.input_json_schemas_by_category()
        version_constraints_by_category = self._config.version_constraints_by_category()

        # Group non-executor envelopes by category, handling executor context inline.
        grouped: dict[str, list[AdditionalContext]] = {}
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

            grouped.setdefault(item.category, []).append(item)

        for category, envelopes in grouped.items():
            schema = jsonschemas_by_category.get(category)
            constraint = version_constraints_by_category.get(category)
            processed_additional_context[category] = self._resolve_envelope_content(
                category, envelopes, schema, constraint
            )

        return processed_additional_context

    def _resume_command(self, goal: str) -> Command:
        # `context.inputs` is populated once, at workflow START
        # (`get_workflow_state`). Re-process the additional context sent with
        # this turn so per-turn inputs (e.g. `plan_context.plan_enabled` from the
        # Duo CLI plan/build picker) refresh the flow state on resume. `context`
        # uses a deep-merge reducer, so this updates only the inputs that changed
        # and leaves the rest of the context intact.
        state_update: dict[str, Any] = {}
        refreshed_inputs = self._process_additional_context(
            self._additional_context or []
        )
        if refreshed_inputs:
            state_update["context"] = {"inputs": refreshed_inputs}

        event = FlowEvent(event_type=FlowEventType.RESPONSE, message=goal)
        if not self._approval or self._approval.WhichOneof("user_decision") is None:
            # Handle case where approval is None
            return Command(resume=event, update=state_update or None)

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
                            message_id=f"user-{uuid4()!s}",
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
            state_update["ui_chat_log"] = ui_chat_log_update
        return Command(resume=event, update=state_update or None)

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
                "environment": self._config.environment,
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
    async def _handle_compile_and_run_exception(
        self,
        e: BaseException,
        compiled_graph: Any,
        graph_config: Any,
    ) -> NoReturn:
        if isinstance(e, GraphRecursionError):
            e = NotifiableAgentException(
                "The workflow reached its maximum step limit and could not complete. "
                "Please try again with a more focused goal, or break the task into smaller steps.",
                internal_detail=f"GraphRecursionError: recursion limit of {self._recursion_limit()} steps exceeded.",
            )
        await super()._handle_compile_and_run_exception(e, compiled_graph, graph_config)

    @override
    async def _handle_workflow_failure(
        self, error: BaseException, compiled_graph: Any, graph_config: Any
    ):
        log_extra: dict[str, Any] = {
            "workflow_id": self._workflow_id,
            "source": __name__,
        }
        if isinstance(error, NotifiableAgentException) and error.internal_detail:
            log_extra["internal_detail"] = error.internal_detail

        log_exception(error, extra=log_extra)

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
                error=error,
            )
