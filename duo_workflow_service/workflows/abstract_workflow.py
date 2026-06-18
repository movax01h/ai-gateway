# pylint: disable=direct-environment-variable-reference,unknown-option-value,too-many-instance-attributes,dangerous-default-value
import asyncio
import os
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, NoReturn, Optional
from uuid import uuid4

import structlog
from dependency_injector.wiring import Provide, inject
from gitlab_cloud_connector import CloudConnectorUser
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

# pylint disable are going to be fixed via
# https://gitlab.com/gitlab-org/duo-workflow/duo-workflow-service/-/issues/78
from langgraph.checkpoint.base import (  # pylint: disable=no-langgraph-langchain-imports
    BaseCheckpointSaver,
)
from langgraph.types import Command
from langsmith import traceable, tracing_context
from pydantic import BaseModel, ConfigDict

from ai_gateway.container import ContainerApplication
from ai_gateway.prompts import BasePromptRegistry
from contract import contract_pb2
from duo_workflow_service.agent_platform.utils.exceptions import (
    NotifiableAgentException,
)
from duo_workflow_service.audit_events.callback_handler import AuditEventCallbackHandler
from duo_workflow_service.audit_events.client import AuditEventClient
from duo_workflow_service.audit_events.collector import AuditEventCollector
from duo_workflow_service.audit_events.context import audit_collector_context
from duo_workflow_service.audit_events.event_types import SessionStartedEvent
from duo_workflow_service.checkpointer.gitlab_workflow import GitLabWorkflow
from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    SUCCESSFUL_WORKFLOW_EXECUTION_STATUSES,
    WorkflowStatusEventEnum,
)
from duo_workflow_service.checkpointer.notifier import UserInterface
from duo_workflow_service.components import ToolsRegistry
from duo_workflow_service.entities import DuoWorkflowStateType, WorkflowStatusEnum
from duo_workflow_service.entities.state import MessageTypeEnum, ToolStatus, UiChatLog
from duo_workflow_service.errors.typing import (
    GENERIC_WORKFLOW_ERROR_MESSAGE,
    NamespaceLevelWorkflowNotSupportedException,
    NotifiableException,
)
from duo_workflow_service.executor.outbox import Outbox, OutboxSignal
from duo_workflow_service.gitlab.events import get_event
from duo_workflow_service.gitlab.executor_http_client import ExecutorGitLabHttpClient
from duo_workflow_service.gitlab.gitlab_api import (
    Namespace,
    Project,
    WorkflowConfig,
    empty_workflow_config,
    fetch_workflow_and_container_data,
)
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.gitlab.schema import PromptInjectionProtectionLevel
from duo_workflow_service.gitlab.url_parser import SESSION_URL_PATH
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.tools.mcp_tools import (
    McpToolConfig,
    convert_mcp_tools_to_configs,
)
from duo_workflow_service.tracking import (
    MonitoringContext,
    current_monitoring_context,
    log_exception,
)
from duo_workflow_service.workflows.type_definitions import (
    AIO_CANCEL_STOP_WORKFLOW_REQUEST,
    AdditionalContext,
)
from lib.events import GLReportingEventContext
from lib.feature_flags.context import FeatureFlag, is_feature_enabled
from lib.internal_events import InternalEventAdditionalProperties, InternalEventsClient
from lib.internal_events.event_enum import EventEnum
from lib.langsmith_tracing import get_langsmith_trace_headers
from lib.language_server import LanguageServerVersion

# Constants
QUEUE_MAX_SIZE = 1
STREAMING_QUEUE_MAX_SIZE = 10
MAX_TOKENS_TO_SAMPLE = 8192
RECURSION_LIMIT = 300
DEBUG = os.getenv("DEBUG")
MAX_MESSAGES_TO_DISPLAY = 5


class ToolAccessPolicies(BaseModel):
    model_config = ConfigDict(extra="ignore")

    allow: list[str] = []
    deny: list[str] = []


class AbstractWorkflow(ABC):
    """Abstract base class for workflow implementations.

    Provides a structure for creating workflow classes with common functionality.
    """

    _outbox: Outbox
    _workflow_id: str
    _project: Project | None
    _namespace: Namespace | None
    _workflow_config: WorkflowConfig
    _http_client: GitlabHttpClient
    _workflow_metadata: dict[str, Any]
    is_done: bool = False
    last_error: BaseException | None = None
    checkpoint_notifier: Optional[UserInterface] = None
    _workflow_type: GLReportingEventContext
    _stream: bool = False
    _additional_context: list[AdditionalContext] | None
    _mcp_tools: list[McpToolConfig]
    _approval: Optional[contract_pb2.Approval]
    _prompt_registry: BasePromptRegistry
    _language_server_version: Optional[LanguageServerVersion]

    @inject
    def __init__(
        self,
        workflow_id: str,
        workflow_metadata: Dict[str, Any],
        workflow_type: GLReportingEventContext,
        user: CloudConnectorUser,
        mcp_tools: list[contract_pb2.McpTool] = [],
        additional_context: Optional[list[AdditionalContext]] = None,
        approval: Optional[contract_pb2.Approval] = None,
        prompt_registry: BasePromptRegistry = Provide[
            ContainerApplication.pkg_prompts.prompt_registry
        ],
        internal_event_client: InternalEventsClient = Provide[
            ContainerApplication.internal_event.client
        ],
        language_server_version: Optional[LanguageServerVersion] = None,
        preapproved_tools: Optional[list[str]] = [],
        audit_event_enabled: bool = Provide[
            ContainerApplication.audit_event.config.enabled  # type: ignore[attr-defined]
        ],
        audit_event_buffer_size: int = Provide[
            ContainerApplication.audit_event.config.buffer_size  # type: ignore[attr-defined]
        ],
        audit_event_flush_interval: float = Provide[
            ContainerApplication.audit_event.config.flush_interval_seconds  # type: ignore[attr-defined]
        ],
        audit_event_max_retries: int = Provide[
            ContainerApplication.audit_event.config.max_retries  # type: ignore[attr-defined]
        ],
    ):
        self._outbox = Outbox()
        self._workflow_id = workflow_id
        self._workflow_metadata = workflow_metadata
        self._user = user
        self.log = structlog.stdlib.get_logger("workflow").bind(workflow_id=workflow_id)
        self._http_client = ExecutorGitLabHttpClient(self._outbox)
        self._workflow_type = workflow_type
        self._additional_context = additional_context
        self._mcp_tools = convert_mcp_tools_to_configs(mcp_tools=mcp_tools)
        self._approval = approval
        self._prompt_registry = prompt_registry
        self._workflow_config = empty_workflow_config()
        self._internal_event_client = internal_event_client
        self._language_server_version = language_server_version
        self._preapproved_tools = preapproved_tools
        self._denied_tools: list[str] = []
        self._session_url: Optional[str] = None
        self._last_gitlab_status: WorkflowStatusEventEnum | None = None
        self._first_response_metric_recorded = False
        self._audit_event_enabled = audit_event_enabled
        self._audit_event_buffer_size = audit_event_buffer_size
        self._audit_event_flush_interval = audit_event_flush_interval
        self._audit_event_max_retries = audit_event_max_retries

    async def run(self, goal: str) -> None:
        with duo_workflow_metrics.time_workflow(
            workflow_type=self._workflow_type.value
        ):
            extended_logging = self._workflow_metadata.get("extended_logging", False)
            monitoring_context: MonitoringContext = current_monitoring_context.get()

            tracing_metadata = {
                "git_url": self._workflow_metadata.get("git_url", ""),
                "git_sha": self._workflow_metadata.get("git_sha", ""),
                "workflow_type": self._workflow_type.value,
                "thread_id": self._workflow_id,
                # Flow versioning identifiers, populated for registry/inline flows only
                # (legacy flows leave these unset, so they are omitted).
                **monitoring_context.flow_versioning_fields(),
            }

            # By default, tracing follows extended_logging. Only disable if LANGSMITH_TRACING_V2 is explicitly "false"
            langsmith_tracing_v2_env = os.getenv("LANGSMITH_TRACING_V2", "").lower()
            tracing_enabled = extended_logging and (langsmith_tracing_v2_env != "false")

            monitoring_context.tracing_enabled = str(tracing_enabled)
            monitoring_context.use_ai_prompt_scanning = is_feature_enabled(
                FeatureFlag.AI_PROMPT_SCANNING
            )

            # Setup langsmith parent tracing headers if any
            parent_trace = get_langsmith_trace_headers()

            with tracing_context(parent=parent_trace, enabled=tracing_enabled):
                try:
                    # pylint: disable=unexpected-keyword-arg
                    await self._compile_and_run_graph(
                        goal=goal,
                        langsmith_extra={"metadata": tracing_metadata},
                    )
                except TraceableException:
                    # Intentionally suppressing the exception here after it has been
                    # properly traced in Langsmith via the TraceableException
                    pass
                finally:
                    self._outbox.close()

    @abstractmethod
    async def _handle_workflow_failure(
        self, error: BaseException, compiled_graph, graph_config
    ):
        pass

    @abstractmethod
    def _compile(
        self,
        goal: str,
        tools_registry: ToolsRegistry,
        checkpointer: BaseCheckpointSaver,
    ) -> Any:
        pass

    @property
    def last_gitlab_status(self) -> WorkflowStatusEventEnum | None:
        return self._last_gitlab_status

    def successful_execution(self) -> bool:
        """Return if the workflow task execution was successful."""
        if self.last_error:
            return False

        return self._last_gitlab_status in SUCCESSFUL_WORKFLOW_EXECUTION_STATUSES

    async def get_from_outbox(self) -> contract_pb2.Action | OutboxSignal:
        return await self._outbox.get()

    def fail_outbox_action(self, request_id: str, message: str):
        self._outbox.fail_action(request_id=request_id, message=message)

    def set_action_response(self, event: contract_pb2.ClientEvent):
        self._outbox.set_action_response(event)

    def _recursion_limit(self):
        return RECURSION_LIMIT

    def _record_first_response_metric(self):
        """Record the time to first response ready metric."""
        if not self._first_response_metric_recorded:
            duo_workflow_metrics.record_time_to_first_response(
                workflow_type=self._workflow_type.value,
            )
            self._first_response_metric_recorded = True

    def _merge_jwt_governance_claims(self) -> None:
        if self._user.claims and self._user.claims.extra:
            raw = self._user.claims.extra.get("tool_access_policies")
            if raw:
                try:
                    policies = (
                        ToolAccessPolicies.model_validate_json(raw)
                        if isinstance(raw, str)
                        else ToolAccessPolicies.model_validate(raw)
                    )
                except Exception:
                    return
                if policies.allow:
                    existing = self._preapproved_tools or []
                    self._preapproved_tools = list(set(existing) | set(policies.allow))
                if policies.deny:
                    self._denied_tools = list(
                        set(self._denied_tools) | set(policies.deny)
                    )

    def _extract_trace_output(self, state: dict | None) -> str | None:
        """Extract the final response content from ui_chat_log for LangSmith tracing.

        Args:
            state: The final state dictionary from the graph execution

        Returns:
            The content of the last ui_chat_log entry, or None if not available
        """
        if not state:
            return None

        ui_chat_log = state.get("ui_chat_log", [])
        if not ui_chat_log:
            return None

        return ui_chat_log[-1].get("content")

    async def _init_audit_events(self, goal: str) -> Optional[AuditEventCollector]:
        if not self._audit_event_enabled:
            return None

        audit_client = AuditEventClient(
            http_client=self._http_client,
            workflow_id=self._workflow_id,
            max_retries=self._audit_event_max_retries,
        )
        audit_collector = AuditEventCollector(
            client=audit_client,
            buffer_size=self._audit_event_buffer_size,
            flush_interval_seconds=self._audit_event_flush_interval,
        )
        await audit_collector.start()
        audit_collector_context.set(audit_collector)
        audit_collector.capture(
            SessionStartedEvent(
                workflow_id=self._workflow_id,
                workflow_type=self._workflow_type.value,
                goal=goal,
            )
        )
        return audit_collector

    async def _handle_non_values_stream_event(
        self, event_type: str, state: Any
    ) -> None:
        if event_type == "updates":
            for step in state:
                self.log.info(f"step: {step}")
        else:
            assert self.checkpoint_notifier is not None
            await self.checkpoint_notifier.send_event(
                type=event_type, state=state, stream=self._stream
            )

    @traceable
    async def _compile_and_run_graph(self, goal: str) -> str | None:
        audit_collector = await self._init_audit_events(goal)

        callbacks: list[BaseCallbackHandler] = (
            [
                AuditEventCallbackHandler(
                    collector=audit_collector, workflow_id=self._workflow_id
                )
            ]
            if audit_collector
            else []
        )
        graph_config: RunnableConfig = {
            "recursion_limit": self._recursion_limit(),
            "configurable": {"thread_id": self._workflow_id},
            "callbacks": callbacks,
        }
        last_state = None
        compiled_graph = None
        self.checkpoint_notifier = UserInterface(outbox=self._outbox, goal=goal)

        try:
            (
                self._project,
                self._namespace,
                self._workflow_config,
            ) = await fetch_workflow_and_container_data(
                client=self._http_client,
                workflow_id=self._workflow_id,
            )

            self._merge_jwt_governance_claims()

            # Update monitoring context with prompt injection protection level
            monitoring_context = current_monitoring_context.get()
            monitoring_context.prompt_injection_protection_level = (
                self._workflow_config.get(
                    "prompt_injection_protection_level",
                    PromptInjectionProtectionLevel.LOG_ONLY,
                )
            )

            if self._project and self._project.get("web_url"):
                self._session_url = (
                    f"{self._project['web_url']}{SESSION_URL_PATH}{self._workflow_id}"
                )

            if self._namespace and self._support_namespace_level_workflow() is False:
                raise NamespaceLevelWorkflowNotSupportedException()

            tools_registry = await ToolsRegistry.configure(
                outbox=self._outbox,
                workflow_config=self._workflow_config,
                gl_http_client=self._http_client,
                project=self._project,
                workflow_id=self._workflow_id,
                mcp_tools=(
                    self._mcp_tools
                    if self._workflow_config.get("mcp_enabled", False)
                    else []
                ),
                language_server_version=self._language_server_version,
                denied_tools=self._denied_tools,
            )

            def on_gitlab_status_update(status: WorkflowStatusEventEnum):
                self._last_gitlab_status = status

            async with GitLabWorkflow(
                self._http_client,
                self._workflow_id,
                self._workflow_type,
                self._workflow_config,
                gitlab_status_update_callback=on_gitlab_status_update,
            ) as checkpointer:
                status_event = getattr(checkpointer, "initial_status_event", None)
                checkpoint_tuple = (
                    self._workflow_config.get("latest_checkpoint", None)
                    or self._workflow_config["first_checkpoint"]
                )
                if not status_event:
                    status_event = (
                        "" if checkpoint_tuple else WorkflowStatusEventEnum.START
                    )

                # Compile is CPU-bound process hence we're using a thread to avoid interrupting the gRPC server.
                # See https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/1468
                # for more info.
                compiled_graph = await asyncio.to_thread(
                    self._compile, goal, tools_registry, checkpointer
                )
                graph_input = await self.get_graph_input(
                    goal, status_event, checkpoint_tuple
                )

                async for type, state in compiled_graph.astream(
                    input=graph_input,
                    config=graph_config,
                    stream_mode=["values", "messages", "updates"],
                ):
                    if type == "values":
                        self._record_first_response_metric()
                        last_state = state
                        assert self.checkpoint_notifier is not None
                        await self.checkpoint_notifier.send_event(
                            type=type, state=state, stream=self._stream
                        )
                    else:
                        await self._handle_non_values_stream_event(type, state)

                return self._extract_trace_output(last_state)
        except BaseException as e:
            await self._handle_compile_and_run_exception(
                e, compiled_graph, graph_config
            )
        finally:
            self.is_done = True
            if audit_collector:
                await audit_collector.close()

    async def _handle_compile_and_run_exception(
        self,
        e: BaseException,
        compiled_graph,
        graph_config: RunnableConfig,
    ) -> NoReturn:
        is_notifiable = isinstance(e, NotifiableException)
        is_notifiable_agent = isinstance(e, NotifiableAgentException)
        is_cancel = str(e) == AIO_CANCEL_STOP_WORKFLOW_REQUEST

        self.last_error = e.__cause__ if (is_notifiable or is_notifiable_agent) else e

        # NotifiableException is chat-specific and carries its own UI semantics; the
        # send_event below surfaces str(e) directly so subclass _handle_workflow_failure
        # is skipped to avoid duplicating UI side-effects. NotifiableAgentException, on
        # the other hand, is delegated to _handle_workflow_failure so the subclass can
        # log internal_detail server-side and persist the safe ui_message to the UI
        # chat log via aupdate_state.
        # This needs to come before the send_event so we don't mess up the ui_chat_log.
        if not is_notifiable and not is_cancel:
            await self._handle_workflow_failure(e, compiled_graph, graph_config)

        if is_cancel:
            # when workflow is cancelled with AIO_CANCEL_STOP_WORKFLOW_REQUEST, a new checkpoint is not created and
            # internal workflow state is not updated, thus the clients don't receive a newCheckpoint notification
            # here we send a notification with stopped status for clients to react accordingly
            status = WorkflowStatusEnum.CANCELLED
            ui_chat_log: list[UiChatLog] = []
        else:
            status = WorkflowStatusEnum.ERROR
            # Only surface the explicit ui_message from NotifiableAgentException or the
            # str() of a NotifiableException; all other exceptions fall back to the generic
            # message to avoid leaking internal details (stack traces, tokens, infra hints)
            # to end users.
            if isinstance(e, NotifiableAgentException):
                content = e.ui_message
            elif isinstance(e, NotifiableException):
                content = str(e)
            else:
                content = GENERIC_WORKFLOW_ERROR_MESSAGE
            ui_chat_log = [
                UiChatLog(
                    message_type=MessageTypeEnum.AGENT,
                    message_sub_type=None,
                    content=content,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    status=ToolStatus.FAILURE,
                    correlation_id=None,
                    tool_info=None,
                    additional_context=None,
                    message_id=f"error-{str(uuid4())}",
                )
            ]

        assert self.checkpoint_notifier is not None
        await self.checkpoint_notifier.send_event(
            type="values",
            state={"status": status, "ui_chat_log": ui_chat_log},
            stream=self._stream,
        )

        raise TraceableException(e)

    async def get_graph_input(
        self, goal: str, status_event: str, checkpoint_tuple: Any
    ) -> Any:
        match status_event:
            case WorkflowStatusEventEnum.START:
                return self.get_workflow_state(goal)
            case WorkflowStatusEventEnum.RESUME:
                event = await get_event(self._http_client, self._workflow_id)
                if not event:
                    return None
                return Command(resume=event)
            case WorkflowStatusEventEnum.RETRY:
                if checkpoint_tuple is None:
                    return self.get_workflow_state(
                        goal
                    )  # no saved checkpoints from last run
                return None  # retry from last checkpoint
            case _:
                return None

    @abstractmethod
    def get_workflow_state(self, goal: str) -> DuoWorkflowStateType:
        pass

    async def cleanup(self, workflow_id: str):
        try:
            self.is_done = True

            self._outbox.check_empty()

            self.log.info("Workflow cleanup completed.")
        except BaseException as cleanup_err:
            log_exception(
                cleanup_err,
                extra={
                    "workflow_id": workflow_id,
                    "context": "Workflow cleanup failed",
                },
            )
            raise

    def _track_internal_event(
        self,
        event_name: EventEnum,
        additional_properties: InternalEventAdditionalProperties,
        category: GLReportingEventContext,
    ):
        self.log.info("Tracking Internal event %s", event_name.value)
        self._internal_event_client.track_event(
            event_name=event_name.value,
            additional_properties=additional_properties,
            category=category.value if category else self.__class__.__name__,
        )

    def _support_namespace_level_workflow(self) -> bool:
        """Indicate if a workflow class supports namespace-level workflows.

        To support namespace-level workflows, make sure that the subclass of AbstractWorkflow
        handle both self._project and self._namespace fields properly, then override this method to return `True`.
        By default, namespace support is disabled in workflow classes.
        """
        return False


TypeWorkflow = type[AbstractWorkflow]


class TraceableException(Exception):
    def __init__(self, original_exception: BaseException):
        self.original_exception = original_exception
        super().__init__(str(original_exception))

    def __repr__(self):
        return f"<TraceableException wrapping {repr(self.original_exception)}>"
