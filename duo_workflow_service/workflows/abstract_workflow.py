# pylint: disable=direct-environment-variable-reference,unknown-option-value,too-many-instance-attributes,dangerous-default-value
import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TypedDict, Union

import structlog
from dependency_injector.wiring import Provide, inject
from gitlab_cloud_connector import CloudConnectorUser
from langchain.tools import BaseTool
from langchain_core.runnables import RunnableConfig

# pylint disable are going to be fixed via
# https://gitlab.com/gitlab-org/duo-workflow/duo-workflow-service/-/issues/78
from langgraph.checkpoint.base import (  # pylint: disable=no-langgraph-langchain-imports
    BaseCheckpointSaver,
    CheckpointTuple,
)
from langgraph.types import Command
from langsmith import traceable, tracing_context

from ai_gateway.code_suggestions.language_server import LanguageServerVersion
from ai_gateway.container import ContainerApplication
from ai_gateway.models import KindAnthropicModel
from ai_gateway.prompts import InMemoryPromptRegistry
from ai_gateway.prompts.registry import LocalPromptRegistry
from contract import contract_pb2
from duo_workflow_service.checkpointer.gitlab_workflow import GitLabWorkflow
from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    SUCCESSFUL_WORKFLOW_EXECUTION_STATUSES,
    WorkflowStatusEventEnum,
)
from duo_workflow_service.checkpointer.notifier import UserInterface
from duo_workflow_service.components import ToolsRegistry
from duo_workflow_service.entities import DuoWorkflowStateType, WorkflowStatusEnum
from duo_workflow_service.executor.outbox import Outbox, OutboxSignal
from duo_workflow_service.gitlab.events import get_event
from duo_workflow_service.gitlab.gitlab_api import (
    Namespace,
    Project,
    WorkflowConfig,
    empty_workflow_config,
    fetch_workflow_and_container_data,
)
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.gitlab.http_client_factory import get_http_client
from duo_workflow_service.gitlab.url_parser import SESSION_URL_PATH
from duo_workflow_service.llm_factory import AnthropicConfig, VertexConfig
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.tools import convert_mcp_tools_to_langchain_tool_classes
from duo_workflow_service.tracking import log_exception
from duo_workflow_service.tracking.llm_usage_context import (
    clear_workflow_checkpointer,
    set_workflow_checkpointer,
)
from duo_workflow_service.workflows.type_definitions import (
    AIO_CANCEL_STOP_WORKFLOW_REQUEST,
    AdditionalContext,
)
from lib.internal_events import InternalEventAdditionalProperties, InternalEventsClient
from lib.internal_events.event_enum import CategoryEnum, EventEnum

# Constants
QUEUE_MAX_SIZE = 1
STREAMING_QUEUE_MAX_SIZE = 10
MAX_TOKENS_TO_SAMPLE = 8192
RECURSION_LIMIT = 300
DEBUG = os.getenv("DEBUG")
MAX_MESSAGES_TO_DISPLAY = 5


class InvocationMetadata(TypedDict):
    base_url: str
    gitlab_token: str


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
    _workflow_type: CategoryEnum
    _stream: bool = False
    _additional_context: list[AdditionalContext] | None
    _mcp_tools: list[type[BaseTool]]
    _approval: Optional[contract_pb2.Approval]
    _prompt_registry: InMemoryPromptRegistry | LocalPromptRegistry
    _language_server_version: Optional[LanguageServerVersion]

    @inject
    def __init__(
        self,
        workflow_id: str,
        workflow_metadata: Dict[str, Any],
        workflow_type: CategoryEnum,
        invocation_metadata: InvocationMetadata = {
            "base_url": "",
            "gitlab_token": "",
        },
        mcp_tools: list[contract_pb2.McpTool] = [],
        user: Optional[CloudConnectorUser] = None,
        additional_context: Optional[list[AdditionalContext]] = None,
        approval: Optional[contract_pb2.Approval] = None,
        prompt_registry: LocalPromptRegistry = Provide[
            ContainerApplication.pkg_prompts.prompt_registry
        ],
        internal_event_client: InternalEventsClient = Provide[
            ContainerApplication.internal_event.client
        ],
        language_server_version: Optional[LanguageServerVersion] = None,
        preapproved_tools: Optional[list[str]] = [],
    ):
        self._outbox = Outbox()
        self._workflow_id = workflow_id
        self._workflow_metadata = workflow_metadata
        self._user = user
        self.log = structlog.stdlib.get_logger("workflow").bind(workflow_id=workflow_id)
        self._http_client = get_http_client(
            self._outbox,
            invocation_metadata.get("base_url", ""),
            invocation_metadata.get("gitlab_token", ""),
        )
        self._workflow_type = workflow_type
        self._additional_context = additional_context
        self._mcp_tools = convert_mcp_tools_to_langchain_tool_classes(
            mcp_tools=mcp_tools
        )
        self._model_config = self._get_model_config()
        self._approval = approval
        self._prompt_registry = prompt_registry
        self._workflow_config = empty_workflow_config()
        self._internal_event_client = internal_event_client
        self._language_server_version = language_server_version
        self._preapproved_tools = preapproved_tools
        self._session_url: Optional[str] = None
        self._last_gitlab_status: WorkflowStatusEventEnum | None = None

    async def run(self, goal: str) -> None:
        with duo_workflow_metrics.time_workflow(
            workflow_type=self._workflow_type.value
        ):
            extended_logging = self._workflow_metadata.get("extended_logging", False)
            tracing_metadata = {
                "git_url": self._workflow_metadata.get("git_url", ""),
                "git_sha": self._workflow_metadata.get("git_sha", ""),
                "workflow_type": self._workflow_type.value,
            }

            # By default, tracing follows extended_logging. Only disable if LANGSMITH_TRACING_V2 is explicitly "false"
            langsmith_tracing_v2_env = os.getenv("LANGSMITH_TRACING_V2", "").lower()
            tracing_enabled = extended_logging and (langsmith_tracing_v2_env != "false")

            with tracing_context(enabled=tracing_enabled):
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

    def set_action_response(self, event: contract_pb2.ClientEvent):
        self._outbox.set_action_response(event)

    def _recursion_limit(self):
        return RECURSION_LIMIT

    @traceable
    async def _compile_and_run_graph(self, goal: str) -> None:
        graph_config: RunnableConfig = {
            "recursion_limit": self._recursion_limit(),
            "configurable": {"thread_id": self._workflow_id},
        }
        compiled_graph = None
        try:
            self._project, self._namespace, self._workflow_config = (
                await fetch_workflow_and_container_data(
                    client=self._http_client,
                    workflow_id=self._workflow_id,
                )
            )

            if self._project and self._project.get("web_url"):
                self._session_url = (
                    f"{self._project['web_url']}{SESSION_URL_PATH}{self._workflow_id}"
                )

            if self._namespace and self._support_namespace_level_workflow() is False:
                raise NotImplementedError(
                    f"This workflow {self._workflow_type.value} does not support namespace-level workflow"
                )

            user_for_registry = (
                self._user
                if self._workflow_type == CategoryEnum.WORKFLOW_CHAT
                else None
            )

            tools_registry = await ToolsRegistry.configure(
                outbox=self._outbox,
                workflow_config=self._workflow_config,
                gl_http_client=self._http_client,
                project=self._project,
                mcp_tools=(
                    self._mcp_tools
                    if self._workflow_config.get("mcp_enabled", False)
                    else []
                ),
                user=user_for_registry,
                language_server_version=self._language_server_version,
            )
            checkpoint_notifier = UserInterface(outbox=self._outbox, goal=goal)

            def on_gitlab_status_update(status: WorkflowStatusEventEnum):
                self._last_gitlab_status = status

            async with GitLabWorkflow(
                self._http_client,
                self._workflow_id,
                self._workflow_type,
                self._workflow_config,
                gitlab_status_update_callback=on_gitlab_status_update,
            ) as checkpointer:
                set_workflow_checkpointer(checkpointer)
                status_event = getattr(checkpointer, "initial_status_event", None)
                checkpoint_tuple = await checkpointer.aget_tuple(graph_config)
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
                    if type == "updates":
                        for step in state:
                            self.log.info(f"step: {step}")
                    else:
                        await checkpoint_notifier.send_event(
                            type=type, state=state, stream=self._stream
                        )
        except BaseException as e:
            self.last_error = e
            if str(e) == AIO_CANCEL_STOP_WORKFLOW_REQUEST:
                # when workflow is cancelled with AIO_CANCEL_STOP_WORKFLOW_REQUEST, a new checkpoint is not created and
                # internal workflow state is not updated, thus the clients don't receive a newCheckpoint notification
                # here we send a notification with stopped status for clients to react accordingly
                await checkpoint_notifier.send_event(
                    type="values",
                    state={"status": WorkflowStatusEnum.CANCELLED, "ui_chat_log": []},
                    stream=self._stream,
                )
            else:
                await self._handle_workflow_failure(e, compiled_graph, graph_config)
            raise TraceableException(e)
        finally:
            clear_workflow_checkpointer()
            self.is_done = True

    async def get_graph_input(
        self, goal: str, status_event: str, checkpoint_tuple: Optional[CheckpointTuple]
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
        category: CategoryEnum,
    ):
        self.log.info("Tracking Internal event %s", event_name.value)
        self._internal_event_client.track_event(
            event_name=event_name.value,
            additional_properties=additional_properties,
            category=category.value if category else self.__class__.__name__,
        )

    def _get_model_config(self) -> Union[AnthropicConfig, VertexConfig]:
        """Determine the appropriate model configuration based on deployment environment.

        This method creates the appropriate configuration object for either
        Vertex AI or standard Anthropic API deployments. It automatically
        detects the deployment environment and returns the corresponding
        configuration with the appropriate model.

        The method checks for the presence of AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT
        environment variable to determine if running on Google Cloud Vertex AI.

        Returns:
            Union[AnthropicConfig, VertexConfig]: The configuration object for
                the current deployment environment:
                - VertexConfig: When running on Google Cloud Vertex AI
                - AnthropicConfig: When using Anthropic API directly

        Note:
            Subclasses can override this method to implement custom model selection
            logic or to use different model versions.
        """
        # Check for Vertex AI project configuration
        _vertex_project_id = os.getenv("AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT")
        if bool(_vertex_project_id and len(_vertex_project_id) > 1):
            return VertexConfig(
                model_name=KindAnthropicModel.CLAUDE_SONNET_4_VERTEX.value
            )

        return AnthropicConfig(model_name=KindAnthropicModel.CLAUDE_SONNET_4.value)

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
