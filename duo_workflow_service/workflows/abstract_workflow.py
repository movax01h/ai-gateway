import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any, Dict

import structlog
from langchain_core.runnables import RunnableConfig

# pylint disable are going to be fixed via
# https://gitlab.com/gitlab-org/duo-workflow/duo-workflow-service/-/issues/78
from langgraph.checkpoint.base import (  # pylint: disable=no-langgraph-langchain-imports
    BaseCheckpointSaver,
)
from langgraph.types import Command
from langsmith import traceable, tracing_context

from contract import contract_pb2
from duo_workflow_service.checkpointer.gitlab_workflow import (
    GitLabWorkflow,
    WorkflowStatusEventEnum,
)
from duo_workflow_service.components import ToolsRegistry
from duo_workflow_service.entities import DuoWorkflowStateType
from duo_workflow_service.gitlab.events import get_event
from duo_workflow_service.gitlab.gitlab_project import (
    Project,
    fetch_project_data_with_workflow_id,
)
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.gitlab.url_parser import GitLabUrlParser
from duo_workflow_service.internal_events import (
    DuoWorkflowInternalEvent,
    EventContext,
    InternalEventAdditionalProperties,
    current_event_context,
)
from duo_workflow_service.internal_events.event_enum import EventEnum
from duo_workflow_service.tracking import log_exception

# Constants
QUEUE_MAX_SIZE = 1
MAX_TOKENS_TO_SAMPLE = 4096
RECURSION_LIMIT = 300
DEBUG = os.getenv("DEBUG")
MAX_MESSAGES_TO_DISPLAY = 5
MAX_MESSAGE_LENGTH = 200


class AbstractWorkflow(ABC):
    """
    Abstract base class for workflow implementations.
    Provides a structure for creating workflow classes with common functionality.
    """

    _outbox: asyncio.Queue
    _inbox: asyncio.Queue
    _workflow_id: str
    _project: Project
    _http_client: GitlabHttpClient
    _workflow_metadata: dict[str, Any]
    is_done: bool = False

    def __init__(self, workflow_id: str, workflow_metadata: Dict[str, Any]):
        self._outbox = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
        self._inbox = asyncio.Queue(maxsize=QUEUE_MAX_SIZE)
        self._workflow_id = workflow_id
        self._workflow_metadata = workflow_metadata
        self.log = structlog.stdlib.get_logger("workflow").bind(workflow_id=workflow_id)
        self._http_client = GitlabHttpClient(
            {"outbox": self._outbox, "inbox": self._inbox}
        )

    async def run(self, goal: str) -> None:
        extended_logging = self._workflow_metadata.get("extended_logging", False)
        tracing_metadata = {
            "git_url": self._workflow_metadata.get("git_url", ""),
            "git_sha": self._workflow_metadata.get("git_sha", ""),
        }

        with tracing_context(enabled=extended_logging):
            # pylint: disable=unexpected-keyword-arg
            await self._compile_and_run_graph(
                goal=goal,
                langsmith_extra={"metadata": tracing_metadata},
            )

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

    def get_from_outbox(self):
        item = self._outbox.get_nowait()
        self._outbox.task_done()
        return item

    def add_to_inbox(self, event: contract_pb2.ClientEvent):
        self._inbox.put_nowait(event)

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
            # Fetch GitLab project information to inject into prompts
            self._project = await fetch_project_data_with_workflow_id(
                client=self._http_client,
                workflow_id=self._workflow_id,
            )

            if self._project:
                context: EventContext = current_event_context.get()
                context.project_id = self._project.get("id")
                context.namespace_id = self._project.get("namespace", {}).get("id")

            if "web_url" not in self._project:
                raise RuntimeError(
                    f"Failed to get web_url from project for workflow {self._workflow_id}"
                )

            gitlab_host = GitLabUrlParser.extract_host_from_url(
                self._project["web_url"]
            )

            if not gitlab_host:
                raise RuntimeError(
                    f"Failed to extract gitlab host from web_url for workflow {self._workflow_id}"
                )

            tools_registry = await ToolsRegistry.configure(
                outbox=self._outbox,
                inbox=self._inbox,
                workflow_id=self._workflow_id,
                gl_http_client=self._http_client,
                gitlab_host=gitlab_host,
            )

            async with GitLabWorkflow(
                self._http_client, self._workflow_id
            ) as checkpointer:
                status_event = getattr(checkpointer, "initial_status_event", None)
                if not status_event:
                    checkpoint_tuple = await checkpointer.aget_tuple(graph_config)
                    status_event = (
                        "" if checkpoint_tuple else WorkflowStatusEventEnum.START
                    )

                compiled_graph = self._compile(goal, tools_registry, checkpointer)
                graph_input = await self.get_graph_input(goal, status_event)

                async for steps in compiled_graph.astream(
                    input=graph_input,
                    config=graph_config,
                ):
                    for step in steps:
                        self.log.info(f"step: {step}")
                        element = steps[step]
                        self.log_workflow_elements(element)

        except BaseException as e:
            await self._handle_workflow_failure(e, compiled_graph, graph_config)
        finally:
            self.is_done = True

    async def get_graph_input(self, goal: str, status_event: str) -> Any:
        match status_event:
            case WorkflowStatusEventEnum.START:
                return self.get_workflow_state(goal)
            case WorkflowStatusEventEnum.RESUME:
                event = await get_event(self._http_client, self._workflow_id)
                if not event:
                    return None
                return Command(resume=event)
            case _:
                return None

    @abstractmethod
    def get_workflow_state(self, goal: str) -> DuoWorkflowStateType:
        pass

    @abstractmethod
    def log_workflow_elements(self, element):
        pass

    async def cleanup(self, workflow_id: str):
        try:
            self.is_done = True

            def drain_queue(queue, queue_name: str):
                while queue.qsize() > 0:
                    msg = queue.get_nowait()
                    queue.task_done()
                    content = str(msg)

                    if len(content) > MAX_MESSAGE_LENGTH:
                        content = f"{content[:MAX_MESSAGE_LENGTH]}..."

                    self.log.info(
                        f"Drained {queue_name} message during cleanup",
                        workflow_id=workflow_id,
                        content=content,
                    )

            drain_queue(self._outbox, "outbox")
            drain_queue(self._inbox, "inbox")

            self.log.info("Workflow cleanup completed.")
        except BaseException as cleanup_err:
            log_exception(
                cleanup_err,
                extra={
                    "workflow_id": workflow_id,
                    "context": "Workflow cleanup failed",
                },
            )

    def _track_internal_event(
        self,
        event_name: EventEnum,
        additional_properties: InternalEventAdditionalProperties,
    ):
        self.log.info("Tracking Internal event %s", event_name.value)
        DuoWorkflowInternalEvent.track_event(
            event_name=event_name.value,
            additional_properties=additional_properties,
            category=self.__class__.__name__,
        )
