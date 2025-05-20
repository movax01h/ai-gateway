from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.memory import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from duo_workflow_service.agents import Agent, HandoverAgent, RunToolNode, ToolsExecutor
from duo_workflow_service.components import ToolsRegistry
from duo_workflow_service.entities import (
    MAX_CONTEXT_TOKENS,
    MessageTypeEnum,
    Plan,
    ToolStatus,
    UiChatLog,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.internal_events.event_enum import CategoryEnum
from duo_workflow_service.llm_factory import new_chat_client
from duo_workflow_service.token_counter.approximate_token_counter import (
    ApproximateTokenCounter,
)
from duo_workflow_service.tracking import log_exception
from duo_workflow_service.workflows.abstract_workflow import (
    DEBUG,
    MAX_MESSAGE_LENGTH,
    MAX_TOKENS_TO_SAMPLE,
    RECURSION_LIMIT,
    AbstractWorkflow,
)
from duo_workflow_service.workflows.convert_to_gitlab_ci.prompts import (
    CI_PIPELINES_MANAGER_FILE_USER_MESSAGE,
    CI_PIPELINES_MANAGER_SYSTEM_MESSAGE,
    CI_PIPELINES_MANAGER_USER_GUIDELINES,
)

AGENT_NAME = "ci_pipelines_manager_agent"


# ROUTERS
class Routes(StrEnum):
    CONTINUE = "continue"
    END = "end"
    AGENT = "agent"
    COMMIT_CHANGES = "commit_changes"


def _router(state: WorkflowState) -> str:
    if state["status"] == WorkflowStatusEnum.CANCELLED:
        return Routes.END

    agent_messages = state["conversation_history"].get(AGENT_NAME, [])
    if not agent_messages or len(agent_messages) < 2:
        return Routes.END

    tool_calls = getattr(agent_messages[-2], "tool_calls", [])
    if len(tool_calls) == 0:
        return Routes.END

    if tool_calls and tool_calls[0].get("name") == "read_file":
        return Routes.AGENT

    if tool_calls[0].get("name") == "create_file_with_contents":
        return Routes.COMMIT_CHANGES

    return Routes.END


def _tools_execution_requested(state: WorkflowState) -> str:
    if state["status"] == WorkflowStatusEnum.CANCELLED:
        return Routes.END

    agent_messages = state["conversation_history"].get(AGENT_NAME, [])
    if agent_messages and getattr(agent_messages[-1], "tool_calls", []):
        return Routes.CONTINUE

    return Routes.END


def _load_file_contents(file_contents: list[str], state: WorkflowState):
    if (
        not file_contents
        or "Error running tool: unable to open file:" in file_contents[0]
    ):
        raise RuntimeError("Failed to load file contents, ensure that file is present")

    logs: list[UiChatLog] = []

    system_prompt = CI_PIPELINES_MANAGER_SYSTEM_MESSAGE
    human_guidelines = CI_PIPELINES_MANAGER_USER_GUIDELINES

    human_prompt = CI_PIPELINES_MANAGER_FILE_USER_MESSAGE.format(
        file_content=file_contents[0],
    )
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_guidelines),
        HumanMessage(content=human_prompt),
    ]

    if ApproximateTokenCounter(AGENT_NAME).count_tokens(messages) > MAX_CONTEXT_TOKENS:
        messages = []
        logs.append(
            UiChatLog(
                message_type=MessageTypeEnum.TOOL,
                content="File too large, skipping.",
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=ToolStatus.FAILURE,
                correlation_id=None,
                tool_info=None,
                context_elements=None,
            )
        )
    else:
        logs.append(
            UiChatLog(
                message_type=MessageTypeEnum.TOOL,
                content="Loaded Jenkins file",
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=ToolStatus.SUCCESS,
                correlation_id=None,
                tool_info=None,
                context_elements=None,
            )
        )

    return {
        "conversation_history": {AGENT_NAME: messages},
        "ui_chat_log": logs,
        "status": WorkflowStatusEnum.EXECUTION,
    }


def _git_output(command_output: list[str], state: WorkflowState):
    logs: list[UiChatLog] = [
        UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            content=f"{command_output[-1]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.SUCCESS,
            correlation_id=None,
            tool_info=None,
            context_elements=None,
        )
    ]

    return {
        "ui_chat_log": logs,
    }


class Workflow(AbstractWorkflow):
    async def _handle_workflow_failure(
        self, error: BaseException, compiled_graph: Any, graph_config: Any
    ):
        log_exception(error, extra={"workflow_id": self._workflow_id})

    def _recursion_limit(self):
        return RECURSION_LIMIT

    def _compile(
        self,
        goal: str,
        tools_registry: ToolsRegistry,
        checkpointer: BaseCheckpointSaver,
    ):
        graph = StateGraph(WorkflowState)

        # Setup workflow graph
        graph = self._setup_workflow_graph(
            graph,
            tools_registry,
            goal,
        )

        return graph.compile(checkpointer=checkpointer)

    def _setup_translator_nodes(self, tools_registry: ToolsRegistry):
        translation_tools = ["create_file_with_contents", "read_file"]
        agents_toolset = tools_registry.toolset(translation_tools)
        translator_agent = Agent(
            goal="N/A",
            system_prompt="N/A",
            name=AGENT_NAME,
            model=new_chat_client(max_tokens=MAX_TOKENS_TO_SAMPLE),
            toolset=agents_toolset,
            http_client=self._http_client,
            workflow_id=self._workflow_id,
            workflow_type=CategoryEnum.WORKFLOW_CONVERT_TO_GITLAB_CI,
        )

        return {
            "agent": translator_agent,
            "tools": translation_tools,
            "tools_executor": ToolsExecutor(
                tools_agent_name=AGENT_NAME,
                toolset=agents_toolset,
                workflow_id=self._workflow_id,
                workflow_type=CategoryEnum.WORKFLOW_CONVERT_TO_GITLAB_CI,
            ),
            "start_node": "request_translation",
        }

    def _setup_workflow_graph(
        self,
        graph: StateGraph,
        tools_registry,
        ci_config_file_path,
    ):
        translator_components = self._setup_translator_nodes(tools_registry)

        self.log.info("Starting %s workflow graph compilation", self._workflow_type)
        graph.set_entry_point("load_files")
        # Load jenkins file contents
        graph.add_node(
            "load_files",
            RunToolNode[WorkflowState](
                tool=tools_registry.get("read_file"),  # type: ignore
                input_parser=lambda _: [{"file_path": ci_config_file_path}],
                output_parser=_load_file_contents,  # type: ignore
            ).run,
        )
        # translator nodes
        graph.add_node(
            translator_components["start_node"], translator_components["agent"].run
        )
        graph.add_node("execution_tools", translator_components["tools_executor"].run)

        # deterministic git actions
        graph.add_node(
            "git_actions",
            RunToolNode[WorkflowState](
                tool=tools_registry.get("run_git_command"),  # type: ignore
                input_parser=lambda _: [
                    {
                        "repository_url": self._project["http_url_to_repo"],
                        "command": "add",
                        "args": "-A",
                    },
                    {
                        "repository_url": self._project["http_url_to_repo"],
                        "command": "commit",
                        "args": "-m 'Duo Workflow: Convert to GitLab CI'",
                    },
                    {
                        "repository_url": self._project["http_url_to_repo"],
                        "command": "push",
                        "args": "-o merge_request.create",
                    },
                ],
                output_parser=_git_output,  # type: ignore
            ).run,
        )

        graph.add_node(
            "complete",
            HandoverAgent(
                new_status=WorkflowStatusEnum.COMPLETED, handover_from=AGENT_NAME
            ).run,
        )

        graph.add_edge("load_files", translator_components["start_node"])
        graph.add_conditional_edges(
            translator_components["start_node"],
            _tools_execution_requested,
            {
                Routes.CONTINUE: "execution_tools",
                Routes.END: "complete",
            },
        )
        graph.add_conditional_edges(
            "execution_tools",
            _router,
            {
                Routes.AGENT: translator_components["start_node"],
                Routes.END: "complete",
                Routes.COMMIT_CHANGES: "git_actions",
            },
        )
        graph.add_edge("git_actions", "complete")
        graph.add_edge("complete", END)
        return graph

    def get_workflow_state(self, goal: str) -> WorkflowState:
        target_file = goal
        initial_ui_chat_log = UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            content=f"Starting Jenkinsfile translation workflow from file: {target_file}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.SUCCESS,
            correlation_id=None,
            tool_info=None,
            context_elements=None,
        )

        return WorkflowState(
            status=WorkflowStatusEnum.NOT_STARTED,
            ui_chat_log=[initial_ui_chat_log],
            conversation_history={},
            plan=Plan(steps=[]),
            handover=[],
            last_human_input=None,
        )

    def log_workflow_elements(self, element):
        self.log.info("###############################")
        if "ui_chat_log" in element:
            for log in element["ui_chat_log"]:
                self.log.info(
                    f"%s: %{'' if DEBUG else f'.{MAX_MESSAGE_LENGTH}'}s",
                    log["message_type"],
                    log["content"],
                )
