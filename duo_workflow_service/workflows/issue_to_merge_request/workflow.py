from datetime import datetime, timezone
from enum import StrEnum
from typing import Annotated, Any

from langchain_core.messages import AIMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.constants import END
from langgraph.graph import StateGraph

from duo_workflow_service.agents import (
    Agent,
    HandoverAgent,
    PlanTerminatorAgent,
    RunToolNode,
    ToolsExecutor,
)
from duo_workflow_service.agents.prompts import HANDOVER_TOOL_NAME
from duo_workflow_service.components import ToolsApprovalComponent, ToolsRegistry
from duo_workflow_service.components.executor import ExecutorComponent
from duo_workflow_service.components.planner import PlannerComponent
from duo_workflow_service.entities import (
    MessageTypeEnum,
    Plan,
    ToolStatus,
    UiChatLog,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.gitlab.url_parser import GitLabUrlParseError, GitLabUrlParser
from duo_workflow_service.llm_factory import create_chat_model
from duo_workflow_service.tracking import log_exception
from duo_workflow_service.workflows.abstract_workflow import (
    MAX_TOKENS_TO_SAMPLE,
    AbstractWorkflow,
)
from duo_workflow_service.workflows.issue_to_merge_request.prompts import (
    BUILD_CONTEXT_SYSTEM_MESSAGE,
)

CONTEXT_BUILDER_TOOLS = [
    "list_issues",
    "get_issue",
    "list_issue_notes",
    "get_issue_note",
    "get_job_logs",
    "get_merge_request",
    "get_project",
    "get_pipeline_errors",
    "list_all_merge_request_notes",
    "list_merge_request_diffs",
    "gitlab_issue_search",
    "gitlab_merge_request_search",
    "read_file",
    "find_files",
    "list_dir",
    "grep",
    "handover_tool",
    "get_epic",
    "list_epics",
    "get_repository_file",
    "list_epic_notes",
    "get_epic_note",
    "get_commit",
    "list_commits",
    "get_commit_comments",
    "get_commit_diff",
    "get_work_item",
    "list_work_items",
    "get_work_item_notes",
    "create_merge_request",
]

PLANNER_TOOLS = [
    "get_plan",
    "add_new_task",
    "remove_task",
    "update_task_description",
    "handover_tool",
    "create_plan",
]

EXECUTOR_TOOLS = [
    "list_issues",
    "get_issue",
    "list_issue_notes",
    "get_issue_note",
    "get_job_logs",
    "get_merge_request",
    "get_pipeline_errors",
    "get_project",
    "list_all_merge_request_notes",
    "list_merge_request_diffs",
    "gitlab_issue_search",
    "gitlab_merge_request_search",
    "read_file",
    "create_file_with_contents",
    "edit_file",
    "find_files",
    "grep",
    "mkdir",
    "get_plan",
    "set_task_status",
    "handover_tool",
    "get_epic",
    "list_epics",
    "get_repository_file",
    "list_dir",
    "list_epic_notes",
    "get_epic_note",
    "get_commit",
]


class Routes(StrEnum):
    END = "end"
    CALL_TOOL = "call_tool"
    HANDOVER = HandoverAgent.__name__
    BUILD_CONTEXT = "build_context"
    STOP = "stop"


def _router(
    state: WorkflowState,
) -> Routes:
    if state["status"] in [WorkflowStatusEnum.CANCELLED, WorkflowStatusEnum.ERROR]:
        return Routes.STOP

    last_message = state["conversation_history"]["context_builder"][-1]
    if isinstance(last_message, AIMessage) and len(last_message.tool_calls) > 0:
        if last_message.tool_calls[0]["name"] == HANDOVER_TOOL_NAME:
            return Routes.HANDOVER
        return Routes.CALL_TOOL

    return Routes.STOP


def _should_continue(
    state: WorkflowState,
) -> Routes:
    if state["status"] in [WorkflowStatusEnum.ERROR, WorkflowStatusEnum.CANCELLED]:
        return Routes.STOP

    return Routes.BUILD_CONTEXT


def _git_output(
    command_output: list[str], state: WorkflowState
):  # pylint: disable=unused-argument
    logs: list[UiChatLog] = [
        UiChatLog(
            message_type=MessageTypeEnum.TOOL,
            message_sub_type=None,
            content=f"{command_output[-1]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.SUCCESS,
            correlation_id=None,
            tool_info=None,
            additional_context=None,
        )
    ]

    return {
        "ui_chat_log": logs,
    }


class Workflow(AbstractWorkflow):
    async def _handle_workflow_failure(
        self, error: BaseException, compiled_graph: Any, graph_config: Any
    ):
        log_exception(
            error, extra={"workflow_id": self._workflow_id, "source": __name__}
        )

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

    def _setup_workflow_graph(
        self,
        graph: StateGraph,
        tools_registry,
        goal,
    ):

        self.log.info("Starting %s workflow graph compilation", self._workflow_type)

        # Add nodes to the graph
        graph.set_entry_point("build_context")

        build_context_handover_node = self._add_context_builder_nodes(
            graph, goal, tools_registry
        )

        planner_component = PlannerComponent(
            user=self._user,
            workflow_id=self._workflow_id,
            workflow_type=self._workflow_type,
            planner_toolset=tools_registry.toolset(PLANNER_TOOLS),
            executor_toolset=tools_registry.toolset(EXECUTOR_TOOLS),
            tools_registry=tools_registry,
            model_config=self._model_config,
            goal=f"Consider the following issue url: {goal}. Create an implementation plan to address the issue "
            f"requirements.",
            project=self._project,  # type: ignore[arg-type]
            http_client=self._http_client,
        )

        planner_entry_node = planner_component.attach(
            graph=graph,
            next_node="set_status_to_execution",
            exit_node="plan_terminator",
            approval_component=None,
        )

        graph.add_edge(build_context_handover_node, planner_entry_node)

        plan_terminator = PlanTerminatorAgent(workflow_id=self._workflow_id)
        graph.add_node("plan_terminator", plan_terminator.run)
        graph.add_edge("plan_terminator", END)

        graph.add_node(
            "set_status_to_execution",
            HandoverAgent(
                new_status=WorkflowStatusEnum.EXECUTION,
                handover_from="planner",
            ).run,
        )

        execution_approval_component = ToolsApprovalComponent(
            workflow_id=self._workflow_id,
            approved_agent_name="executor",
            approved_agent_state=WorkflowStatusEnum.EXECUTION,
            toolset=tools_registry.toolset(EXECUTOR_TOOLS),
        )

        executor_component = ExecutorComponent(
            workflow_id=self._workflow_id,
            workflow_type=self._workflow_type,
            executor_toolset=tools_registry.toolset(EXECUTOR_TOOLS),
            tools_registry=tools_registry,
            model_config=self._model_config,
            goal=f"Consider the following issue url: {goal}. Implement changes to meet issue requirements.",
            project=self._project,  # type: ignore[arg-type]
            http_client=self._http_client,
            user=self._user,
        )

        executor_entry_node = executor_component.attach(
            graph=graph,
            next_node="git_actions",
            exit_node="plan_terminator",
            approval_component=execution_approval_component,
        )
        graph.add_edge("set_status_to_execution", executor_entry_node)

        issue_iid = self._fetch_issue_iid(goal)
        # deterministic git actions
        graph.add_node(
            "git_actions",
            RunToolNode[WorkflowState](
                tool=tools_registry.get("run_git_command"),  # type: ignore
                input_parser=lambda _: [
                    {
                        "repository_url": self._project["http_url_to_repo"],  # type: ignore[index]
                        "command": "add",
                        "args": "-A",
                    },
                    {
                        "repository_url": self._project["http_url_to_repo"],  # type: ignore[index]
                        "command": "commit",
                        "args": f"-m 'Duo Workflow: Resolve issue #{issue_iid}'",
                    },
                    {
                        "repository_url": self._project["http_url_to_repo"],  # type: ignore[index]
                        "command": "push",
                    },
                ],
                output_parser=_git_output,  # type: ignore
                flow_type=self._workflow_type,
            ).run,
        )
        graph.add_edge("git_actions", END)
        return graph

    def _add_context_builder_nodes(
        self, graph: StateGraph, goal: str, tools_registry: ToolsRegistry
    ) -> Annotated[str, "The name of the last handover node"]:
        context_builder_components = self._setup_context_builder(goal, tools_registry)

        graph.add_node("build_context", context_builder_components["agent"].run)
        graph.add_node(
            "build_context_tools", context_builder_components["tools_executor"].run
        )
        graph.add_node(
            "build_context_handover", context_builder_components["handover"].run
        )

        graph.add_conditional_edges(
            "build_context",
            _router,
            {
                Routes.CALL_TOOL: "build_context_tools",
                Routes.HANDOVER: "build_context_handover",
                Routes.STOP: "plan_terminator",
            },
        )
        graph.add_conditional_edges(
            "build_context_tools",
            _should_continue,
            {
                Routes.BUILD_CONTEXT: "build_context",
                Routes.STOP: "plan_terminator",
            },
        )

        return "build_context_handover"

    def _setup_context_builder(
        self,
        goal: str,
        tools_registry: ToolsRegistry,
    ):
        context_builder_toolset = tools_registry.toolset(CONTEXT_BUILDER_TOOLS)
        context_builder = Agent(
            goal=f"Consider the following issue url: {goal}. Build context and identify development tasks from the "
            f"issue requirements.",
            model=create_chat_model(
                max_tokens=MAX_TOKENS_TO_SAMPLE,
                config=self._model_config,
            ),  # type: ignore
            name="context_builder",
            system_prompt=BUILD_CONTEXT_SYSTEM_MESSAGE.format(
                handover_tool_name=HANDOVER_TOOL_NAME,
                issue_url=goal,
                current_branch=self._workflow_metadata["git_branch"],
                default_branch=self._project["default_branch"],  # type: ignore[index]
                project_id=self._project["id"],  # type: ignore[index]
                workflow_id=self._workflow_id,
                session_url=self._session_url,
            ),
            toolset=context_builder_toolset,
            workflow_id=self._workflow_id,
            http_client=self._http_client,
            workflow_type=self._workflow_type,
        )

        return {
            "agent": context_builder,
            "toolset": context_builder_toolset,
            "handover": HandoverAgent(
                new_status=WorkflowStatusEnum.PLANNING,
                handover_from=context_builder.name,
                include_conversation_history=True,
            ),
            "tools_executor": ToolsExecutor(
                tools_agent_name=context_builder.name,
                toolset=context_builder_toolset,
                workflow_id=self._workflow_id,
                workflow_type=self._workflow_type,
            ),
        }

    def get_workflow_state(self, goal: str) -> WorkflowState:
        return WorkflowState(
            status=WorkflowStatusEnum.NOT_STARTED,
            ui_chat_log=[],
            conversation_history={},
            plan=Plan(steps=[]),
            handover=[],
            last_human_input=None,
            project=self._project,
            goal=goal,
            additional_context=None,
        )

    def _fetch_issue_iid(self, issue_url: str):
        try:
            gitlab_host = GitLabUrlParser.extract_host_from_url(
                self._project["web_url"]  # type: ignore[index]
            )
            _, issue_iid = GitLabUrlParser.parse_issue_url(issue_url, gitlab_host)
            return issue_iid
        except GitLabUrlParseError as e:
            log_exception(e, extra={"workflow_id": self._workflow_id})
            return ""
