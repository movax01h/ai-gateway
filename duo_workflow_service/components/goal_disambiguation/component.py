# pylint: disable=direct-environment-variable-reference

import os
from datetime import datetime, timezone
from enum import StrEnum
from functools import partial
from typing import Annotated, List, Literal, Union

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from duo_workflow_service.agents.agent import Agent
from duo_workflow_service.agents.handover import HandoverAgent
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.entities.event import WorkflowEvent, WorkflowEventType
from duo_workflow_service.entities.state import (
    MessageTypeEnum,
    ToolStatus,
    UiChatLog,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.tools.request_user_clarification import (
    RequestUserClarificationTool,
)

from ...internal_events.event_enum import CategoryEnum
from .prompts import (
    ASSIGNMENT_PROMPT,
    CLARITY_JUDGE_RESPONSE_TEMPLATE,
    PROMPT,
    SYS_PROMPT,
)

_AGENT_NAME = "clarity_judge"

_MIN_CLARITY_THRESHOLD = 4
_MIN_CLARITY_GRADE = "CLEAR"


class Routes(StrEnum):
    UNCLEAR = "unclear"
    CLEAR = "clear"
    SKIP = "skip_further_clarification"
    CONTINUE = "continue"
    BACK = "back"
    STOP = "stop"


class GoalDisambiguationComponent:
    #: pylint: disable=too-many-positional-arguments
    def __init__(
        self,
        goal: str,
        model: BaseChatModel,
        workflow_id: str,
        allow_agent_to_request_user: bool,
        tools_registry: ToolsRegistry,
        http_client: GitlabHttpClient,
        workflow_type: CategoryEnum,
    ):
        self._goal = goal
        self._model = model
        self._workflow_id = workflow_id
        self._http_client = http_client
        self._tools_registry = tools_registry
        self._allow_agent_to_request_user = self._allowed_to_clarify(
            allow_agent_to_request_user
        )
        self._workflow_type = workflow_type

    # pylint: enable=too-many-positional-arguments

    def attach(
        self,
        graph: StateGraph,
        component_exit_node: str,
        component_execution_state: WorkflowStatusEnum,
        graph_termination_node: str = END,
    ) -> Annotated[str, "Entry node name"]:
        if not self._allow_agent_to_request_user:
            return component_exit_node

        task_clarity_judge = Agent(
            goal="N/A",  # "Not used, Agent always gets prepared messages from previous steps",
            system_prompt="N/A",
            name=_AGENT_NAME,
            model=self._model,
            toolset=self._tools_registry.toolset(
                [RequestUserClarificationTool.tool_title]
            ),
            http_client=self._http_client,
            workflow_id=self._workflow_id,
            workflow_type=self._workflow_type,
        )
        task_clarity_handover = HandoverAgent(
            new_status=WorkflowStatusEnum.PLANNING,
            handover_from=_AGENT_NAME,
            include_conversation_history=True,
        )
        graph.add_node("task_clarity_build_prompt", self._build_prompt)
        graph.add_edge("task_clarity_build_prompt", "task_clarity_check")
        graph.add_node("task_clarity_check", task_clarity_judge.run)
        graph.add_conditional_edges(
            "task_clarity_check",
            self._clarification_required,
            {
                Routes.CLEAR: "task_clarity_handover",
                Routes.SKIP: "task_clarity_cancel_pending_tool_call",
                Routes.UNCLEAR: "task_clarity_request_clarification",
                Routes.STOP: graph_termination_node,
            },
        )

        graph.add_node("task_clarity_request_clarification", self._ask_question)
        graph.add_edge(
            "task_clarity_request_clarification", "task_clarity_fetch_user_response"
        )
        graph.add_node(
            "task_clarity_fetch_user_response",
            partial(self._handle_clarification, component_execution_state),
        )
        graph.add_conditional_edges(
            "task_clarity_fetch_user_response",
            self._clarification_provided,
            {
                Routes.BACK: "task_clarity_fetch_user_response",
                Routes.CONTINUE: "task_clarity_check",
                Routes.STOP: graph_termination_node,
            },
        )

        graph.add_node(
            "task_clarity_cancel_pending_tool_call", self._cancel_optional_tool_call
        )
        graph.add_edge("task_clarity_cancel_pending_tool_call", "task_clarity_handover")
        graph.add_node("task_clarity_handover", task_clarity_handover.run)
        graph.add_edge("task_clarity_handover", component_exit_node)

        return "task_clarity_build_prompt"

    def _allowed_to_clarify(self, allow_agent_to_request_user: bool) -> bool:
        return (
            os.environ.get("FEATURE_GOAL_DISAMBIGUATION", "False").lower()
            in ("true", "1", "t")
            and os.environ.get("USE_MEMSAVER", "False").lower()
            not in ("true", "1", "t")
            and allow_agent_to_request_user
        )

    async def _build_prompt(
        self, state: WorkflowState
    ) -> dict[str, dict[str, list[BaseMessage]]]:
        return {
            "conversation_history": {
                _AGENT_NAME: [
                    SystemMessage(content=SYS_PROMPT),
                    HumanMessage(
                        content=PROMPT.format(
                            clarification_tool=RequestUserClarificationTool.tool_title
                        )
                    ),
                    HumanMessage(
                        content=ASSIGNMENT_PROMPT.format(
                            goal=self._goal,
                            conversation_history="\n".join(
                                map(
                                    lambda x: x.pretty_repr(),
                                    state["handover"],
                                )
                            ),
                        )
                    ),
                ]
            }
        }

    async def _ask_question(
        self, state: WorkflowState
    ) -> dict[str, Union[list[UiChatLog], WorkflowStatusEnum]]:
        last_message: AIMessage = state["conversation_history"][_AGENT_NAME][-1]  # type: ignore
        if last_message.tool_calls is None:
            return {"ui_chat_log": []}

        tool_call = last_message.tool_calls[0]["args"]

        recommendations = (
            "\n".join(
                [
                    f"{i}. {recommendation}"
                    for i, recommendation in enumerate(tool_call["recommendations"], 1)
                ]
            )
            if isinstance(tool_call["recommendations"], list)
            else f"1. {tool_call['recommendations']}"
        )

        response = f"{tool_call['response']}\n" if tool_call.get("response") else ""

        return {
            "ui_chat_log": [
                UiChatLog(
                    message_type=MessageTypeEnum.REQUEST,
                    content=CLARITY_JUDGE_RESPONSE_TEMPLATE.format(
                        response=response,
                        message=tool_call.get("message", ""),
                        recommendations=recommendations,
                    ).strip(),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    status=None,
                    correlation_id=None,
                    tool_info=None,
                )
            ],
            "status": WorkflowStatusEnum.INPUT_REQUIRED,
        }

    async def _handle_clarification(
        self, component_execution_state: WorkflowStatusEnum, state: WorkflowState
    ) -> dict[
        str, Union[list[UiChatLog], WorkflowStatusEnum, dict[str, list[BaseMessage]]]
    ]:
        event: WorkflowEvent = interrupt(
            "Workflow interrupted; waiting for user's clarification."
        )

        if event["event_type"] == WorkflowEventType.STOP:
            return {"status": WorkflowStatusEnum.CANCELLED}

        if event["event_type"] != WorkflowEventType.MESSAGE:
            return {"status": WorkflowStatusEnum.INPUT_REQUIRED}

        message = event["message"]
        ui_chat_logs = [
            UiChatLog(
                correlation_id=(
                    event["correlation_id"] if event.get("correlation_id") else None
                ),
                message_type=MessageTypeEnum.USER,
                content=message,
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=ToolStatus.SUCCESS,
                tool_info=None,
            )
        ]

        last_message = state["conversation_history"][_AGENT_NAME][-1]
        messages: List[BaseMessage] = [
            ToolMessage(
                content=f"{message}",
                tool_call_id=tool_call.get("id"),
            )
            for tool_call in getattr(last_message, "tool_calls", [])
        ]
        messages.append(
            HumanMessage(
                content=(
                    f"Review my feedback in the {RequestUserClarificationTool.tool_title} tool response.\n"
                    "Answer all question within my feedback, and finally reevaluate clarity."
                )
            )
        )

        return {
            "status": component_execution_state,
            "ui_chat_log": ui_chat_logs,
            "conversation_history": {_AGENT_NAME: messages},
        }

    def _clarification_required(
        self, state: WorkflowState
    ) -> Literal[Routes.CLEAR, Routes.UNCLEAR, Routes.SKIP, Routes.STOP]:
        last_message: AIMessage = state["conversation_history"][_AGENT_NAME][-1]  # type: ignore
        if state["status"] == WorkflowStatusEnum.CANCELLED:
            return Routes.STOP

        if last_message.tool_calls is None or len(last_message.tool_calls) == 0:
            return Routes.CLEAR

        tool_call = last_message.tool_calls[0]["args"]  # type: ignore
        if (
            tool_call["clarity_verdict"] == _MIN_CLARITY_GRADE
            or tool_call["clarity_score"] >= _MIN_CLARITY_THRESHOLD
        ):
            return Routes.SKIP

        return Routes.UNCLEAR

    def _clarification_provided(
        self, state: WorkflowState
    ) -> Literal[Routes.CONTINUE, Routes.BACK, Routes.STOP]:
        if state["status"] == WorkflowStatusEnum.CANCELLED:
            return Routes.STOP

        if state["status"] == WorkflowStatusEnum.INPUT_REQUIRED:
            return Routes.BACK

        return Routes.CONTINUE

    def _cancel_optional_tool_call(self, state: WorkflowState):
        last_message = state["conversation_history"][_AGENT_NAME][-1]
        messages: List[BaseMessage] = [
            ToolMessage(
                content="Task is specific enough, no further clarification is required.",
                tool_call_id=tool_call.get("id"),
            )
            for tool_call in getattr(last_message, "tool_calls", [])
        ]
        return {"conversation_history": {_AGENT_NAME: messages}}
