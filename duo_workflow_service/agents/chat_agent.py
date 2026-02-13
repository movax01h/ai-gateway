from datetime import datetime, timezone
from typing import Any, Dict, List
from uuid import uuid4

import structlog
from anthropic import APIStatusError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from duo_workflow_service.agents.prompt_adapter import BasePromptAdapter
from duo_workflow_service.components.tools_registry import ToolsRegistry
from duo_workflow_service.entities.state import (
    ApprovalStateRejection,
    ChatWorkflowState,
    MessageTypeEnum,
    ToolInfo,
    ToolStatus,
    UiChatLog,
    WorkflowStatusEnum,
)
from duo_workflow_service.gitlab.gitlab_instance_info_service import (
    GitLabInstanceInfoService,
)
from duo_workflow_service.gitlab.gitlab_service_context import GitLabServiceContext
from duo_workflow_service.slash_commands.error_handler import (
    SlashCommandValidationError,
)
from duo_workflow_service.tracking.errors import log_exception
from lib.context import LLMFinishReason

log = structlog.stdlib.get_logger("chat_agent")


class ChatAgent:
    def __init__(
        self,
        name: str,
        prompt_adapter: BasePromptAdapter,
        tools_registry: ToolsRegistry,
        system_template_override: str | None,
    ):
        self.name = name
        self.prompt_adapter = prompt_adapter
        self.tools_registry = tools_registry
        self.system_template_override = system_template_override

    def _get_approvals(
        self, message: AIMessage, preapproved_tools: List[str]
    ) -> tuple[bool, list[UiChatLog]]:
        approval_required = False
        approval_messages = []

        for call in message.tool_calls:
            tool_name = call["name"]
            tool_args = call["args"]
            is_agentic_mock = getattr(
                self.prompt_adapter.get_model(),
                "_is_auto_approved_by_agentic_mock_model",
                False,
            )
            needs_approval = (
                self.tools_registry
                and self.tools_registry.approval_required(tool_name, tool_args)
                and tool_name not in preapproved_tools
                and not is_agentic_mock
            )

            if needs_approval:
                approval_required = True
                approval_messages.append(
                    UiChatLog(
                        message_type=MessageTypeEnum.REQUEST,
                        message_sub_type=None,
                        content=f"Tool {tool_name} requires approval. Please confirm if you want to proceed.",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        status=ToolStatus.SUCCESS,
                        correlation_id=None,
                        tool_info=ToolInfo(name=tool_name, args=tool_args),
                        additional_context=None,
                        message_id=f'request-{call["id"]}',
                    )
                )

        return approval_required, approval_messages

    def _handle_wrong_messages_order_for_tool_execution(self, state: ChatWorkflowState):
        # A special fix for the following use case:
        #
        # - A user is asked to approve/deny a tool execution
        # - The user stops the chat instead and specifies a follow up message
        #
        # LLM returns an error because a tool call execution was followed by a human message instead of a tool result
        #
        # Expected to be refactored in:
        # - https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/1461
        if (
            self.name in state["conversation_history"]
            and len(state["conversation_history"][self.name]) > 1
        ):
            tool_call_message = state["conversation_history"][self.name][-2]
            user_message = state["conversation_history"][self.name][-1]

            if (
                isinstance(tool_call_message, AIMessage)
                and len(tool_call_message.tool_calls) > 0
                and isinstance(user_message, HumanMessage)
            ):
                messages: list[BaseMessage] = [
                    ToolMessage(
                        content="Tool is cancelled and a user will provide a follow up message.",
                        tool_call_id=tool_call.get("id"),
                    )
                    for tool_call in getattr(tool_call_message, "tool_calls", [])
                ]

                state["conversation_history"][self.name][-2:] = [
                    tool_call_message,
                    *messages,
                    user_message,
                ]

    def _handle_approval_rejection(
        self, state: ChatWorkflowState, approval_state: ApprovalStateRejection
    ) -> list[BaseMessage]:
        last_message = state["conversation_history"][self.name][-1]

        # An empty text box for tool cancellation results in a 'null' message. Converting to None
        # todo: remove this line once we have fixed the frontend to return None instead of 'null'
        # https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/1259
        normalized_message = (
            None if approval_state.message == "null" else approval_state.message
        )

        tool_message = (
            f"Tool is cancelled temporarily as user has a comment. Comment: {normalized_message}"
            if normalized_message
            else "Tool is cancelled by user. Don't run the command and stop tool execution in progress."
        )

        messages: list[BaseMessage] = [
            ToolMessage(
                content=tool_message,
                tool_call_id=tool_call.get("id"),
            )
            for tool_call in getattr(last_message, "tool_calls", [])
        ]

        # update history
        state["conversation_history"][self.name].extend(messages)
        return messages

    async def _get_agent_response(self, state: ChatWorkflowState) -> BaseMessage:
        return await self.prompt_adapter.get_response(
            state, system_template_override=self.system_template_override
        )

    def _build_response(
        self, agent_response: BaseMessage, state: ChatWorkflowState
    ) -> Dict[str, Any]:
        result = {
            "conversation_history": {self.name: [agent_response]},
            "status": WorkflowStatusEnum.INPUT_REQUIRED,
        }

        self._build_text_response(agent_response, result)
        if isinstance(agent_response, AIMessage) and agent_response.tool_calls:
            self._build_tool_response(agent_response, state, result)

        return result

    def _build_text_response(self, agent_response: BaseMessage, result: Dict[str, Any]):
        content = agent_response.text()
        ui_chat_log = []

        if content:
            ui_chat_log.append(
                UiChatLog(
                    message_type=MessageTypeEnum.AGENT,
                    message_sub_type=None,
                    content=content,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    status=ToolStatus.SUCCESS,
                    correlation_id=None,
                    tool_info=None,
                    additional_context=None,
                    message_id=agent_response.id,
                )
            )

        result["ui_chat_log"] = ui_chat_log

    def _build_tool_response(
        self,
        agent_response: AIMessage,
        state: ChatWorkflowState,
        result: Dict[str, Any],
    ):
        result["status"] = WorkflowStatusEnum.EXECUTION

        preapproved_tools = state.get("preapproved_tools") or []
        tools_need_approval, approval_messages = self._get_approvals(
            agent_response, preapproved_tools
        )

        if len(agent_response.tool_calls) > 0 and tools_need_approval:
            result["status"] = WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED
            result["ui_chat_log"].extend(approval_messages)

    def _create_error_response(self, error: Exception) -> Dict[str, Any]:
        log.info("We are hitting an error while making an llm call.")

        if isinstance(error, APIStatusError) and 500 <= error.status_code < 600:
            ui_content = (
                "There was an error connecting to the chosen LLM provider, please contact support "
                "if the issue persists."
            )
        else:
            ui_content = (
                "There was an error processing your request in the Duo Agent Platform, please contact support "
                "if the issue persists."
            )

        ui_chat_log = UiChatLog(
            message_type=MessageTypeEnum.AGENT,
            message_sub_type=None,
            content=ui_content,
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.FAILURE,
            correlation_id=None,
            tool_info=None,
            additional_context=None,
            message_id=f"error-{str(uuid4())}",
        )

        return {
            "status": WorkflowStatusEnum.ERROR,
            "ui_chat_log": [ui_chat_log],
        }

    async def run(self, state: ChatWorkflowState) -> Dict[str, Any]:
        approval_state = state.get("approval", None)

        self._handle_wrong_messages_order_for_tool_execution(state)

        # Handle approval rejection
        if isinstance(approval_state, ApprovalStateRejection):
            self._handle_approval_rejection(state, approval_state)

        try:
            with GitLabServiceContext(
                GitLabInstanceInfoService(),
                project=state.get("project"),
                namespace=state.get("namespace"),
            ):
                agent_response = await self._get_agent_response(state)

            # Check for abnormal finish reasons
            finish_reason = agent_response.response_metadata.get("finish_reason")
            if finish_reason in LLMFinishReason.abnormal_values():
                log.warning(f"LLM stopped abnormally with reason: {finish_reason}")

            return self._build_response(agent_response, state)

        except SlashCommandValidationError as error:
            log_exception(
                error, extra={"context": "User provided an invalid slash command"}
            )
            # Handle invalid slash commands with a user-friendly message
            error_message = AIMessage(content=str(error))
            ui_chat_log = UiChatLog(
                message_type=MessageTypeEnum.AGENT,
                message_sub_type=None,
                content=str(error),
                timestamp=datetime.now(timezone.utc).isoformat(),
                status=ToolStatus.FAILURE,
                correlation_id=None,
                tool_info=None,
                additional_context=None,
                message_id=f"error-{str(uuid4())}",
            )
            return {
                "conversation_history": {self.name: [error_message]},
                "status": WorkflowStatusEnum.INPUT_REQUIRED,
                "ui_chat_log": [ui_chat_log],
            }
        except Exception as error:
            log_exception(error, extra={"context": "Error processing chat agent"})
            return self._create_error_response(error)
