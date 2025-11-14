from datetime import datetime, timezone
from typing import Any, Dict, List

import structlog
from anthropic import APIStatusError
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.output_parsers.string import StrOutputParser

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
from duo_workflow_service.llm_factory import LLMFinishReason
from duo_workflow_service.tracking.errors import log_exception

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
            if (
                self.tools_registry
                and self.tools_registry.approval_required(call["name"])
                and call["name"] not in preapproved_tools
                and not getattr(
                    self.prompt_adapter.get_model(), "_is_agentic_mock_model", False
                )
            ):
                approval_required = True
                approval_messages.append(
                    UiChatLog(
                        message_type=MessageTypeEnum.REQUEST,
                        message_sub_type=None,
                        content=f"Tool {call['name']} requires approval. Please confirm if you want to proceed.",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        status=ToolStatus.SUCCESS,
                        correlation_id=None,
                        tool_info=ToolInfo(name=call["name"], args=call["args"]),
                        additional_context=None,
                    )
                )

        return approval_required, approval_messages

    def _handle_wrong_messages_order_for_tool_execution(self, input: ChatWorkflowState):
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
            self.name in input["conversation_history"]
            and len(input["conversation_history"][self.name]) > 1
        ):
            tool_call_message = input["conversation_history"][self.name][-2]
            user_message = input["conversation_history"][self.name][-1]

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

                input["conversation_history"][self.name][-2:] = [
                    tool_call_message,
                    *messages,
                    user_message,
                ]

    def _handle_approval_rejection(
        self, input: ChatWorkflowState, approval_state: ApprovalStateRejection
    ) -> list[BaseMessage]:
        last_message = input["conversation_history"][self.name][-1]

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
        input["conversation_history"][self.name].extend(messages)
        return messages

    async def _get_agent_response(self, input: ChatWorkflowState) -> BaseMessage:
        return await self.prompt_adapter.get_response(
            input, system_template_override=self.system_template_override
        )

    def _build_response(
        self, agent_response: BaseMessage, input: ChatWorkflowState
    ) -> Dict[str, Any]:
        if not isinstance(agent_response, AIMessage) or not agent_response.tool_calls:
            return self._build_text_response(agent_response)

        return self._build_tool_response(agent_response, input)

    def _build_text_response(self, agent_response: BaseMessage) -> Dict[str, Any]:
        ui_chat_log = UiChatLog(
            message_type=MessageTypeEnum.AGENT,
            message_sub_type=None,
            content=StrOutputParser().invoke(agent_response) or "",
            timestamp=datetime.now(timezone.utc).isoformat(),
            status=ToolStatus.SUCCESS,
            correlation_id=None,
            tool_info=None,
            additional_context=None,
        )

        return {
            "conversation_history": {self.name: [agent_response]},
            "status": WorkflowStatusEnum.INPUT_REQUIRED,
            "ui_chat_log": [ui_chat_log],
        }

    def _build_tool_response(
        self, agent_response: AIMessage, input: ChatWorkflowState
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "conversation_history": {self.name: [agent_response]},
            "status": WorkflowStatusEnum.EXECUTION,
        }

        preapproved_tools = input.get("preapproved_tools") or []
        tools_need_approval, approval_messages = self._get_approvals(
            agent_response, preapproved_tools
        )

        if len(agent_response.tool_calls) > 0 and tools_need_approval:
            result["status"] = WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED
            result["ui_chat_log"] = approval_messages

        return result

    def _create_error_response(self, error: Exception) -> Dict[str, Any]:
        error_message = HumanMessage(
            content=f"There was an error processing your request: {error}"
        )

        if isinstance(error, APIStatusError) and 500 <= error.status_code < 600:
            ui_content = (
                "There was an error connecting to the chosen LLM provider, please try again or contact support "
                "if the issue persists."
            )
        else:
            ui_content = (
                "There was an error processing your request in the Duo Agent Platform, please try again or "
                "contact support if the issue persists."
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
        )

        return {
            "conversation_history": {self.name: [error_message]},
            "status": WorkflowStatusEnum.INPUT_REQUIRED,
            "ui_chat_log": [ui_chat_log],
        }

    async def run(self, input: ChatWorkflowState) -> Dict[str, Any]:
        approval_state = input.get("approval", None)

        self._handle_wrong_messages_order_for_tool_execution(input)

        # Handle approval rejection
        if isinstance(approval_state, ApprovalStateRejection):
            self._handle_approval_rejection(input, approval_state)

        try:
            with GitLabServiceContext(
                GitLabInstanceInfoService(),
                project=input.get("project"),
                namespace=input.get("namespace"),
            ):
                agent_response = await self._get_agent_response(input)

            # Check for abnormal finish reasons
            finish_reason = agent_response.response_metadata.get("finish_reason")
            if finish_reason in LLMFinishReason.abnormal_values():
                log.warning(f"LLM stopped abnormally with reason: {finish_reason}")

            return self._build_response(agent_response, input)

        except Exception as error:
            log_exception(error, extra={"context": "Error processing chat agent"})
            return self._create_error_response(error)
