from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

import structlog
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.runnables import Runnable, RunnableConfig

from ai_gateway.prompts import Prompt, jinja2_formatter
from ai_gateway.prompts.config.base import PromptConfig
from ai_gateway.prompts.config.models import ModelClassProvider
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
from duo_workflow_service.gitlab.gitlab_api import Namespace, Project
from duo_workflow_service.gitlab.gitlab_instance_info_service import (
    GitLabInstanceInfoService,
)
from duo_workflow_service.gitlab.gitlab_service_context import GitLabServiceContext
from duo_workflow_service.llm_factory import AnthropicStopReason
from duo_workflow_service.slash_commands.goal_parser import parse as slash_command_parse
from duo_workflow_service.structured_logging import _workflow_id
from lib.internal_events import InternalEventAdditionalProperties
from lib.internal_events.event_enum import CategoryEnum, EventEnum, EventPropertyEnum

log = structlog.stdlib.get_logger("chat_agent")


class ChatAgentPromptTemplate(Runnable[ChatWorkflowState, PromptValue]):
    def __init__(self, prompt_template: dict[str, str]):
        self.prompt_template = prompt_template

    def invoke(
        self,
        input: ChatWorkflowState,
        config: Optional[RunnableConfig] = None,  # pylint: disable=unused-argument
        **_kwargs: Any,
    ) -> PromptValue:
        messages: list[BaseMessage] = []
        agent_name = _kwargs["agent_name"]
        project: Project | None = input.get("project")
        namespace: Namespace | None = input.get("namespace")

        # Get GitLab instance info from context
        gitlab_instance_info = GitLabServiceContext.get_current_instance_info()

        # Handle system messages with static and dynamic parts
        # Create separate system messages for static and dynamic parts
        if "system_static" in self.prompt_template:
            static_content_text = jinja2_formatter(
                self.prompt_template["system_static"],
                gitlab_instance_type=(
                    gitlab_instance_info.instance_type
                    if gitlab_instance_info
                    else "Unknown"
                ),
                gitlab_instance_url=(
                    gitlab_instance_info.instance_url
                    if gitlab_instance_info
                    else "Unknown"
                ),
                gitlab_instance_version=(
                    gitlab_instance_info.instance_version
                    if gitlab_instance_info
                    else "Unknown"
                ),
            )
            # Always cache static system prompt for Anthropic models
            is_anthropic = _kwargs.get("is_anthropic_model", False)
            if is_anthropic:
                cached_static_content: list[Union[str, dict]] = [
                    {
                        "text": static_content_text,
                        "type": "text",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ]
                messages.append(SystemMessage(content=cached_static_content))
            else:
                messages.append(SystemMessage(content=static_content_text))

        if "system_dynamic" in self.prompt_template:
            dynamic_content = jinja2_formatter(
                self.prompt_template["system_dynamic"],
                current_date=datetime.now().strftime("%Y-%m-%d"),
                current_time=datetime.now().strftime("%H:%M:%S"),
                current_timezone=datetime.now().astimezone().tzname(),
                project=project,
                namespace=namespace,
            )
            messages.append(SystemMessage(content=dynamic_content))

        for m in input["conversation_history"][agent_name]:
            if isinstance(m, HumanMessage):
                slash_command = None

                if isinstance(m.content, str) and m.content.strip().startswith("/"):
                    command_name, remaining_text = slash_command_parse(m.content)
                    slash_command = {
                        "name": command_name,
                        "input": remaining_text,
                    }

                messages.append(
                    HumanMessage(
                        jinja2_formatter(
                            self.prompt_template["user"],
                            message=m,
                            slash_command=slash_command,
                        )
                    )
                )
            else:
                messages.append(m)  # AIMessage or ToolMessage

        return ChatPromptValue(messages=messages)


class ChatAgent(Prompt[ChatWorkflowState, BaseMessage]):
    tools_registry: Optional[ToolsRegistry] = None
    prompt_runnable: Prompt | None = None

    @classmethod
    def _build_prompt_template(cls, config: PromptConfig) -> Runnable:
        return ChatAgentPromptTemplate(config.prompt_template)

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
                and not getattr(self.model, "_is_agentic_mock_model", False)
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
        if len(input["conversation_history"][self.name]) > 1:
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
        is_anthropic_model = self.model_provider == ModelClassProvider.ANTHROPIC

        if self.prompt_runnable:
            conversation_history = input["conversation_history"].get(self.name, [])
            variables = {
                "goal": input["goal"],
                "project": input["project"],
                "namespace": input["namespace"],
                "current_date": datetime.now().strftime("%Y-%m-%d"),
                "current_time": datetime.now().strftime("%H:%M:%S"),
                "current_timezone": datetime.now().astimezone().tzname(),
            }

            return await self.prompt_runnable.ainvoke(
                input={**variables, "history": conversation_history}
            )

        return await super().ainvoke(
            input=input,
            agent_name=self.name,
            is_anthropic_model=is_anthropic_model,
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

        ui_chat_log = UiChatLog(
            message_type=MessageTypeEnum.AGENT,
            message_sub_type=None,
            content=(
                "There was an error processing your request. Please try again or contact support if "
                "the issue persists."
            ),
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

            # Check for abnormal stop reasons
            stop_reason = agent_response.response_metadata.get("stop_reason")
            if stop_reason in AnthropicStopReason.abnormal_values():
                log.warning(f"LLM stopped abnormally with reason: {stop_reason}")

            # Track tokens for AI messages
            if isinstance(agent_response, AIMessage):
                self._track_tokens_data(agent_response)

            return self._build_response(agent_response, input)

        except Exception as error:
            log.warning(f"Error processing chat agent: {error}")
            return self._create_error_response(error)

    def _track_tokens_data(self, message: AIMessage):
        if not self.internal_event_client:
            return

        usage_metadata = message.usage_metadata if message.usage_metadata else {}  # type: ignore[typeddict-item]

        additional_properties = InternalEventAdditionalProperties(
            label=self.name,
            property=EventPropertyEnum.WORKFLOW_ID.value,
            value=_workflow_id.get(),
            input_tokens=usage_metadata.get("input_tokens"),
            output_tokens=usage_metadata.get("output_tokens"),
            total_tokens=usage_metadata.get("total_tokens"),
        )
        self.internal_event_client.track_event(
            event_name=EventEnum.TOKEN_PER_USER_PROMPT.value,
            additional_properties=additional_properties,
            category=CategoryEnum.WORKFLOW_CHAT.value,
        )
