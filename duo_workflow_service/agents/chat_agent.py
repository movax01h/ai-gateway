from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

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
from ai_gateway.prompts.config import ModelConfig
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
from duo_workflow_service.gitlab.gitlab_project import Project
from duo_workflow_service.slash_commands.goal_parser import parse as slash_command_parse
from duo_workflow_service.structured_logging import _workflow_id
from lib.feature_flags.context import FeatureFlag, is_feature_enabled
from lib.internal_events import InternalEventAdditionalProperties
from lib.internal_events.event_enum import CategoryEnum, EventEnum, EventPropertyEnum

log = structlog.stdlib.get_logger("chat_agent")


class ChatAgentPromptTemplate(Runnable[ChatWorkflowState, PromptValue]):
    def __init__(self, prompt_template: dict[str, str], model_config: ModelConfig):
        self.prompt_template = prompt_template
        self.model_config = model_config

    def invoke(
        self,
        input: ChatWorkflowState,
        config: Optional[RunnableConfig] = None,  # pylint: disable=unused-argument
        **_kwargs: Any,
    ) -> PromptValue:
        messages: list[BaseMessage] = []
        agent_name = _kwargs["agent_name"]
        project: Project | None = input.get("project")

        # Handle system messages with static and dynamic parts
        # Create separate system messages for static and dynamic parts
        if "system_static" in self.prompt_template:
            static_content_text = jinja2_formatter(
                self.prompt_template["system_static"]
            )
            if is_feature_enabled(FeatureFlag.ENABLE_ANTHROPIC_PROMPT_CACHING):
                cached_static_content: list[Union[str, dict]] = [
                    {
                        "text": static_content_text,
                        "type": "text",
                        "cache_control": {"type": "ephemeral"},
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

    @classmethod
    def _build_prompt_template(
        cls, prompt_template: dict[str, str], model_config: ModelConfig
    ) -> Runnable:
        return ChatAgentPromptTemplate(prompt_template, model_config)

    def _get_approvals(self, message: AIMessage) -> tuple[bool, list[UiChatLog]]:
        approval_required = False
        approval_messages = []

        for call in message.tool_calls:
            if self.tools_registry and self.tools_registry.approval_required(
                call["name"]
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

    async def run(self, input: ChatWorkflowState) -> Dict[str, Any]:
        new_messages = []
        approval_state = input.get("approval", None)

        if isinstance(approval_state, ApprovalStateRejection):
            last_message = input["conversation_history"][self.name][-1]

            if approval_state.message:
                tool_message = f"Tool is cancelled temporarily as user has a comment. Comment: {approval_state.message}"
            else:
                tool_message = "Tool is cancelled by user."

            messages: list[BaseMessage] = [
                ToolMessage(
                    content=tool_message,
                    tool_call_id=tool_call.get("id"),
                )
                for tool_call in getattr(last_message, "tool_calls", [])
            ]
            new_messages.extend(messages)
            # update history
            input["conversation_history"][self.name].extend(messages)

        try:
            agent_response = await super().ainvoke(input=input, agent_name=self.name)
            new_messages.append(agent_response)

            if isinstance(agent_response, AIMessage):
                self._track_tokens_data(agent_response)

            if (
                isinstance(agent_response, AIMessage)
                and len(agent_response.tool_calls) > 0
            ):
                status = WorkflowStatusEnum.EXECUTION
            else:
                status = WorkflowStatusEnum.INPUT_REQUIRED

            result: dict[str, Any] = {
                "conversation_history": {self.name: [agent_response]},
                "status": status,
            }

            if (
                not isinstance(agent_response, AIMessage)
                or len(agent_response.tool_calls) == 0
            ):
                result["ui_chat_log"] = [
                    UiChatLog(  # type: ignore[list-item]
                        message_type=MessageTypeEnum.AGENT,
                        message_sub_type=None,
                        content=StrOutputParser().invoke(agent_response) or "",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        status=ToolStatus.SUCCESS,
                        correlation_id=None,
                        tool_info=None,
                        additional_context=None,
                    )
                ]
                result["status"] = WorkflowStatusEnum.INPUT_REQUIRED
                return result

            tools_need_approval, approval_messages = self._get_approvals(agent_response)
            if len(agent_response.tool_calls) > 0 and tools_need_approval:
                result["status"] = WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED
                result["ui_chat_log"] = approval_messages

            return result
        except Exception as error:
            log.warning(f"Error processing chat agent: {error}")

            error_message = HumanMessage(
                content=f"There was an error processing your request: {error}"
            )

            return {
                "conversation_history": {self.name: [error_message]},
                "status": WorkflowStatusEnum.INPUT_REQUIRED,
                "ui_chat_log": [
                    UiChatLog(
                        message_type=MessageTypeEnum.AGENT,
                        message_sub_type=None,
                        content="There was an error processing your request. Please try again or contact support if the issue persists.",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        status=ToolStatus.FAILURE,
                        correlation_id=None,
                        tool_info=None,
                        additional_context=None,
                    )
                ],
            }

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
