from datetime import datetime, timezone
from typing import Any, Dict, Optional

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
    ChatWorkflowState,
    MessageTypeEnum,
    ToolInfo,
    ToolStatus,
    UiChatLog,
    WorkflowStatusEnum,
)
from duo_workflow_service.gitlab.gitlab_project import Project
from duo_workflow_service.slash_commands.goal_parser import parse as slash_command_parse

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

        if "system" in self.prompt_template:
            content = jinja2_formatter(
                self.prompt_template["system"],
                current_date=datetime.now().strftime("%Y-%m-%d"),
                project=project,
            )
            messages.append(SystemMessage(content=content))

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
                        context_elements=[],
                    )
                )

        return approval_required, approval_messages

    async def run(self, input: ChatWorkflowState) -> Dict[str, Any]:
        new_messages = []

        if input.get("cancel_tool_message", False):
            last_message = input["conversation_history"][self.name][-1]
            messages: list[BaseMessage] = [
                ToolMessage(
                    content=f"Tool cancelled temporarily as user has a comment. Comment: {input['cancel_tool_message']}",
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
                        context_elements=[],
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
                        context_elements=[],
                    )
                ],
            }
