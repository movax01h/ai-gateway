from datetime import datetime, timezone
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers.string import StrOutputParser
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.runnables import Runnable, RunnableConfig

from ai_gateway.prompts import Prompt, jinja2_formatter
from ai_gateway.prompts.config import ModelConfig
from duo_workflow_service.entities.state import (
    ChatWorkflowState,
    MessageTypeEnum,
    ToolStatus,
    UiChatLog,
    WorkflowStatusEnum,
)
from duo_workflow_service.gitlab.gitlab_project import Project
from duo_workflow_service.slash_commands.goal_parser import parse as slash_command_parse


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
    @classmethod
    def _build_prompt_template(
        cls, prompt_template: dict[str, str], model_config: ModelConfig
    ) -> Runnable:
        return ChatAgentPromptTemplate(prompt_template, model_config)

    async def run(self, input: ChatWorkflowState) -> Dict[str, Any]:
        agent_response = await super().ainvoke(input=input, agent_name=self.name)

        if isinstance(agent_response, AIMessage) and len(agent_response.tool_calls) > 0:
            status = WorkflowStatusEnum.EXECUTION
        else:
            status = WorkflowStatusEnum.INPUT_REQUIRED

        result = {
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
                    content=StrOutputParser().invoke(agent_response) or "",
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    status=ToolStatus.SUCCESS,
                    correlation_id=None,
                    tool_info=None,
                    context_elements=[],
                )
            ]

        return result
