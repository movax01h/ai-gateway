from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.runnables import Runnable, RunnableConfig

from ai_gateway.model_metadata import current_model_metadata_context
from ai_gateway.prompts import Prompt, jinja2_formatter
from ai_gateway.prompts.config.base import PromptConfig
from duo_workflow_service.entities.state import ChatWorkflowState
from duo_workflow_service.gitlab.gitlab_api import Namespace, Project
from duo_workflow_service.gitlab.gitlab_service_context import GitLabServiceContext
from duo_workflow_service.slash_commands.goal_parser import parse as slash_command_parse


class ChatAgentPromptTemplate(Runnable[ChatWorkflowState, PromptValue]):
    def __init__(self, config: PromptConfig):
        self.prompt_template = config.prompt_template

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

        model_metadata = current_model_metadata_context.get()

        # Handle system messages with static and dynamic parts
        # Create separate system messages for static and dynamic parts
        if "system_static" in self.prompt_template:
            static_content_text = jinja2_formatter(
                self.prompt_template["system_static"],
                system_template_override=_kwargs.get("system_template_override"),
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
                model_friendly_name=(
                    model_metadata.friendly_name
                    if model_metadata and model_metadata.friendly_name
                    else "Unknown"
                ),
            )
            # Always cache static system prompt for prompt caching supported models
            system_msg = SystemMessage(content=static_content_text)
            messages.append(system_msg)

        if "system_dynamic" in self.prompt_template:
            dynamic_content = jinja2_formatter(
                self.prompt_template["system_dynamic"],
                current_date=datetime.now().strftime("%Y-%m-%d"),
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


class BasePromptAdapter(ABC):
    prompt: Prompt

    def __init__(self, prompt: Prompt):
        self.prompt = prompt

    @abstractmethod
    async def get_response(self, input: ChatWorkflowState, **kwargs) -> BaseMessage:
        pass

    @abstractmethod
    def get_model(self):
        pass


class DefaultPromptAdapter(BasePromptAdapter):
    async def get_response(self, input: ChatWorkflowState, **kwargs) -> BaseMessage:
        return await self.prompt.ainvoke(
            input=input,
            agent_name=self.prompt.name,
            **kwargs,
        )

    def get_model(self):
        return self.prompt.model
