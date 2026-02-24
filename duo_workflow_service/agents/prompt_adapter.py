from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, override

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.runnables import Runnable, RunnableConfig

from ai_gateway.model_selection.models import ModelClassProvider
from ai_gateway.prompts import Prompt, jinja2_formatter
from ai_gateway.prompts.config.base import PromptConfig
from duo_workflow_service.entities.state import ChatWorkflowState
from duo_workflow_service.gitlab.gitlab_api import Namespace, Project
from duo_workflow_service.gitlab.gitlab_service_context import GitLabServiceContext
from duo_workflow_service.slash_commands.error_handler import (
    SlashCommandValidationError,
)
from duo_workflow_service.slash_commands.goal_parser import parse as slash_command_parse
from lib.context import current_model_metadata_context
from lib.prompts.caching import prompt_caching_enabled_in_current_request

VALID_SLASH_COMMANDS = ["explain", "refactor", "tests", "fix"]


class ChatAgentPromptTemplate(Runnable[ChatWorkflowState, PromptValue]):
    def __init__(self, model_provider: ModelClassProvider, config: PromptConfig):
        self.model_provider = model_provider
        self.prompt_template = config.prompt_template

    def is_slash_command_format(self, message):
        content = message.content.strip()
        if not content.startswith("/"):
            return False

        # Exclude file paths like '/home/dir'
        if content.split(" ")[0].count("/") > 1:
            return False

        return True

    @override
    def invoke(
        self,
        input: ChatWorkflowState,
        config: Optional[RunnableConfig] = None,
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
            caching_status = prompt_caching_enabled_in_current_request()
            user_opted_out_of_caching = caching_status == "false"
            should_show_current_time = (
                self.model_provider == ModelClassProvider.OPENAI
                and user_opted_out_of_caching
            )

            dynamic_content = jinja2_formatter(
                self.prompt_template["system_dynamic"],
                current_date=datetime.now().strftime("%Y-%m-%d"),
                current_time=datetime.now().strftime("%H:%M:%S"),
                current_timezone=datetime.now().astimezone().tzname(),
                should_show_current_time=should_show_current_time,
                project=project,
                namespace=namespace,
            )
            messages.append(SystemMessage(content=dynamic_content))

        for m in input["conversation_history"][agent_name]:
            if isinstance(m, HumanMessage):
                slash_command = None

                if isinstance(m.content, str) and self.is_slash_command_format(m):
                    command_name, remaining_text = slash_command_parse(m.content)

                    # Check if this is the last message and validate it
                    is_last_message = m == input["conversation_history"][agent_name][-1]
                    if is_last_message and command_name not in VALID_SLASH_COMMANDS:
                        raise SlashCommandValidationError(
                            f"The command '/{command_name}' does not exist."
                        )

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
