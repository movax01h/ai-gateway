from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, Union

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.runnables import Runnable, RunnableConfig

from ai_gateway.prompts import Prompt, jinja2_formatter
from ai_gateway.prompts.config.base import PromptConfig
from ai_gateway.prompts.config.models import ModelClassProvider
from duo_workflow_service.agents.base import BaseAgent
from duo_workflow_service.entities.state import ChatWorkflowState
from duo_workflow_service.gitlab.gitlab_api import Namespace, Project
from duo_workflow_service.gitlab.gitlab_service_context import GitLabServiceContext
from duo_workflow_service.slash_commands.goal_parser import parse as slash_command_parse


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


class ChatPrompt(BaseAgent[ChatWorkflowState, BaseMessage]):
    @classmethod
    def _build_prompt_template(cls, config: PromptConfig) -> Runnable:
        return ChatAgentPromptTemplate(config.prompt_template)


class BasePromptAdapter(ABC):
    @abstractmethod
    async def get_response(self, input: ChatWorkflowState) -> BaseMessage:
        pass

    @abstractmethod
    def get_model(self):
        pass


class DefaultPromptAdapter(BasePromptAdapter):
    def __init__(self, base_prompt: Prompt):
        self._base_prompt = base_prompt

    async def get_response(self, input: ChatWorkflowState) -> BaseMessage:
        is_anthropic_model = (
            self._base_prompt.model_provider == ModelClassProvider.ANTHROPIC
        )

        return await self._base_prompt.ainvoke(
            input=input,
            agent_name=self._base_prompt.name,
            is_anthropic_model=is_anthropic_model,
        )

    def get_model(self):
        return self._base_prompt.model


class CustomPromptAdapter(BasePromptAdapter):
    def __init__(self, prompt: Prompt):
        self._prompt = prompt
        self._agent_name = prompt.name

    async def get_response(self, input: ChatWorkflowState) -> BaseMessage:
        conversation_history = input["conversation_history"].get(self._agent_name, [])
        variables = {
            "goal": input["goal"],
            "project": input["project"],
            "namespace": input["namespace"],
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "current_time": datetime.now().strftime("%H:%M:%S"),
            "current_timezone": datetime.now().astimezone().tzname(),
        }

        return await self._prompt.ainvoke(
            input={**variables, "history": conversation_history}
        )

    def get_model(self):
        return self._prompt.model


def create_adapter(
    prompt: Prompt,
    use_custom_adapter: bool = False,
) -> BasePromptAdapter:
    if use_custom_adapter:
        return CustomPromptAdapter(prompt)
    return DefaultPromptAdapter(prompt)
