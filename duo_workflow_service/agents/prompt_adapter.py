from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional, cast, override

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.runnables import Runnable, RunnableConfig

from ai_gateway.model_metadata import ModelMetadata
from ai_gateway.model_selection.models import ModelClassProvider
from ai_gateway.prompts import Prompt, jinja2_formatter
from ai_gateway.prompts.config.base import PromptConfig
from duo_workflow_service.conversation.trimmer import restore_message_consistency
from duo_workflow_service.entities.state import ChatWorkflowState
from duo_workflow_service.gitlab.gitlab_api import Namespace, Project
from duo_workflow_service.gitlab.gitlab_instance_info_service import GitLabInstanceInfo
from duo_workflow_service.gitlab.gitlab_service_context import GitLabServiceContext
from duo_workflow_service.slash_commands.error_handler import (
    SlashCommandValidationError,
)
from duo_workflow_service.slash_commands.goal_parser import (
    is_slash_command,
)
from duo_workflow_service.slash_commands.goal_parser import parse as slash_command_parse
from lib.context import get_model_metadata
from lib.feature_flags.context import FeatureFlag, is_feature_enabled
from lib.prompts.caching import prompt_caching_enabled_in_current_request
from lib.prompts.utilities import render_security_block

VALID_SLASH_COMMANDS = ["explain", "refactor", "tests", "fix"]


class ChatAgentPromptTemplate(Runnable[ChatWorkflowState, PromptValue]):
    def __init__(self, model_provider: ModelClassProvider, config: PromptConfig):
        self.model_provider = model_provider
        self.prompt_template = config.prompt_template

    def is_slash_command_format(self, message):
        return is_slash_command(message.content)

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

        gitlab_instance_info = GitLabServiceContext.get_current_instance_info()
        model_metadata = get_model_metadata()

        static_template_context = self._build_static_template_context(
            gitlab_instance_info, model_metadata
        )
        dynamic_template_context = self._build_dynamic_template_context(
            project, namespace
        )
        denied_tools = list(dict.fromkeys(input.get("denied_tools", []) or []))
        system_template_override = _kwargs.get("system_template_override")

        if "system" in self.prompt_template:
            messages.extend(
                self._build_single_system_message(
                    static_template_context,
                    dynamic_template_context,
                    system_template_override,
                    denied_tools,
                )
            )
        else:
            messages.extend(
                self._build_split_system_messages(
                    static_template_context,
                    dynamic_template_context,
                    system_template_override,
                    denied_tools,
                )
            )

        history = restore_message_consistency(input["conversation_history"][agent_name])
        for m in history:
            if isinstance(m, HumanMessage):
                slash_command = None

                if isinstance(m.content, str) and self.is_slash_command_format(m):
                    command_name, remaining_text = slash_command_parse(m.content)

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
                            cast(str, self.prompt_template["user"]),
                            message=m,
                            slash_command=slash_command,
                        )
                    )
                )
            else:
                messages.append(m)  # AIMessage or ToolMessage

        return ChatPromptValue(messages=messages)

    @staticmethod
    def _build_static_template_context(
        gitlab_instance_info: GitLabInstanceInfo | None,
        model_metadata: ModelMetadata | None,
    ) -> dict[str, Any]:
        return {
            "gitlab_instance_type": (
                gitlab_instance_info.instance_type
                if gitlab_instance_info
                else "Unknown"
            ),
            "gitlab_instance_url": (
                gitlab_instance_info.instance_url if gitlab_instance_info else "Unknown"
            ),
            "gitlab_instance_version": (
                gitlab_instance_info.instance_version
                if gitlab_instance_info
                else "Unknown"
            ),
            "model_friendly_name": (
                model_metadata.friendly_name
                if model_metadata and model_metadata.friendly_name
                else "Unknown"
            ),
            "clarification_question_tool_enabled": is_feature_enabled(
                FeatureFlag.DUO_CHAT_CLARIFICATION_QUESTION_TOOL
            ),
            "foundational_flow_tool_enabled": is_feature_enabled(
                FeatureFlag.AGENTIC_FOUNDATIONAL_FLOW_TOOL
            ),
        }

    def _build_dynamic_template_context(
        self,
        project: Project | None,
        namespace: Namespace | None,
    ) -> dict[str, Any]:
        caching_status = prompt_caching_enabled_in_current_request()
        user_opted_out_of_caching = caching_status == "false"
        should_show_current_time = (
            self.model_provider == ModelClassProvider.OPENAI
            and user_opted_out_of_caching
        )
        return {
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "current_time": datetime.now().strftime("%H:%M:%S"),
            "current_timezone": datetime.now().astimezone().tzname(),
            "should_show_current_time": should_show_current_time,
            "project": project,
            "namespace": namespace,
        }

    @staticmethod
    def _build_denied_tools_text(denied_tools: list[str]) -> str:
        denied_tools_str = ", ".join(denied_tools)
        return (
            f"The following tools have been blocked by an administrator"
            f" and are not available: {denied_tools_str}. "
            "Do not attempt to use these tools directly or indirectly through other means. "
            "If asked to perform an action that requires a blocked tool, inform the user that "
            "this action has been blocked by an administrator."
        )

    def _build_single_system_message(
        self,
        static_template_context: dict[str, Any],
        dynamic_template_context: dict[str, Any],
        system_template_override: Any,
        denied_tools: list[str],
    ) -> list[SystemMessage]:
        """Build a single SystemMessage for models that require only one system message (e.g. Qwen).

        The security block and denied tools context are appended to the system content so that the static portion at the
        beginning of the message remains cacheable at the prefix level.
        """
        all_context = {**static_template_context, **dynamic_template_context}

        if isinstance(system_template_override, str):
            base_system = jinja2_formatter(
                cast(str, self.prompt_template["system"]),
                system_template_override=None,
                **all_context,
            )
            base_system = base_system.lstrip().rstrip("\n") + "\n"
            system_template_override = jinja2_formatter(
                system_template_override,
                base_agentic_chat_system=base_system,
                **all_context,
            )

        content = jinja2_formatter(
            cast(str, self.prompt_template["system"]),
            system_template_override=system_template_override,
            **all_context,
        )

        content += "\n" + render_security_block()

        if denied_tools:
            content += "\n" + self._build_denied_tools_text(denied_tools)

        return [SystemMessage(content=content)]

    def _build_split_system_messages(
        self,
        static_template_context: dict[str, Any],
        dynamic_template_context: dict[str, Any],
        system_template_override: Any,
        denied_tools: list[str],
    ) -> list[SystemMessage]:
        """Build separate SystemMessages for static, security, denied tools, and dynamic content.

        This is the default path for models that support multiple system messages, enabling prompt caching of the static
        system message.
        """
        messages: list[SystemMessage] = []

        if "system_static" in self.prompt_template:
            if isinstance(system_template_override, str):
                base_agentic_chat_system = jinja2_formatter(
                    cast(str, self.prompt_template["system_static"]),
                    system_template_override=None,
                    **static_template_context,
                )
                base_agentic_chat_system = (
                    base_agentic_chat_system.lstrip().rstrip("\n") + "\n"
                )
                system_template_override = jinja2_formatter(
                    system_template_override,
                    base_agentic_chat_system=base_agentic_chat_system,
                    **static_template_context,
                )

            static_content_text = jinja2_formatter(
                cast(str, self.prompt_template["system_static"]),
                system_template_override=system_template_override,
                **static_template_context,
            )
            messages.append(SystemMessage(content=static_content_text))

        messages.append(SystemMessage(content=render_security_block()))

        if denied_tools:
            messages.append(
                SystemMessage(content=self._build_denied_tools_text(denied_tools))
            )

        if "system_dynamic" in self.prompt_template:
            dynamic_content = jinja2_formatter(
                cast(str, self.prompt_template["system_dynamic"]),
                **dynamic_template_context,
            )
            messages.append(SystemMessage(content=dynamic_content))

        return messages


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
