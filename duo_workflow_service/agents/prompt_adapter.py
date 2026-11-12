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
from ai_gateway.prompts.model_variables import (
    model_friendly_name,
    should_show_current_time,
)
from duo_workflow_service.conversation.trimmer import restore_message_consistency
from duo_workflow_service.entities.message_ingestion import (
    non_blank_text_block,
    split_text_blocks,
)
from duo_workflow_service.entities.state import (
    MAX_TOOL_CALLS_PER_MODEL_CALL,
    ChatWorkflowState,
)
from duo_workflow_service.gitlab.gitlab_api import Namespace, Project
from duo_workflow_service.gitlab.gitlab_instance_info_service import GitLabInstanceInfo
from duo_workflow_service.gitlab.gitlab_service_context import GitLabServiceContext
from duo_workflow_service.slash_commands.error_handler import (
    SlashCommandValidationError,
)
from duo_workflow_service.slash_commands.goal_parser import is_slash_command
from duo_workflow_service.slash_commands.goal_parser import parse as slash_command_parse
from duo_workflow_service.tools.version_compatibility import (
    supports_chat_tool_call_limits,
)
from duo_workflow_service.tools.web_search import WebSearchState
from lib.context import get_model_metadata
from lib.feature_flags.context import FeatureFlag, is_feature_enabled
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
            project, namespace, _kwargs.get("web_search", WebSearchState())
        )
        denied_tools = list(dict.fromkeys(input.get("denied_tools", []) or []))
        system_template_override = _kwargs.get("system_template_override")

        # ``Prompt`` merges the split system messages into one for models that
        # need a single system message (``merges_system_messages``).
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

                if isinstance(m.content, str):
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
                    # Multimodal content: render the template against a text view
                    # of the message, then reattach the non-text blocks — jinja
                    # would flatten them to their repr.
                    text, passthrough_blocks = split_text_blocks(m.content)
                    rendered = jinja2_formatter(
                        cast(str, self.prompt_template["user"]),
                        message=HumanMessage(
                            content=text, additional_kwargs=m.additional_kwargs
                        ),
                        slash_command=slash_command,
                    )
                    # A blocks-only turn can render to whitespace; providers
                    # reject empty text blocks, so omit the block entirely.
                    messages.append(
                        HumanMessage(
                            content=[
                                *non_blank_text_block(rendered),
                                *passthrough_blocks,
                            ]
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
            "model_friendly_name": model_friendly_name(model_metadata),
            "clarification_question_tool_enabled": is_feature_enabled(
                FeatureFlag.DUO_CHAT_CLARIFICATION_QUESTION_TOOL
            ),
        }

    def _build_dynamic_template_context(
        self,
        project: Project | None,
        namespace: Namespace | None,
        web_search: WebSearchState,
    ) -> dict[str, Any]:
        return {
            "current_date": datetime.now().strftime("%Y-%m-%d"),
            "current_time": datetime.now().strftime("%H:%M:%S"),
            "current_timezone": datetime.now().astimezone().tzname(),
            "should_show_current_time": should_show_current_time(self.model_provider),
            "web_search_supported": web_search.supported,
            "web_search_enabled": web_search.active,
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

    @staticmethod
    def _build_tool_call_limit_text() -> str:
        return (
            f"Request at most {MAX_TOOL_CALLS_PER_MODEL_CALL} tool calls in one response. "
            "Calls past that are skipped, so split larger batches across responses."
        )

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

        if (
            is_feature_enabled(FeatureFlag.DUO_CHAT_TOOL_CALL_LIMITS)
            and supports_chat_tool_call_limits()
        ):
            messages.append(SystemMessage(content=self._build_tool_call_limit_text()))

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
