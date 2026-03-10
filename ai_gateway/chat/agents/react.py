import re
from typing import Any, AsyncIterator, Optional, Union, cast, override

import starlette_context
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import BaseCumulativeTransformOutputParser
from langchain_core.outputs import Generation
from langchain_core.prompt_values import ChatPromptValue, PromptValue
from langchain_core.runnables import Runnable, RunnableBinding, RunnableConfig

from ai_gateway.chat.agents.typing import (
    AgentError,
    AgentEventType,
    AgentFinalAnswer,
    AgentToolAction,
    AgentUnknownAction,
    ReActAgentInputs,
    TypeAgentEvent,
)
from ai_gateway.model_selection.models import ModelClassProvider
from ai_gateway.models.base_chat import Role
from ai_gateway.prompts import Prompt, jinja2_formatter
from ai_gateway.prompts.config.base import PromptConfig
from ai_gateway.structured_logging import get_request_logger

__all__ = [
    "ReActPlainTextParser",
    "ReActPromptTemplate",
    "ReActAgent",
]


_REACT_AGENT_TOOL_ACTION_CONTEXT_KEY = "duo_chat.agent_tool_action"
_RESPONSE_MAX_TOKENS_WARNING = (
    "**Warning:** Response was incomplete due to token limits. "
    "Please try again with less context."
)

request_log = get_request_logger("react")


class ReActPlainTextParser(BaseCumulativeTransformOutputParser):
    re_thought: re.Pattern = re.compile(
        r"<message>Thought:\s*([\s\S]*?)\s*(?:Action|Final Answer):"
    )
    re_action: re.Pattern = re.compile(r"Action:\s*([\s\S]*?)[\s-]*Action", re.DOTALL)
    re_action_input: re.Pattern = re.compile(r"Action Input:\s*([\s\S]*?)\s*</message>")
    re_final_answer: re.Pattern = re.compile(r"Final Answer:\s*([\s\S]*?)\s*</message>")

    def _parse_final_answer(
        self, message: str, finish_reason: Optional[str]
    ) -> Optional[AgentFinalAnswer]:
        if match_answer := self.re_final_answer.search(message):

            return AgentFinalAnswer(
                text=match_answer.group(1), finish_reason=finish_reason
            )

        return None

    def _parse_agent_action(self, message: str) -> Optional[AgentToolAction]:
        match_action = self.re_action.search(message)
        match_action_input = self.re_action_input.search(message)
        match_thought = self.re_thought.search(message)

        if match_action and match_action_input:
            tool_name = match_action.group(1)
            return AgentToolAction(
                tool=self._modify_tool_name(tool_name),
                tool_input=match_action_input.group(1),
                thought=(
                    match_thought.group(1).replace("\\_", "_") if match_thought else ""
                ),
            )

        return None

    def _modify_tool_name(self, name: str) -> str:
        """Process special case when LLM returns wrong name.

        In some cases LLM could return the name of the Merge Request tool in CamelCase, not in underscore_case. This bug
        was fixed in upstream version of GitLab 17.7 However older GitLab instances could still have this bug. Would be
        cleaned up with
        https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/757
        """
        if name == "MergeRequestReader":
            return "merge_request_reader"

        return name.replace("\\_", "_")

    def _parse(self, text: str, finish_reason: Optional[str]) -> AgentEventType:
        wrapped_text = f"<message>Thought: {text}</message>"

        event: AgentEventType  # Explicit declaration avoids mypy confusion

        if final_answer := self._parse_final_answer(wrapped_text, finish_reason):
            event = final_answer
        elif agent_action := self._parse_agent_action(wrapped_text):
            event = agent_action
        else:
            event = AgentUnknownAction(text=text)

        return event

    def parse_finish_reason(self, meta_data: Optional[dict]) -> Optional[str]:
        if not meta_data:
            return None

        reason = None

        if finish_reason := meta_data.get("finish_reason"):  # Vertex
            reason = finish_reason

        if stop_reason := meta_data.get("stop_reason"):  # Anthropic
            reason = stop_reason
            if stop_reason == "max_tokens":
                reason = "length"  # Convert to Vertex format

        return reason

    @override
    def parse_result(
        self, result: list[Generation], *, partial: bool = False
    ) -> Optional[AgentEventType]:
        event = None
        text = result[0].text.strip()
        message = getattr(result[0], "message", None)
        response_metadata = getattr(message, "response_metadata", None)
        finish_reason = self.parse_finish_reason(response_metadata)

        try:
            event = self._parse(text, finish_reason)
        except ValueError as e:
            if not partial:
                msg = f"Invalid output: {text}"
                raise OutputParserException(msg, llm_output=text) from e

        return event

    @override
    def parse(self, text: str) -> Optional[AgentEventType]:
        return self.parse_result([Generation(text=text)])


class ReActPromptTemplate(Runnable[ReActAgentInputs, PromptValue]):
    def __init__(self, model_provider: ModelClassProvider, config: PromptConfig):
        self.model_provider = model_provider
        self.prompt_template = config.prompt_template
        self.model_config = config.model

    @override
    def invoke(
        self,
        input: ReActAgentInputs,
        config: Optional[RunnableConfig] = None,
        **_kwargs: Any,
    ) -> PromptValue:
        messages: list[BaseMessage] = []

        if "system" in self.prompt_template:
            content = jinja2_formatter(
                self.prompt_template["system"],
                tools=input.tools,
                unavailable_resources=input.unavailable_resources,
                current_date=input.current_date,
            )
            if self.model_provider == ModelClassProvider.ANTHROPIC:
                content_block: list[Union[str, dict]] = [
                    {
                        "text": content,
                        "type": "text",
                        "cache_control": {"type": "ephemeral", "ttl": "1h"},
                    }
                ]
                messages.append(SystemMessage(content=content_block))
            else:
                messages.append(SystemMessage(content=content))

        for m in input.messages:
            if m.role is Role.USER:
                messages.append(
                    HumanMessage(
                        jinja2_formatter(self.prompt_template["user"], message=m)
                    )
                )
            elif m.role is Role.ASSISTANT:
                if m.content is None:
                    continue
                messages.append(
                    AIMessage(
                        jinja2_formatter(
                            self.prompt_template["assistant"],
                            agent_scratchpad=m.agent_scratchpad,
                            final_answer=m.content,
                        )
                    )
                )
            else:
                raise ValueError("Unsupported message")

        if not isinstance(messages[-1], HumanMessage):
            raise ValueError("Last message must be a human message")

        if "assistant" in self.prompt_template:
            messages.append(
                AIMessage(
                    jinja2_formatter(
                        self.prompt_template["assistant"],
                        agent_scratchpad=input.agent_scratchpad,
                    )
                )
            )

        return ChatPromptValue(messages=messages)


class ReActAgent(RunnableBinding[ReActAgentInputs, TypeAgentEvent]):
    RETRYABLE_ERRORS: list[str] = [
        "overloaded_error",
        "AnthropicError - Overloaded",
    ]

    def __init__(self, prompt: Prompt) -> None:
        super().__init__(bound=prompt | ReActPlainTextParser())

    def _append_warning_if_response_exceeded_max_tokens(
        self, event: AgentFinalAnswer
    ) -> AgentFinalAnswer:
        if event.finish_reason == "length":
            event.text = event.text + "\n\n" + _RESPONSE_MAX_TOKENS_WARNING
        return event

    @override
    async def astream(
        self,
        input: ReActAgentInputs,
        config: Optional[RunnableConfig] = None,
        **kwargs: Optional[Any],
    ) -> AsyncIterator[TypeAgentEvent]:
        events: list[TypeAgentEvent] = []
        astream = super().astream(input, config, **kwargs)
        len_final_answer = 0
        agent_final_answer_found = False
        agent_tool_action_found = False

        try:
            async for event in astream:
                request_log.info(
                    "Response streaming", source=__name__, streamed_event=event
                )
                if isinstance(event, AgentToolAction):
                    agent_tool_action_found = True
                elif isinstance(event, AgentFinalAnswer):
                    agent_final_answer_found = True
                    if len(event.text) > 0:
                        response = AgentFinalAnswer(
                            text=event.text[len_final_answer:],
                            finish_reason=event.finish_reason,
                        )
                        response = self._append_warning_if_response_exceeded_max_tokens(
                            response
                        )
                        yield cast(TypeAgentEvent, response)

                        len_final_answer = len(event.text)

                events.append(event)
        except Exception as e:
            error_message = str(e)
            retryable = any(err in error_message for err in self.RETRYABLE_ERRORS)

            yield cast(
                TypeAgentEvent, AgentError(message=error_message, retryable=retryable)
            )
            raise

        if agent_final_answer_found:
            pass  # no-op
        elif agent_tool_action_found:
            agent_tool_action = cast(AgentToolAction, events[-1])
            starlette_context.context[_REACT_AGENT_TOOL_ACTION_CONTEXT_KEY] = (
                agent_tool_action.tool
            )
            yield cast(TypeAgentEvent, events[-1])
        elif isinstance(events[-1], AgentUnknownAction):
            yield cast(TypeAgentEvent, events[-1])
