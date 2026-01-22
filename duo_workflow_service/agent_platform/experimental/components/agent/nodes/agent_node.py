from typing import ClassVar, Self, cast

import structlog
from anthropic import APIStatusError
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import ValidationError

from ai_gateway.instrumentators.model_requests import LLMFinishReason
from ai_gateway.prompts import Prompt
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    FlowStateKeys,
    IOKey,
    get_vars_from_state,
)
from duo_workflow_service.errors.error_handler import ModelError, ModelErrorHandler
from lib.events import GLReportingEventContext
from lib.internal_events import InternalEventsClient

__all__ = ["AgentNode", "AgentFinalOutput"]

log = structlog.stdlib.get_logger("agent_node")


class AgentFinalOutput(BaseModel):
    """
    MANDATORY COMPLETION TOOL: You MUST use this tool to provide your final answer
    when you have completed the user's request. This is the ONLY way to properly
    end the conversation and deliver your response to the user.
    Use this tool when:
    1. You have gathered all necessary information
    2. You have completed the requested task or analysis
    3. You are ready to give your final answer
    CRITICAL: Always use this tool to conclude your work - do not continue
    using other tools once you have the information needed to answer the user.
    The final_response must contain your complete answer.
    """

    model_config = ConfigDict(title="final_response_tool", frozen=True)

    tool_title: ClassVar[str] = "final_response_tool"

    final_response: str = Field(
        description="The final response to the user to communicate work completion"
    )

    @classmethod
    def from_ai_message(cls, ai_message: AIMessage) -> Self:
        """Generate an AgentFinalOutput from an AI message."""
        return cls(**ai_message.tool_calls[0]["args"])


class AgentNode:
    name: str
    _prompt: Prompt

    _inputs: list[IOKey]

    _component_name: str

    _internal_event_client: InternalEventsClient

    _flow_id: str
    _flow_type: GLReportingEventContext
    _error_handler: ModelErrorHandler

    def __init__(
        self,
        flow_id: str,
        flow_type: GLReportingEventContext,
        name: str,
        prompt: Prompt,
        inputs: list[IOKey],
        component_name: str,
        internal_event_client: InternalEventsClient,
    ):
        self._flow_id = flow_id
        self._flow_type = flow_type
        self.name = name
        self._prompt = prompt
        self._inputs = inputs
        self._component_name = component_name
        self._internal_event_client = internal_event_client
        self._error_handler = ModelErrorHandler()

    async def run(self, state: FlowState) -> dict:
        history = state[FlowStateKeys.CONVERSATION_HISTORY].get(
            self._component_name, []
        )
        variables = get_vars_from_state(self._inputs, state)

        while True:
            try:
                completion: AIMessage = cast(
                    AIMessage,
                    await self._prompt.ainvoke(input={**variables, "history": history}),
                )
                finish_reason = completion.response_metadata.get("finish_reason")
                if finish_reason in LLMFinishReason.abnormal_values():
                    log.warning(f"LLM stopped abnormally with reason: {finish_reason}")

                if len(updates := self._final_answer_validate(completion)) > 0:
                    history = [*history, *updates]
                    continue

                # Append new completion to existing history for replace-based reducer.
                # The reducer will replace this component's conversation history with
                # the complete list returned here.
                return {
                    FlowStateKeys.CONVERSATION_HISTORY: {
                        self._component_name: history + [completion]
                    },
                }
            except APIStatusError as e:
                error_message = str(e)
                status_code = e.response.status_code
                model_error = ModelError(
                    error_type=self._error_handler.get_error_type(status_code),
                    status_code=status_code,
                    message=error_message,
                )

                await self._error_handler.handle_error(model_error)

    def _final_answer_validate(self, completion: AIMessage) -> list:
        final_answer = next(
            (
                tool_call
                for tool_call in completion.tool_calls
                if tool_call["name"] == AgentFinalOutput.tool_title
            ),
            None,
        )

        if not final_answer:
            return []

        if len(completion.tool_calls) > 1:
            return [completion] + [
                ToolMessage(
                    content=f"{AgentFinalOutput.tool_title} mustn't be combined with other tool calls",
                    tool_call_id=tool_call["id"],
                )
                for tool_call in completion.tool_calls
            ]

        try:
            AgentFinalOutput.from_ai_message(completion)
            return []
        except ValidationError as ve:
            return [
                completion,
                ToolMessage(
                    content=f"{AgentFinalOutput.tool_title} raised validation error: {ve}",
                    tool_call_id=final_answer["id"],
                ),
            ]
