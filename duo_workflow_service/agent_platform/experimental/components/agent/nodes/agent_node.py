from typing import ClassVar, Self

import structlog
from anthropic import APIStatusError
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel, ConfigDict, Field
from pydantic_core import ValidationError

from ai_gateway.prompts import Prompt
from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    FlowStateKeys,
    IOKey,
    get_vars_from_state,
)
from duo_workflow_service.errors.error_handler import ModelError, ModelErrorHandler
from duo_workflow_service.llm_factory import LLMFinishReason
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.token_counter.approximate_token_counter import (
    ApproximateTokenCounter,
)
from lib.internal_events import InternalEventAdditionalProperties, InternalEventsClient
from lib.internal_events.event_enum import CategoryEnum, EventEnum, EventPropertyEnum

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
    _approximate_token_counter: ApproximateTokenCounter

    _flow_id: str
    _flow_type: CategoryEnum
    _error_handler: ModelErrorHandler

    def __init__(
        self,
        flow_id: str,
        flow_type: CategoryEnum,
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
        self._approximate_token_counter = ApproximateTokenCounter(component_name)
        self._error_handler = ModelErrorHandler()

    async def run(self, state: FlowState) -> dict:
        history = state[FlowStateKeys.CONVERSATION_HISTORY].get(
            self._component_name, []
        )
        variables = get_vars_from_state(self._inputs, state)
        model_name = getattr(self._prompt.model, "model_name", "unknown")
        request_type = f"{self._component_name}_completion"
        model_provider = self._prompt.model_provider

        while True:
            try:
                with duo_workflow_metrics.time_llm_request(
                    model=model_name, request_type=request_type
                ):
                    completion: AIMessage = await self._prompt.ainvoke(
                        input={**variables, "history": history}
                    )
                    finish_reason = completion.response_metadata.get("finish_reason")
                    if finish_reason in LLMFinishReason.abnormal_values():
                        log.warning(
                            f"LLM stopped abnormally with reason: {finish_reason}"
                        )
                self._track_tokens_data(completion, history)
                duo_workflow_metrics.count_llm_response(
                    model=model_name,
                    provider=model_provider,
                    request_type=request_type,
                    stop_reason=finish_reason,
                    # Hardcoded 200 status since model_completion only returns status codes for failures
                    status_code="200",
                    error_type="none",
                )

                if len(updates := self._final_answer_validate(completion)) > 0:
                    history = [*history, *updates]
                    continue

                return {
                    FlowStateKeys.CONVERSATION_HISTORY: {
                        self._component_name: [completion]
                    },
                }
            except APIStatusError as e:
                error_message = str(e)
                status_code = e.response.status_code
                duo_workflow_metrics.count_llm_response(
                    model=model_name,
                    provider=model_provider,
                    request_type=request_type,
                    stop_reason="error",
                    status_code=e.response.status_code,
                    error_type=self._error_handler.get_error_type(status_code),
                )
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

    def _track_tokens_data(self, message, history):
        estimated = self._approximate_token_counter.count_tokens(history)
        usage_metadata = message.usage_metadata if message.usage_metadata else {}

        additional_properties = InternalEventAdditionalProperties(
            label=self._component_name,
            property=EventPropertyEnum.WORKFLOW_ID.value,
            value=self._flow_id,
            input_tokens=usage_metadata.get("input_tokens"),
            output_tokens=usage_metadata.get("output_tokens"),
            total_tokens=usage_metadata.get("total_tokens"),
            estimated_input_tokens=estimated,
        )
        self._internal_event_client.track_event(
            event_name=EventEnum.TOKEN_PER_USER_PROMPT.value,
            additional_properties=additional_properties,
            category=self._flow_type.value,
        )
