import json
from typing import Optional, Type

import structlog
from langchain_core.messages import AIMessage, ToolMessage

from ai_gateway.response_schemas.base import BaseAgentOutput
from duo_workflow_service.agent_platform.v1.components.agent.ui_log import (
    UILogEventsAgent,
)
from duo_workflow_service.agent_platform.v1.state import (
    FlowState,
    FlowStateKeys,
    IOKey,
    create_nested_dict,
)
from duo_workflow_service.agent_platform.v1.ui_log import DefaultUILogWriter, UIHistory
from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.tracking.response_schema_tracking_context import (
    response_schema_tracking_results,
)
from lib.events import GLReportingEventContext
from lib.internal_events import InternalEventAdditionalProperties
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.event_enum import EventEnum

__all__ = ["FinalResponseNode"]

log = structlog.stdlib.get_logger("final_response_node")


class FinalResponseNode:
    def __init__(
        self,
        *,
        component_name: str,
        name: str,
        output: Optional[IOKey],
        ui_history: UIHistory[DefaultUILogWriter, UILogEventsAgent],
        response_schema: Optional[Type[BaseAgentOutput]] = None,
        response_schema_tracking: bool = False,
        flow_id: Optional[str] = None,
        flow_type: Optional[GLReportingEventContext] = None,
        internal_event_client: Optional[InternalEventsClient] = None,
    ):
        self._component_name = component_name
        self.name = name
        self._output = output
        self._ui_history = ui_history
        self._response_schema = response_schema
        self._response_schema_tracking = response_schema_tracking
        self._flow_id = flow_id
        self._flow_type = flow_type
        self._internal_event_client = internal_event_client

    async def run(self, state: FlowState) -> dict:
        last_message = self._get_last_ai_message(state)

        if self._response_schema is not None:
            final_response_text, updates = self._extract_structured_response(
                last_message
            )
        else:
            final_response_text, updates = self._extract_text_response(last_message)

        self._ui_history.log.success(
            final_response_text,
            event=UILogEventsAgent.ON_AGENT_FINAL_ANSWER,
        )

        return {**self._ui_history.pop_state_updates(), **updates}

    def _get_last_ai_message(self, state: FlowState) -> AIMessage:
        history = state[FlowStateKeys.CONVERSATION_HISTORY].get(
            self._component_name, []
        )

        if not history:
            raise ValueError(f"No messages found for {self._component_name}")

        last_message = history[-1]

        if not isinstance(last_message, AIMessage):
            raise ValueError(
                f"The last message of {self._component_name} is not of type AIMessage"
            )

        return last_message

    def _extract_structured_response(self, last_message: AIMessage) -> tuple[str, dict]:
        if not last_message.tool_calls:
            raise ValueError(
                f"Response schema requires a tool call but got a text-only response "
                f"for {self._component_name}"
            )

        if len(last_message.tool_calls) > 1:
            raise ValueError(
                f"Too many tool calls found in the last message of {self._component_name}"
            )

        final_response_call = last_message.tool_calls[0]

        if self._response_schema is None:
            raise ValueError(
                "Response schema is required for structured response extraction"
            )
        if final_response_call["name"] != self._response_schema.tool_title:
            raise ValueError(
                f"Final response tool call not found in the conversation history "
                f"of {self._component_name}"
            )

        parsed_response = self._response_schema(**final_response_call["args"])

        if self._response_schema_tracking:
            self._track_response_schema_output(parsed_response)

        updates: dict = {
            FlowStateKeys.CONVERSATION_HISTORY: {
                self._component_name: [
                    ToolMessage(content="", tool_call_id=final_response_call["id"])
                ]
            },
        }

        if self._output:
            output_data = parsed_response.to_output()
            if self._output.subkeys is not None:
                updates[self._output.target] = create_nested_dict(
                    self._output.subkeys, output_data
                )
            else:
                updates[self._output.target] = output_data

        return parsed_response.to_string_output(), updates

    def _extract_text_response(self, last_message: AIMessage) -> tuple[str, dict]:
        final_response_text = last_message.text

        updates: dict = {}

        if self._output:
            if self._output.subkeys is not None:
                updates[self._output.target] = create_nested_dict(
                    self._output.subkeys, final_response_text
                )
            else:
                updates[self._output.target] = final_response_text

        return final_response_text, updates

    def _track_response_schema_output(self, parsed_response: BaseAgentOutput) -> None:
        output_data = parsed_response.to_output()
        flow_type_value = self._flow_type.value if self._flow_type else "unknown"

        log.info(
            "Response schema output tracked",
            component_name=self._component_name,
            flow_id=self._flow_id,
            flow_type=flow_type_value,
            response_output=output_data,
        )

        duo_workflow_metrics.count_response_schema_output(
            flow_type=flow_type_value,
            component_name=self._component_name,
        )

        if self._internal_event_client and self._flow_type:
            extra = {}
            if isinstance(output_data, dict):
                extra = {field: str(value) for field, value in output_data.items()}

            additional_properties = InternalEventAdditionalProperties(
                label=self._component_name,
                property=json.dumps(output_data, default=str),
                value=self._flow_id,
                **extra,
            )
            self._internal_event_client.track_event(
                event_name=EventEnum.WORKFLOW_RESPONSE_SCHEMA_OUTPUT.value,
                additional_properties=additional_properties,
                category=flow_type_value,
            )

        # Store in ContextVar for access by the checkpointer at workflow completion.
        # Mutate the dict in-place so changes are visible to the parent async task
        # (ContextVar.set() in a child task only affects that task's context copy).
        current_results = response_schema_tracking_results.get()
        if current_results is not None:
            current_results[self._component_name] = output_data
