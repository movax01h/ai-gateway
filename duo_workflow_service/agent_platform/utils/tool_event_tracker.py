import json
from typing import Any

from langchain_core.tools import BaseTool, ToolException
from pydantic_core import ValidationError

from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.tracking.tool_governance import track_tool_governance_event
from lib.context import client_capabilities, tool_executions
from lib.events import GLReportingEventContext
from lib.internal_events import InternalEventAdditionalProperties, InternalEventsClient
from lib.internal_events.event_enum import EventEnum, EventLabelEnum

__all__ = ["ToolEventTracker"]

# The tool whose successful result means a new merge request now exists,
# and the key it wraps the GitLab API response
CREATE_MERGE_REQUEST_TOOL_NAME = "create_merge_request"
_CREATED_MR_RESPONSE_KEY = "created_merge_request"


class ToolEventTracker:
    """Class that centralises tool event tracking and error formatting."""

    def __init__(
        self,
        flow_id: str,
        flow_type: GLReportingEventContext,
        internal_event_client: InternalEventsClient,
    ):

        self._flow_id = flow_id
        self._flow_type = flow_type
        self._internal_event_client = internal_event_client

    def track_internal_event(
        self,
        event_name: EventEnum,
        tool_name: str,
        extra: dict | None = None,
    ) -> None:
        """Track an internal event and record the corresponding Prometheus metric."""
        extra = {
            **(extra or {}),
            "client_capabilities": list(client_capabilities.get()),
        }

        additional_properties = InternalEventAdditionalProperties(
            label=EventLabelEnum.WORKFLOW_TOOL_CALL_LABEL.value,
            property=tool_name,
            value=self._flow_id,
            **extra,
        )
        if event_name == EventEnum.WORKFLOW_TOOL_SUCCESS:
            current = tool_executions.get()
            if current is not None:
                current.append(tool_name)
        self._record_metric(
            event_name=event_name,
            additional_properties=additional_properties,
        )
        self._internal_event_client.track_event(
            event_name=event_name.value,
            additional_properties=additional_properties,
            category=self._flow_type.value,
        )

    def track_merge_request_created(self, tool_name: str, tool_response: Any) -> None:
        """Record which session opened a merge request, and which merge request it was."""

        if tool_name != CREATE_MERGE_REQUEST_TOOL_NAME:
            return

        try:
            payload = (
                tool_response
                if isinstance(tool_response, dict)
                else json.loads(tool_response)
            )
            merge_request = payload[_CREATED_MR_RESPONSE_KEY]
            merge_request_id = merge_request["id"]
        except (json.JSONDecodeError, KeyError, TypeError):
            return

        additional_properties = InternalEventAdditionalProperties(
            label=tool_name,
            property=str(merge_request_id),
            value=self._flow_id,
            merge_request_iid=merge_request.get("iid"),
            created_merge_request_url=merge_request.get("web_url"),
            source_branch=merge_request.get("source_branch"),
            target_branch=merge_request.get("target_branch"),
        )
        self._internal_event_client.track_event(
            event_name=EventEnum.WORKFLOW_MERGE_REQUEST_CREATED.value,
            additional_properties=additional_properties,
            category=self._flow_type.value,
        )

    def track_tool_governance_event(
        self,
        event_name: EventEnum,
        tool_name: str | None = None,
        outcome: str | None = None,
    ) -> None:
        """Track a tool-governance internal event (approval requested/resolved, denied tool blocked)."""
        track_tool_governance_event(
            self._internal_event_client,
            event_name,
            flow_type=self._flow_type.value,
            flow_id=self._flow_id,
            tool_name=tool_name,
            outcome=outcome,
        )

    def _record_metric(
        self,
        event_name: EventEnum,
        additional_properties: InternalEventAdditionalProperties,
    ) -> None:
        """Record Prometheus metrics for tool failures."""
        if event_name == EventEnum.WORKFLOW_TOOL_FAILURE:
            tool_name = additional_properties.property or "unknown"
            failure_reason = additional_properties.extra.get("error_type", "unknown")
            duo_workflow_metrics.count_agent_platform_tool_failure(
                flow_type=self._flow_type.value,
                tool_name=tool_name,
                failure_reason=failure_reason,
            )

    def handle_type_error_response(self, tool: BaseTool, error: TypeError) -> str:
        """Handle a TypeError response and track the failure event."""
        if tool.args_schema:
            schema = f"The schema is: {tool.args_schema.model_json_schema()}"  # type: ignore[union-attr]
        else:
            schema = "The tool does not accept any argument"

        response = (
            f"Tool {tool.name} execution failed due to wrong arguments."
            f" You must adhere to the tool args schema! {schema}"
        )

        self.track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_FAILURE,
            tool_name=tool.name,
            extra={
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

        return response

    def handle_validation_error(
        self,
        tool_name: str,
        error: ValidationError,
    ) -> str:
        """Handle a ValidationError response and track the failure event."""
        self.track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_FAILURE,
            tool_name=tool_name,
            extra={
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )
        return f"Tool {tool_name} raised validation error {error!s}"

    def handle_execution_error(
        self,
        tool_name: str,
        error: Exception,
    ) -> str:
        """Handle a generic execution error response and track the failure event."""
        self.track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_FAILURE,
            tool_name=tool_name,
            extra={
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )

        return f"Tool runtime exception due to {error!s} {getattr(error, 'response', None)}"

    def handle_tool_exception(self, tool_name: str, error: ToolException) -> str:
        """Handle a ToolException response and track the failure event."""
        self.track_internal_event(
            event_name=EventEnum.WORKFLOW_TOOL_FAILURE,
            tool_name=tool_name,
            extra={
                "error": str(error),
                "error_type": type(error).__name__,
            },
        )
        return f"Tool exception occurred due to {error!s}"
