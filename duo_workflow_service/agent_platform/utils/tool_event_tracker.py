import json
from typing import Any, Optional

from langchain_core.tools import BaseTool, ToolException
from pydantic import ValidationError

from duo_workflow_service.monitoring import duo_workflow_metrics
from duo_workflow_service.tracking.tool_governance import track_tool_governance_event
from lib.context import client_capabilities, tool_executions
from lib.events import GLReportingEventContext
from lib.internal_events import InternalEventAdditionalProperties, InternalEventsClient
from lib.internal_events.event_enum import EventEnum, EventLabelEnum

__all__ = ["ToolEventTracker", "WorkflowEventTracker"]

# The tool whose successful result means a new merge request now exists,
# and the key it wraps the GitLab API response
CREATE_MERGE_REQUEST_TOOL_NAME = "create_merge_request"
_CREATED_MR_RESPONSE_KEY = "created_merge_request"


class WorkflowEventTracker:
    """Reports the internal events of one flow: it knows which flow they belong to and where they go.

    Every event of a flow carries the flow's ID as its value and is filed under the flow's type. The nodes that report
    take one tracker and ask it to report, so none of them repeats that and none has to be handed the three parts.

    Args:
        flow_id: The flow's ID, the value of every event it reports.
        flow_type: The flow's reporting context, whose value is the category every event is filed under.
        internal_event_client: Where the events go.
    """

    def __init__(
        self,
        flow_id: str,
        flow_type: GLReportingEventContext,
        internal_event_client: InternalEventsClient,
    ):

        self._flow_id = flow_id
        self._flow_type = flow_type
        self._internal_event_client = internal_event_client

    @property
    def flow_id(self) -> str:
        """The flow's ID."""
        return self._flow_id

    @property
    def flow_type(self) -> GLReportingEventContext:
        """The flow's reporting context."""
        return self._flow_type

    def track(
        self,
        event_name: EventEnum,
        label: Optional[str],
        property_: str,
        /,
        **extra: Any,
    ) -> None:
        """Report an event of this flow: its ID is the value, and its type is the category.

        The first three arguments are positional-only, so a field of the event can be named like one of them: an agent's
        response schema may have a field called ``event_name`` or ``property_``, and it is sent as an extra.

        Args:
            event_name: The event.
            label: What the event is about, as the event's label. May be absent.
            property_: The event's property.
            **extra: Further fields the event carries.
        """
        self._internal_event_client.track_event(
            event_name=event_name.value,
            additional_properties=InternalEventAdditionalProperties(
                label=label,
                property=property_,
                value=self._flow_id,
                **extra,
            ),
            category=self._flow_type.value,
        )


class ToolEventTracker(WorkflowEventTracker):
    """Class that centralises tool event tracking and error formatting."""

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

        if event_name == EventEnum.WORKFLOW_TOOL_SUCCESS:
            current = tool_executions.get()
            if current is not None:
                current.append(tool_name)
        self._record_metric(event_name=event_name, tool_name=tool_name, extra=extra)
        self.track(
            event_name,
            EventLabelEnum.WORKFLOW_TOOL_CALL_LABEL.value,
            tool_name,
            **extra,
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

        self.track(
            EventEnum.WORKFLOW_MERGE_REQUEST_CREATED,
            tool_name,
            str(merge_request_id),
            merge_request_iid=merge_request.get("iid"),
            created_merge_request_url=merge_request.get("web_url"),
            source_branch=merge_request.get("source_branch"),
            target_branch=merge_request.get("target_branch"),
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
        tool_name: str,
        extra: dict[str, Any],
    ) -> None:
        """Record Prometheus metrics for tool failures."""
        if event_name == EventEnum.WORKFLOW_TOOL_FAILURE:
            tool_name = tool_name or "unknown"
            failure_reason = extra.get("error_type", "unknown")
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
