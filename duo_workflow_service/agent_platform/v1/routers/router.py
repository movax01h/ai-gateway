from typing import Annotated, ClassVar, Optional, Self, override

import structlog
from langgraph.graph import StateGraph
from pydantic import model_validator

from duo_workflow_service.agent_platform.utils.exceptions import (
    MissingAdditionalContextError,
    NotifiableAgentException,
)
from duo_workflow_service.agent_platform.v1.components.base import BaseComponent
from duo_workflow_service.agent_platform.v1.routers.base import BaseRouter
from duo_workflow_service.agent_platform.v1.state import FlowState, IOKey
from duo_workflow_service.monitoring import duo_workflow_metrics
from lib.events import GLReportingEventContext
from lib.internal_events import InternalEventAdditionalProperties
from lib.internal_events.ai_context import AIContext
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.event_enum import EventEnum

__all__ = ["Router"]

log = structlog.stdlib.get_logger("router")


class Router(BaseRouter):
    from_component: BaseComponent
    to_component: BaseComponent | dict[str | int | bool, BaseComponent]
    flow_id: Optional[str] = None
    flow_type: Optional[GLReportingEventContext] = None
    agent_name: Optional[str] = None
    internal_event_client: Optional[InternalEventsClient] = None

    _allowed_input_targets: ClassVar[tuple[str, ...]] = ("context", "status")

    @model_validator(mode="after")
    def validate_router_fields(self) -> Self:
        if self.input is None and not isinstance(self.to_component, BaseComponent):
            raise ValueError(
                "If input is None, then to_component must be a BaseComponent"
            )

        if self.input is not None and not isinstance(self.to_component, dict):
            raise ValueError("If input is not None, then to_component must be a dict")

        return self

    @override
    def attach(self, graph: StateGraph):
        self.from_component.attach(graph, self)

    @override
    def route(
        self, state: FlowState
    ) -> Annotated[str, "Next component entry hook node"]:
        if self.input is None:
            return self.to_component.__entry_hook__()  # type: ignore[union-attr]

        try:
            route_value = str(self.input.value_from_state(state))
        except MissingAdditionalContextError:
            # The caller did not supply a required additional context category.
            # That is an invalid request rather than a flow fault, and the server
            # already maps this to INVALID_ARGUMENT, so it must reach the server
            # unchanged instead of being reshaped into an internal error here.
            raise
        except KeyError as err:
            raise self._missing_context_key_error(self.input, err) from err

        is_default_route = route_value not in self.to_component

        self._track_route_decision(route_value, is_default_route=is_default_route)

        if not is_default_route:
            return self.to_component[route_value].__entry_hook__()  # type: ignore[index]

        if BaseRouter.DEFAULT_ROUTE in self.to_component:
            return self.to_component[BaseRouter.DEFAULT_ROUTE].__entry_hook__()  # type: ignore[index]

        raise KeyError(
            f"Route key {self.input} not found in conditions {self.to_component}"
        )

    def _missing_context_key_error(
        self, input_key: IOKey, err: KeyError
    ) -> NotifiableAgentException:
        """Turn a bare ``KeyError`` from a context lookup into a domain error.

        A router decides the next node by reading a value out of the flow
        context (for example ``context:classify_impact.execution_result``). When
        the producing node never wrote that value, ``IOKey.value_from_state``
        raises a raw ``KeyError`` that reaches the end user as an opaque
        ``KeyError: 'execution_result'``. Re-raise it as a
        :class:`NotifiableAgentException` whose safe message names the flow node
        and the missing key in the documented ``context:{node}.{key}`` form,
        while keeping the original error server-side for engineers.

        ``context:inputs.*`` is not written by a node, so blaming a step called
        ``inputs`` would point the user at something that does not exist. A
        missing additional context *category* raises
        :class:`MissingAdditionalContextError` and is re-raised unchanged by
        :meth:`route`; the message here covers the remaining ``inputs`` shapes
        and describes them as caller-supplied input.
        """
        subkeys = input_key.subkeys or []
        reads_caller_input = bool(subkeys) and subkeys[0] == "inputs"
        node = (
            self.from_component.name
            if reads_caller_input or not subkeys
            else subkeys[0]
        )
        context_key = (
            f"{input_key.target}:{'.'.join(subkeys)}" if subkeys else input_key.target
        )
        detail = str(err.args[0] if err.args else err)

        log.error(
            "Flow route failed: missing context key",
            component_name=self.from_component.name,
            node=node,
            context_key=context_key,
            flow_id=self.flow_id,
            flow_type=self.flow_type.value if self.flow_type else "unknown",
            error=detail,
        )

        if reads_caller_input:
            ui_message = (
                "An internal error occurred: the flow could not continue because "
                f"the required input '{context_key}' was not supplied."
            )
        else:
            ui_message = (
                "An internal error occurred: the flow could not continue because the "
                f"step '{node}' did not produce the expected output '{context_key}'."
            )

        return NotifiableAgentException(ui_message, internal_detail=detail)

    def _track_route_decision(self, route_value: str, is_default_route: bool) -> None:
        component_name = self.from_component.name
        flow_type_value = self.flow_type.value if self.flow_type else "unknown"
        logged_route_value = "invalid_route" if is_default_route else route_value

        log.info(
            "Flow route decision",
            component_name=component_name,
            route_value=logged_route_value,
            flow_id=self.flow_id,
            flow_type=flow_type_value,
            is_default_route=is_default_route,
        )

        duo_workflow_metrics.count_flow_route_decision(
            flow_type=flow_type_value,
            component_name=component_name,
            route_value=logged_route_value,
            is_default_route=is_default_route,
        )

        if self.internal_event_client and self.flow_type:
            additional_properties = InternalEventAdditionalProperties(
                label=component_name,
                property=logged_route_value,
                value=self.flow_id,
                is_default_route=is_default_route,
            )
            ai_context = AIContext(
                workflow_id=self.flow_id,
                flow_type=flow_type_value,
                agent_name=self.agent_name,
            )
            self.internal_event_client.track_event(
                event_name=EventEnum.WORKFLOW_ROUTE_DECISION.value,
                additional_properties=additional_properties,
                category=flow_type_value,
                ai_context=ai_context,
            )
