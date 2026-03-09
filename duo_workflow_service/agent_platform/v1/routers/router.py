from typing import Annotated, ClassVar, Optional, Self, override

import structlog
from langgraph.graph import StateGraph
from pydantic import model_validator

from duo_workflow_service.agent_platform.v1.components import BaseComponent
from duo_workflow_service.agent_platform.v1.routers.base import BaseRouter
from duo_workflow_service.agent_platform.v1.state import FlowState
from duo_workflow_service.monitoring import duo_workflow_metrics
from lib.events import GLReportingEventContext
from lib.internal_events import InternalEventAdditionalProperties
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.event_enum import EventEnum

__all__ = ["Router"]

log = structlog.stdlib.get_logger("router")


class Router(BaseRouter):
    from_component: BaseComponent
    to_component: BaseComponent | dict[str | int | bool, BaseComponent]
    flow_id: Optional[str] = None
    flow_type: Optional[GLReportingEventContext] = None
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

        route_value = str(self.input.value_from_state(state))
        is_default_route = route_value not in self.to_component

        self._track_route_decision(route_value, is_default_route=is_default_route)

        if not is_default_route:
            return self.to_component[route_value].__entry_hook__()  # type: ignore[index]

        if BaseRouter.DEFAULT_ROUTE in self.to_component:
            return self.to_component[BaseRouter.DEFAULT_ROUTE].__entry_hook__()  # type: ignore[index]

        raise KeyError(
            f"Route key {self.input} not found in conditions {self.to_component}"
        )

    def _track_route_decision(self, route_value: str, is_default_route: bool) -> None:
        component_name = self.from_component.name
        flow_type_value = self.flow_type.value if self.flow_type else "unknown"

        log.info(
            "Flow route decision",
            component_name=component_name,
            route_value=route_value,
            flow_id=self.flow_id,
            flow_type=flow_type_value,
            is_default_route=is_default_route,
        )

        duo_workflow_metrics.count_flow_route_decision(
            flow_type=flow_type_value,
            component_name=component_name,
            route_value=route_value,
            is_default_route=is_default_route,
        )

        if self.internal_event_client and self.flow_type:
            additional_properties = InternalEventAdditionalProperties(
                label=component_name,
                property=route_value,
                value=self.flow_id,
                is_default_route=is_default_route,
            )
            self.internal_event_client.track_event(
                event_name=EventEnum.WORKFLOW_ROUTE_DECISION.value,
                additional_properties=additional_properties,
                category=flow_type_value,
            )
