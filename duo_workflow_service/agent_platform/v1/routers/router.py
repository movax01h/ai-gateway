from typing import Annotated, ClassVar, Self

from langgraph.graph import StateGraph
from pydantic import model_validator

from duo_workflow_service.agent_platform.v1.components import BaseComponent
from duo_workflow_service.agent_platform.v1.routers.base import BaseRouter
from duo_workflow_service.agent_platform.v1.state import FlowState

__all__ = ["Router"]


class Router(BaseRouter):
    from_component: BaseComponent
    to_component: BaseComponent | dict[str | int | bool, BaseComponent]

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

    def attach(self, graph: StateGraph):
        self.from_component.attach(graph, self)

    def route(
        self, state: FlowState
    ) -> Annotated[str, "Next component entry hook node"]:
        if self.input is None:
            return self.to_component.__entry_hook__()  # type: ignore[union-attr]

        route_value = str(self.input.value_from_state(state))

        if route_value in self.to_component:
            return self.to_component[route_value].__entry_hook__()  # type: ignore[index]

        if BaseRouter.DEFAULT_ROUTE in self.to_component:
            return self.to_component[BaseRouter.DEFAULT_ROUTE].__entry_hook__()  # type: ignore[index]

        raise KeyError(
            f"Route key {self.input} not found in conditions {self.to_component}"
        )
