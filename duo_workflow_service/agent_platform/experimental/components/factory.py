"""AgentComponent factory for the experimental Flow Registry.

This module provides a factory that is registered in the ``ComponentRegistry``
under the name ``"AgentComponent"``.  The factory transparently dispatches to
either :class:`AgentComponent` or :class:`SupervisorAgentComponent` depending
on whether the ``managed_agents`` keyword argument is present in the component
configuration.

This allows flow authors to use a single ``type: AgentComponent`` declaration
for both standalone agents and supervisor agents, reducing the perceived
complexity of the framework.  When ``managed_agents`` is present and non-empty
the factory creates a :class:`SupervisorAgentComponent`; otherwise it creates
a plain :class:`AgentComponent`.

When creating a :class:`SupervisorAgentComponent`, the factory passes the
shared ``_built_components`` dict (injected by the flow builder) as
``subagent_components``.  All subagent-selection and validation logic is
centralised in
:meth:`SupervisorAgentComponent.validate_and_consume_managed_agents`, which
filters the pool to only the agents named in ``managed_agents``.

The factory does **not** mutate ``_built_components`` — removing consumed
subagents from the shared dict is the responsibility of the flow builder
(``Flow._instantiate_component``), which detects consumed components via the
``subagent_components`` attribute on the created component.

Note: This module must be imported **after** both ``agent.component`` and
``supervisor.component`` have been loaded (as ``__init__.py`` ensures) so that
the module-level imports below do not create circular dependencies.
"""

from typing import Any

from duo_workflow_service.agent_platform.experimental.components.agent.component import (
    AgentComponent,
    AgentComponentBase,
)
from duo_workflow_service.agent_platform.experimental.components.base import (
    BaseComponent,
)
from duo_workflow_service.agent_platform.experimental.components.registry import (
    register_component_factory,
)
from duo_workflow_service.agent_platform.experimental.components.supervisor.component import (
    SupervisorAgentComponent,
)

__all__ = ["agent_component_factory"]


@register_component_factory("AgentComponent")
def agent_component_factory(**kwargs: Any) -> AgentComponentBase:
    """Dispatch to AgentComponent or SupervisorAgentComponent.

    Inspects the ``managed_agents`` keyword argument to decide which concrete
    component class to instantiate:

    - If ``managed_agents`` is present (and non-empty): SupervisorAgentComponent.
        The factory passes the shared ``_built_components`` dict as
        ``subagent_components`` so that
        :meth:`SupervisorAgentComponent.validate_and_consume_managed_agents`
        can select and validate the named subagents.  The dict is passed
        read-only — removing consumed subagents is the caller's responsibility.
    - Otherwise: plain AgentComponent.

    All remaining keyword arguments are forwarded to the chosen class.

    The ``_built_components`` key is extracted from ``kwargs`` before forwarding
    so it is never passed to the component constructors as an unexpected field.

    Args:
        **kwargs: Component constructor arguments as parsed from the flow YAML,
            plus ``_built_components`` injected by the flow builder.
            The ``managed_agents`` key is the discriminator.

    Returns:
        The created :class:`AgentComponentBase` instance (either
        :class:`AgentComponent` or :class:`SupervisorAgentComponent`).
    """
    built_components: dict[str, BaseComponent] = kwargs.pop("_built_components", {})

    if kwargs.get("managed_agents"):
        return SupervisorAgentComponent(subagent_components=built_components, **kwargs)

    return AgentComponent(**kwargs)
