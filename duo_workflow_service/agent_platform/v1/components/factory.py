"""AgentComponent factory for the Flow Registry.

Registered in the v1 :class:`ComponentRegistry` under ``"AgentComponent"``.
Transparently dispatches to :class:`AgentComponent`,
:class:`SupervisorAgentComponent`, or (when the ``dap_parallel_subagents``
feature flag is enabled) :class:`SupervisorAgentComponentV2` depending on
whether ``subagents`` is present in the component configuration, so flow YAML
configs always use ``type: AgentComponent`` regardless of mode.

Note: This module must be imported **after** ``agent.component``,
``supervisor.component``, and ``supervisor_v2.component`` have been loaded
(as ``__init__.py`` ensures) so that the module-level imports below do not
create circular dependencies.
"""

from typing import Any

from duo_workflow_service.agent_platform.v1.components.agent.component import (
    AgentComponent,
    AgentComponentBase,
)
from duo_workflow_service.agent_platform.v1.components.base import (
    BaseComponent,
)
from duo_workflow_service.agent_platform.v1.components.registry import (
    register_component_factory,
)
from duo_workflow_service.agent_platform.v1.components.supervisor.component import (
    SupervisorAgentComponent,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.component import (
    SupervisorAgentComponentV2,
)
from lib.feature_flags.context import FeatureFlag, is_feature_enabled

__all__ = ["agent_component_factory"]


@register_component_factory("AgentComponent")
def agent_component_factory(
    **kwargs: Any,
) -> AgentComponentBase:
    """Dispatch to AgentComponent, SupervisorAgentComponent, or SupervisorAgentComponentV2.

    Creates a :class:`SupervisorAgentComponent` (or :class:`SupervisorAgentComponentV2`,
    see below) when ``subagents`` is present and non-empty, passing the
    ``_built_components`` dict injected by the flow builder as
    ``subagent_components``.  Otherwise creates a plain :class:`AgentComponent`.
    ``_built_components`` is popped from ``kwargs`` before forwarding to the
    constructor, since it is not a field of any of the three component classes.

    Sequential vs. parallel delegation is chosen solely by the
    ``dap_parallel_subagents`` feature flag: every component that declares
    ``subagents`` delegates concurrently once the flag is on. The flag makes the
    rollout controllable per instance/group and acts as a kill switch that does
    not require shipping a new flow config revision. Both supervisors accept the
    same config, so switching modes never requires a config change.

    Args:
        **kwargs: Component constructor arguments from the flow YAML, plus
            ``_built_components`` injected by the flow builder.

    Returns:
        An :class:`AgentComponent`, :class:`SupervisorAgentComponent`, or
        :class:`SupervisorAgentComponentV2` instance.
    """
    built_components: dict[str, BaseComponent] = kwargs.pop("_built_components", {})

    if kwargs.get("subagents"):
        if is_feature_enabled(FeatureFlag.DAP_PARALLEL_SUBAGENTS):
            return SupervisorAgentComponentV2(
                subagent_components=built_components, **kwargs
            )
        return SupervisorAgentComponent(subagent_components=built_components, **kwargs)

    return AgentComponent(**kwargs)
