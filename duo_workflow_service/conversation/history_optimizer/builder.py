"""Builder for ``HistoryOptimizerPipeline`` from a list of configs."""

from dataclasses import dataclass

from dependency_injector.wiring import Provide, inject
from gitlab_cloud_connector import CloudConnectorUser

from ai_gateway.container import ContainerApplication
from ai_gateway.prompts.base import BasePromptRegistry
from duo_workflow_service.conversation.history_optimizer.base import HistoryOptimizer
from duo_workflow_service.conversation.history_optimizer.optimizers.compaction import (
    CompactionOptimizer,
)
from duo_workflow_service.conversation.history_optimizer.optimizers.legacy_trim import (
    LegacyTrimOptimizer,
)
from duo_workflow_service.conversation.history_optimizer.pipeline import (
    HistoryOptimizerPipeline,
)
from duo_workflow_service.conversation.history_optimizer.schema import (
    CompactionConfig,
    HistoryOptimizerConfig,
    LegacyTrimConfig,
    ToolResultPrunerConfig,
)
from duo_workflow_service.conversation.history_optimizer.validation import (
    validate_history_optimizer_configs,
)
from lib.context import StarletteUser
from lib.internal_events.client import InternalEventsClient


@dataclass(frozen=True)
class FlowContext:
    """Flow-level context required to build optimizers.

    Bundles flow_id, flow_type, and user so the builder's call sites stay short and a future context-var / request-
    scoped resolver can replace this struct without churning every call site.
    """

    flow_id: str
    flow_type: str
    user: StarletteUser | CloudConnectorUser


@inject
def build_history_optimizer_pipeline(
    configs: list[HistoryOptimizerConfig],
    *,
    flow_context: FlowContext,
    agent_name: str,
    prompt_registry: BasePromptRegistry = Provide[
        ContainerApplication.pkg_prompts.prompt_registry
    ],
    internal_events_client: InternalEventsClient = Provide[
        ContainerApplication.internal_event.client
    ],
) -> HistoryOptimizerPipeline:
    """Construct a ``HistoryOptimizerPipeline`` from a list of configs.

    Validates the config list (see ``validate_history_optimizer_configs``),
    then instantiates one optimizer per config in the same order. The
    ``prompt_registry`` and ``internal_events_client`` dependencies are
    DI-injected from ``ContainerApplication``.

    Args:
        configs: Ordered list of optimizer configs. An empty list yields an
            empty pipeline; callers typically default to
            ``[LegacyTrimConfig()]`` upstream.
        flow_context: Flow-level identifiers passed to optimizers.
        agent_name: Agent name forwarded to optimizers for logging and
            telemetry.
        prompt_registry: Injected. Used by ``CompactionOptimizer``.
        internal_events_client: Injected. Used by both optimizers.

    Returns:
        A configured ``HistoryOptimizerPipeline``.

    Raises:
        NotImplementedError: When a ``ToolResultPrunerConfig`` is provided
            (the pruner is stubbed in MR 1).
        ValueError: When validation fails or an unknown config type appears.
    """
    validate_history_optimizer_configs(configs)

    optimizers: list[HistoryOptimizer] = []
    for cfg in configs:
        if isinstance(cfg, CompactionConfig):
            optimizers.append(
                CompactionOptimizer(
                    config=cfg,
                    prompt_registry=prompt_registry,
                    user=flow_context.user,
                    agent_name=agent_name,
                    workflow_id=flow_context.flow_id,
                    workflow_type=flow_context.flow_type,
                    internal_events_client=internal_events_client,
                )
            )
        elif isinstance(cfg, LegacyTrimConfig):
            optimizers.append(
                LegacyTrimOptimizer(
                    agent_name=agent_name,
                    internal_events_client=internal_events_client,
                )
            )
        elif isinstance(cfg, ToolResultPrunerConfig):
            raise NotImplementedError(
                "ToolResultPruner is stubbed; implement in a follow-up MR."
            )
        else:
            raise ValueError(
                f"Unknown HistoryOptimizerConfig type: {type(cfg).__name__}"
            )

    return HistoryOptimizerPipeline(optimizers)
