"""Builder for ``HistoryOptimizerPipeline`` from typed feature flags."""

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
    *,
    compaction: CompactionConfig | bool = False,
    flow_context: FlowContext,
    agent_name: str,
    prompt_registry: BasePromptRegistry = Provide[
        ContainerApplication.pkg_prompts.prompt_registry
    ],
    internal_events_client: InternalEventsClient = Provide[
        ContainerApplication.internal_event.client
    ],
) -> HistoryOptimizerPipeline:
    """Construct a ``HistoryOptimizerPipeline`` from typed opt-in flags.

    Framework-enforced invariant: the main optimizer is compaction if enabled,
    otherwise the legacy trim safety net. Order between future pre-main
    optimizers (e.g., tool-result pruning) is encoded structurally in this
    function's body -- flow authors do not choose order.
    """
    optimizers: list[HistoryOptimizer] = []

    # Pre-main optimizers (e.g., ToolResultPruner) must run before the main
    # optimizer so that the main optimizer sees already-pruned messages and
    # its token-count thresholds remain accurate. Encoding this order here,
    # rather than letting flow authors compose lists, prevents accidental
    # inversion that would cause the main optimizer to compact messages that
    # would have been pruned anyway.

    if compaction:
        cfg = (
            compaction
            if isinstance(compaction, CompactionConfig)
            else CompactionConfig()
        )
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
    else:
        optimizers.append(
            LegacyTrimOptimizer(
                agent_name=agent_name,
                internal_events_client=internal_events_client,
            )
        )

    return HistoryOptimizerPipeline(optimizers)
