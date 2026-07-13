from gitlab_cloud_connector import CloudConnectorUser

from ai_gateway.prompts import BasePromptRegistry, Prompt
from duo_workflow_service.agents.chat_agent import ChatAgent
from duo_workflow_service.agents.prompt_adapter import DefaultPromptAdapter
from duo_workflow_service.client_capabilities import is_client_capable
from duo_workflow_service.components.tools_registry import Toolset, ToolsRegistry
from duo_workflow_service.conversation.history_optimizer.builder import (
    FlowContext,
    build_history_optimizer_pipeline,
)
from duo_workflow_service.conversation.history_optimizer.optimizers.compaction import (
    CompactionOptimizer,
)
from duo_workflow_service.conversation.history_optimizer.pipeline import (
    HistoryOptimizerPipeline,
)
from duo_workflow_service.conversation.history_optimizer.schema import CompactionConfig
from lib.events import GLReportingEventContext
from lib.feature_flags.context import FeatureFlag, is_feature_enabled


def _extract_manual_compactor(
    pipeline: HistoryOptimizerPipeline,
    compaction: CompactionConfig | None,
) -> CompactionOptimizer | None:
    """Return the first ``CompactionOptimizer`` found inside *pipeline*, or ``None``.

    When compaction is enabled, ``build_history_optimizer_pipeline`` places a
    ``CompactionOptimizer`` somewhere in the pipeline.  Reusing that instance
    for ``/compact`` avoids constructing a duplicate object with identical
    configuration.  The pipeline may grow with pre-main optimizers (e.g.
    ``ToolResultPruner``) ahead of the ``CompactionOptimizer``, so we search
    rather than assuming a fixed index.
    """
    if compaction is None:
        return None
    return next(
        (opt for opt in pipeline.optimizers if isinstance(opt, CompactionOptimizer)),
        None,
    )


def create_agent(
    user: CloudConnectorUser,
    tools_registry: ToolsRegistry,
    internal_event_category: str,
    tools: Toolset,
    prompt_registry: BasePromptRegistry,
    workflow_id: str,
    workflow_type: GLReportingEventContext,
    system_template_override: str | None,
    agent_name_override: str | None = None,
    compaction: CompactionConfig | None = None,
) -> ChatAgent:
    # Use agent_name_override for chat-partial flows, default to "chat"
    agent_name = agent_name_override if agent_name_override else "chat"

    # Include web_search_options conditionally based on feature flag and client capability
    bind_tools_params: dict[str, dict] = {}
    if is_feature_enabled(FeatureFlag.DAP_WEB_SEARCH) and is_client_capable(
        "web_search"
    ):
        bind_tools_params["web_search_options"] = {}

    prompt: Prompt = prompt_registry.get_on_behalf(
        user=user,
        prompt_id="chat/agent",
        prompt_version="^1.0.0",
        internal_event_category=internal_event_category,
        tools=tools.bindable,  # type: ignore[arg-type]
        bind_tools_params=bind_tools_params,
        internal_event_extra={
            "agent_name": agent_name,
            "workflow_id": workflow_id,
            "workflow_type": workflow_type.value,
        },
    )

    flow_context = FlowContext(
        flow_id=workflow_id,
        flow_type=workflow_type.value,
        user=user,
    )

    optimizer_pipeline = build_history_optimizer_pipeline(
        compaction=compaction if compaction is not None else False,
        flow_context=flow_context,
        agent_name=agent_name,
    )

    manual_compactor = _extract_manual_compactor(optimizer_pipeline, compaction)

    return ChatAgent(
        name=prompt.name,
        prompt_adapter=DefaultPromptAdapter(prompt),
        tools_registry=tools_registry,
        system_template_override=system_template_override,
        toolset=tools,
        optimizer_pipeline=optimizer_pipeline,
        manual_compactor=manual_compactor,
    )
