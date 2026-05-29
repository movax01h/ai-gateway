from dependency_injector.wiring import Provide, inject
from langchain_core.messages import BaseMessage
from structlog import get_logger

from ai_gateway.config import get_config
from ai_gateway.container import ContainerApplication
from duo_workflow_service.conversation.compaction.compactor import ConversationCompactor
from duo_workflow_service.conversation.trimmer import apply_token_based_trim
from duo_workflow_service.entities.state import get_model_max_context_token_limit
from lib.context import is_gitlab_team_member
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import InternalEventAdditionalProperties
from lib.internal_events.event_enum import EventEnum

log = get_logger("compaction.integration")


@inject
async def maybe_compact_history(
    compactor: ConversationCompactor | None,
    history: list[BaseMessage],
    agent_name: str,
    internal_events_client: InternalEventsClient = Provide[
        ContainerApplication.internal_event.client
    ],
) -> list[BaseMessage]:
    """Compact or trim conversation history.

    Uses compaction (LLM summarization) when a compactor is supplied, otherwise falls back to legacy token-based
    trimming. Compaction is available for both cloud-hosted and self-hosted deployments.
    """
    config = get_config()
    is_self_hosted = config.custom_models.enabled

    if compactor is not None:
        log.info(
            "Start trying context compaction",
            is_self_hosted=is_self_hosted,
            is_gitlab_team_member=is_gitlab_team_member.get(),
            agent_name=agent_name,
        )
        result = await compactor.compact(messages=history)
        return result.messages

    log.info(
        "Fallback to legacy trim messages",
        is_self_hosted=is_self_hosted,
        is_gitlab_team_member=is_gitlab_team_member.get(),
        agent_name=agent_name,
    )
    trim_result = apply_token_based_trim(
        messages=history,
        component_name=agent_name,
        max_context_tokens=get_model_max_context_token_limit(),
    )

    if trim_result.was_trimmed:
        internal_events_client.track_event(
            event_name=EventEnum.LEGACY_TRIM_EXECUTED.value,
            additional_properties=InternalEventAdditionalProperties(
                label=agent_name,
                tokens_before=trim_result.tokens_before,
                tokens_after=trim_result.tokens_after,
                tokens_removed=trim_result.tokens_before - trim_result.tokens_after,
                messages_before=trim_result.messages_before,
                messages_after=trim_result.messages_after,
                token_budget=trim_result.token_budget,
                max_context_tokens=trim_result.max_context_tokens,
                duration_ms=trim_result.duration_ms,
            ),
            category="legacy_trimmer",
        )

    return trim_result.messages
