from langchain_core.messages import BaseMessage
from structlog import get_logger

from ai_gateway.config import get_config
from duo_workflow_service.conversation.compaction.compactor import ConversationCompactor
from duo_workflow_service.conversation.trimmer import apply_token_based_trim
from duo_workflow_service.entities.state import get_model_max_context_token_limit
from lib.context import is_gitlab_team_member
from lib.feature_flags.context import FeatureFlag, is_feature_enabled
from lib.internal_events.context import InternalEventAdditionalProperties
from lib.internal_events.event_enum import EventEnum

log = get_logger("compaction.integration")


async def maybe_compact_history(
    compactor: ConversationCompactor | None,
    history: list[BaseMessage],
    agent_name: str,
) -> list[BaseMessage]:
    """Compact or trim conversation history.

    Uses compaction (LLM summarization) when enabled, otherwise falls back to legacy token-based trimming. Compaction is
    available for both cloud-hosted and self-hosted deployments.
    """
    config = get_config()
    is_self_hosted = config.custom_models.enabled

    # Compact history if enabled and ff is true otherwise fallback
    is_ff_on = is_feature_enabled(FeatureFlag.AI_CONTEXT_COMPACTION)

    if compactor and is_ff_on:
        log.info(
            "Start trying context compaction",
            compactor_enable=compactor is not None,
            ff_enable=is_ff_on,
            is_self_hosted=is_self_hosted,
            is_gitlab_team_member=is_gitlab_team_member.get(),
            agent_name=agent_name,
        )
        result = await compactor.compact(messages=history)
        return result.messages

    log.info(
        "Fallback to legacy trim messages",
        compactor_enable=compactor is not None,
        ff_enable=is_ff_on,
        is_self_hosted=is_self_hosted,
        is_gitlab_team_member=is_gitlab_team_member.get(),
        agent_name=agent_name,
    )
    trim_result = apply_token_based_trim(
        messages=history,
        component_name=agent_name,
        max_context_tokens=get_model_max_context_token_limit(),
    )

    # Reuse internal event client from compactor instance for now, will refactor it in
    # https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/4862
    if (
        trim_result.was_trimmed
        and compactor is not None
        and compactor._internal_events_client is not None
    ):
        compactor._internal_events_client.track_event(
            event_name=EventEnum.LEGACY_TRIM_EXECUTED.value,
            additional_properties=InternalEventAdditionalProperties(
                label=agent_name,
                property="workflow_id",
                value=compactor._workflow_id,
                workflow_type=compactor._workflow_type,
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
