from langchain_core.messages import BaseMessage
from structlog import get_logger

from duo_workflow_service.conversation.compaction.compactor import ConversationCompactor
from duo_workflow_service.conversation.trimmer import apply_token_based_trim
from duo_workflow_service.entities.state import get_model_max_context_token_limit
from lib.feature_flags.context import FeatureFlag, is_feature_enabled

log = get_logger("compaction.integration")


async def maybe_compact_history(
    compactor: ConversationCompactor | None,
    history: list[BaseMessage],
    agent_name: str,
) -> list[BaseMessage]:
    # Compact history if enabled and ff is true otherwise fallback
    is_ff_on = is_feature_enabled(FeatureFlag.AI_CONTEXT_COMPACTION)

    if compactor and is_ff_on:
        log.info(
            "Start trying context compaction",
            compactor_enable=compactor is not None,
            ff_enable=is_ff_on,
            agent_name=agent_name,
        )
        result = await compactor.compact(messages=history)
        return result.messages

    log.info(
        "Fallback to legacy trim messages",
        compactor_enable=compactor is not None,
        ff_enable=is_ff_on,
        agent_name=agent_name,
    )
    return apply_token_based_trim(
        messages=history,
        component_name=agent_name,
        max_context_tokens=get_model_max_context_token_limit(),
    )
