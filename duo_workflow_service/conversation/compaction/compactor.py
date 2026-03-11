from typing import cast

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable, RunnableBinding
from langgraph.constants import TAG_NOSTREAM
from structlog import get_logger

from ai_gateway.prompts.typing import Model
from duo_workflow_service.conversation.compaction.schema import (
    CompactionConfig,
    CompactionResult,
)
from duo_workflow_service.conversation.compaction.token_estimator import (
    CompactionTokenEstimator,
)
from duo_workflow_service.conversation.compaction.utils import slice_for_summarization
from duo_workflow_service.entities.state import get_model_max_context_token_limit

log = get_logger("compactor")


class ConversationCompactor:
    """Handles conversation history compaction via summarization.

    This class encapsulates all logic for determining when compaction is needed and performing the actual summarization
    of conversation history.
    """

    def __init__(
        self,
        llm_model: Runnable,
        config: CompactionConfig,
        token_estimator: CompactionTokenEstimator,
    ):
        self._llm = llm_model
        self._token_estimator = token_estimator
        self._config = config
        self._summary_prompt_tokens = self._token_estimator.estimate_arbitrary_messages(
            [
                SystemMessage(content=self._config.summarizer_system_prompt),
                HumanMessage(content=self._config.summarizer_user_prompt),
            ]
        )

    def should_compact(self, messages: list[BaseMessage]) -> bool:
        """Determine if messages should be summarized.

        Summarization is recommended when:
        - Message count exceeds max_recent_messages, AND
        - Total token count exceeds trim_threshold of the model's max context limit

        Args:
            messages: List of messages to evaluate

        Returns:
            True if messages should be summarized, False otherwise
        """
        if len(messages) <= self._config.max_recent_messages:
            return False

        token_count = self._token_estimator.estimate_complete_history(messages)
        threshold = self._config.trim_threshold * get_model_max_context_token_limit()
        log.info(
            "Checking if compact is needed.",
            message_token=token_count,
            threshold=threshold,
        )
        return token_count > threshold

    async def compact(self, messages: list[BaseMessage]) -> CompactionResult:
        """Compact conversation history by summarizing older messages.

        Keeps:
        - First consecutive HumanMessages (initial context)
        - Recent messages within token and count limits
        - Summarizes middle messages into a single AIMessage

        Args:
            messages: List of messages to compact

        Returns:
            CompactionResult with compacted messages and metadata
        """
        if not messages or not self.should_compact(messages):
            return CompactionResult(messages=messages, was_compacted=False)

        slices = slice_for_summarization(messages, self._config, self._token_estimator)

        if not slices.to_summarize:
            log.warning(
                "No messages to summarize",
                n_msgs=len(messages),
                n_leading=len(slices.leading_context),
                n_recent=len(slices.recent_to_keep),
            )
            return CompactionResult(messages=messages, was_compacted=False)

        original_tokens = self._token_estimator.estimate_complete_history(messages)

        try:
            summary = await self._invoke_summarizer(slices.to_summarize)
        except Exception as e:
            log.warning(
                "Failed to summarize messages, keeping original",
                error=str(e),
                error_type=type(e).__name__,
                n_msgs=len(messages),
            )
            return CompactionResult(
                messages=messages,
                was_compacted=False,
                error=e,
            )

        compacted_messages = slices.leading_context + [summary] + slices.recent_to_keep

        compacted_tokens = self._calculate_compacted_tokens(original_tokens, summary)

        log.info(
            "Finish context compaction",
            num_msg_before=len(messages),
            num_msg_after=len(compacted_messages),
            msg_tokens_before=original_tokens,
            msg_tokens_after=compacted_tokens,
            compaction_ratio=1 - compacted_tokens / original_tokens,
        )

        return CompactionResult(
            messages=compacted_messages,
            was_compacted=True,
            tokens_before=original_tokens,
            tokens_after=compacted_tokens,
            messages_summarized=len(slices.to_summarize),
        )

    async def _invoke_summarizer(self, messages: list[BaseMessage]) -> AIMessage:
        """Invoke the LLM to summarize messages.

        Uses the "nostream" tag to prevent LangGraph from streaming the
        summarization to the UI. This is the recommended LangGraph pattern
        for hiding internal LLM calls from stream_mode="messages" output.
        See: langgraph.constants.TAG_NOSTREAM
        """
        # TODO: migrate to prompt registry needed in
        # https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/2014
        log.info("Start compaction summarization llm call.")
        result = await self._llm.ainvoke(
            [
                SystemMessage(content=self._config.summarizer_system_prompt),
                *messages,
                HumanMessage(content=self._config.summarizer_user_prompt),
            ],
            config={"tags": [TAG_NOSTREAM], "run_name": "Compaction Summarization"},
        )
        return cast(AIMessage, result)

    def _calculate_compacted_tokens(
        self, original_tokens: int, summary: AIMessage
    ) -> int:
        """Calculate the token count after compaction."""
        usage_metadata = summary.usage_metadata
        input_tokens = usage_metadata.get("input_tokens", 0) if usage_metadata else 0
        output_tokens = usage_metadata.get("output_tokens", 0) if usage_metadata else 0

        return (
            original_tokens - input_tokens + self._summary_prompt_tokens + output_tokens
        )


def create_conversation_compactor(
    config: CompactionConfig, llm_model: Model
) -> ConversationCompactor:
    return ConversationCompactor(
        llm_model=cast(
            Runnable,
            llm_model.bound if isinstance(llm_model, RunnableBinding) else llm_model,
        ),
        config=config,
        token_estimator=CompactionTokenEstimator(),
    )
