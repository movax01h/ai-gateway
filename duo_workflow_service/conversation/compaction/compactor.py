import time
from enum import StrEnum
from typing import cast

from gitlab_cloud_connector import CloudConnectorUser
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.constants import TAG_NOSTREAM
from structlog import get_logger

from ai_gateway.prompts.base import BasePromptRegistry, Prompt
from duo_workflow_service.conversation.compaction.schema import (
    CompactionConfig,
    CompactionResult,
)
from duo_workflow_service.conversation.compaction.utils import (
    slice_for_summarization,
    strip_tool_metadata,
)
from duo_workflow_service.conversation.token_estimator import count_tokens
from duo_workflow_service.entities.state import get_model_max_context_token_limit
from duo_workflow_service.monitoring import duo_workflow_metrics
from lib.context import StarletteUser, is_gitlab_team_member
from lib.context.model import get_model_metadata
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import InternalEventAdditionalProperties
from lib.internal_events.event_enum import EventEnum

log = get_logger("compactor")

COMPACTION_PROMPT_ID = "conversation_compaction"
COMPACTION_PROMPT_VERSION = "^1.0.0"

COMPACTION_CONTINUE_MESSAGE = (
    "Continue working on the task based on the conversation above."
)


class CompactionStatus(StrEnum):
    SUCCESS = "success"
    ERROR = "error"


class ConversationCompactor:
    """Handles conversation history compaction via summarization.

    This class encapsulates all logic for determining when compaction is needed and performing the actual summarization
    of conversation history.
    """

    OPERATION_TYPE = "compaction_auto"

    def __init__(
        self,
        prompt_registry: BasePromptRegistry,
        user: StarletteUser | CloudConnectorUser,
        config: CompactionConfig,
        agent_name: str,
        workflow_id: str,
        workflow_type: str,
        internal_events_client: InternalEventsClient | None = None,
    ):
        # Use the same (default) model as the parent workflow rather than a
        # smaller variant.  The compaction trigger threshold is based on the
        # workflow model's context window, so the messages handed to the
        # summarizer can exceed a smaller model's context limit and cause
        # API errors.  Using the default model guarantees the summarizer
        # can always handle the payload.
        # is_graph_node=True is a safety net: if no gRPC model metadata
        # context is set (tests, edge cases), it falls back to
        # duo_agent_platform instead of failing on an unknown feature
        # setting derived from the prompt_id.
        model_metadata = get_model_metadata()
        self._prompt: Prompt = prompt_registry.get_on_behalf(
            user,
            COMPACTION_PROMPT_ID,
            COMPACTION_PROMPT_VERSION,
            model_metadata=model_metadata,
            is_graph_node=True,
            internal_event_extra={
                "agent_name": agent_name,
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "is_compaction_call": True,
                "operation_type": self.OPERATION_TYPE,
            },
        )
        self._agent_name = agent_name
        self._workflow_id = workflow_id
        self._workflow_type = workflow_type
        self._model_name = getattr(model_metadata, "name", "unknown")
        self._internal_events_client = internal_events_client
        self._config = config
        prompt_tpl = cast(ChatPromptTemplate, self._prompt.prompt_tpl)
        self._prompt_overhead_tokens = count_tokens(
            prompt_tpl.format_messages(history=[]), is_complete_history=False
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

        token_count = count_tokens(messages, is_complete_history=True)
        threshold = self._config.trim_threshold * get_model_max_context_token_limit()
        log.info(
            "Checking if compact is needed.",
            message_token=token_count,
            threshold=threshold,
            is_gitlab_team_member=is_gitlab_team_member.get(),
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

        slices = slice_for_summarization(messages, self._config)

        if not slices.to_summarize:
            log.warning(
                "No messages to summarize",
                n_msgs=len(messages),
                n_leading=len(slices.leading_context),
                n_recent=len(slices.recent_to_keep),
            )
            return CompactionResult(messages=messages, was_compacted=False)

        original_tokens = count_tokens(messages, is_complete_history=True)
        start_time = time.time()

        try:
            with duo_workflow_metrics.time_compaction_llm(
                flow_type=self._workflow_type,
                agent_name=self._agent_name,
            ):
                summary = await self._invoke_summarizer(slices.to_summarize)
            duration = time.time() - start_time

            # Prometheus counter
            duo_workflow_metrics.count_compaction_execution(
                flow_type=self._workflow_type,
                agent_name=self._agent_name,
                status=CompactionStatus.SUCCESS,
            )
        except Exception as e:
            duration = time.time() - start_time

            # Prometheus counter for failure
            duo_workflow_metrics.count_compaction_execution(
                flow_type=self._workflow_type,
                agent_name=self._agent_name,
                status=CompactionStatus.ERROR,
            )

            self._fire_compaction_event(
                status=CompactionStatus.ERROR,
                model_name=self._model_name,
                tokens_before=original_tokens,
                duration_seconds=duration,
                error_type=type(e).__name__,
            )

            log.error(
                "Failed to summarize messages, keeping original",
                error=str(e),
                error_type=type(e).__name__,
                n_msgs=len(messages),
                n_to_summarize=len(slices.to_summarize),
                is_gitlab_team_member=is_gitlab_team_member.get(),
                exc_info=True,
            )
            return CompactionResult(
                messages=messages,
                was_compacted=False,
                error=e,
            )

        compacted_messages = slices.leading_context + [summary] + slices.recent_to_keep

        if isinstance(compacted_messages[-1], AIMessage):
            # Vertex AI / Gemini requires conversations to end with a user
            # message (strict role alternation: user <-> model). After
            # compaction, the message list may end with an AIMessage (summary
            # or recent agent turn), which the Gemini API rejects with HTTP
            # 400. Append a synthetic HumanMessage to satisfy this constraint.
            # Note: ToolMessages are not affected -- the Gemini API converts
            # them to role-less ContentType objects that don't participate in
            # role alternation.
            compacted_messages.append(HumanMessage(content=COMPACTION_CONTINUE_MESSAGE))

        compacted_tokens = self._calculate_compacted_tokens(original_tokens, summary)

        # Extract token usage from the summary for the Snowplow event
        usage = summary.usage_metadata
        compaction_input_tokens = usage.get("input_tokens", 0) if usage else 0
        compaction_output_tokens = usage.get("output_tokens", 0) if usage else 0

        max_context_tokens = get_model_max_context_token_limit()
        self._fire_compaction_event(
            status=CompactionStatus.SUCCESS,
            model_name=self._model_name,
            compaction_input_tokens=compaction_input_tokens,
            compaction_output_tokens=compaction_output_tokens,
            tokens_before=original_tokens,
            tokens_after=compacted_tokens,
            tokens_saved=original_tokens - compacted_tokens,
            compaction_ratio=(
                round(1 - compacted_tokens / original_tokens, 4)
                if original_tokens > 0
                else 0
            ),
            messages_summarized=len(slices.to_summarize),
            token_budget=int(self._config.trim_threshold * max_context_tokens),
            max_context_tokens=max_context_tokens,
            duration_seconds=round(duration, 3),
        )

        log.info(
            "Finish context compaction",
            num_msg_before=len(messages),
            num_msg_after=len(compacted_messages),
            msg_tokens_before=original_tokens,
            msg_tokens_after=compacted_tokens,
            compaction_ratio=1 - compacted_tokens / original_tokens,
            compaction_usage_data=summary.usage_metadata,
            compaction_model_details=summary.response_metadata,
            workflow_id=self._workflow_id,
            workflow_type=self._workflow_type,
            agent_name=self._agent_name,
            model_name=self._model_name,
            compaction_config=self._config.model_dump(),
            is_gitlab_team_member=is_gitlab_team_member.get(),
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
        log.info(
            "Start compaction summarization llm call.",
            is_gitlab_team_member=is_gitlab_team_member.get(),
        )
        # Strip tool metadata from messages before summarization.
        # The summarizer only needs text content, and some LLM providers
        # reject messages with tool_calls when no tools= param is specified.
        # Stripping unconditionally is safe since we convert tool call/result
        # info to human-readable text.
        messages = strip_tool_metadata(messages)
        config: RunnableConfig = {
            "tags": [TAG_NOSTREAM],
            "run_name": "Compaction Summarization",
        }
        result = await self._prompt.ainvoke({"history": messages}, config=config)
        return cast(AIMessage, result)

    def _calculate_compacted_tokens(
        self, original_tokens: int, summary: AIMessage
    ) -> int:
        """Calculate the token count after compaction."""
        usage_metadata = summary.usage_metadata
        input_tokens = usage_metadata.get("input_tokens", 0) if usage_metadata else 0
        output_tokens = usage_metadata.get("output_tokens", 0) if usage_metadata else 0
        return (
            original_tokens
            - (input_tokens - self._prompt_overhead_tokens)
            + output_tokens
        )

    def _fire_compaction_event(self, **kwargs) -> None:
        """Fire a compaction_executed Snowplow event."""
        if self._internal_events_client is None:
            return
        self._internal_events_client.track_event(
            event_name=EventEnum.COMPACTION_EXECUTED.value,
            additional_properties=InternalEventAdditionalProperties(
                label=self._agent_name,
                property="workflow_id",
                value=self._workflow_id,
                workflow_type=self._workflow_type,
                **kwargs,
            ),
            category=self.__class__.__name__,
        )


def create_conversation_compactor(
    config: CompactionConfig,
    prompt_registry: BasePromptRegistry,
    user: StarletteUser | CloudConnectorUser,
    agent_name: str,
    workflow_id: str,
    workflow_type: str,
    internal_events_client: InternalEventsClient | None = None,
) -> ConversationCompactor:
    return ConversationCompactor(
        prompt_registry=prompt_registry,
        user=user,
        config=config,
        agent_name=agent_name,
        workflow_id=workflow_id,
        workflow_type=workflow_type,
        internal_events_client=internal_events_client,
    )
