from enum import StrEnum
from typing import Annotated, Any, Dict, List, Optional, TypedDict, Union

import structlog
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    trim_messages,
)
from pydantic import BaseModel

from duo_workflow_service.entities.event import WorkflowEvent
from duo_workflow_service.token_counter.approximate_token_counter import (
    ApproximateTokenCounter,
)

# max content tokens is 200K but adding a buffer of 5% just in case
MAX_CONTEXT_TOKENS = int(200_000 * 0.95)
MAX_SINGLE_MESSAGE_TOKENS = int(MAX_CONTEXT_TOKENS * 0.75)

logger = structlog.stdlib.get_logger("workflow")


class TaskStatus(StrEnum):
    NOT_STARTED = "Not Started"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    CANCELLED = "Cancelled"


class Task(TypedDict):
    id: str
    description: str
    status: TaskStatus


class Plan(TypedDict):
    steps: List[Task]


class WorkflowStatusEnum(StrEnum):
    NOT_STARTED = "Not Started"
    PLANNING = "Planning"
    EXECUTION = "Execution"
    COMPLETED = "Completed"
    ERROR = "Error"
    PAUSED = "Paused"
    CANCELLED = "Cancelled"
    INPUT_REQUIRED = "input_required"
    PLAN_APPROVAL_REQUIRED = "plan_approval_required"
    TOOL_CALL_APPROVAL_REQUIRED = "tool_call_approval_required"


class MessageTypeEnum(StrEnum):
    AGENT = "agent"
    USER = "user"
    TOOL = "tool"
    REQUEST = "request"
    WORKFLOW_END = "workflow_end"


class ToolStatus(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"


class SlashCommandStatus(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"


class ToolInfo(TypedDict):
    name: str
    args: dict[str, Any]


class UiChatLog(TypedDict):
    message_type: MessageTypeEnum
    content: str
    timestamp: str
    status: Optional[Union[ToolStatus, SlashCommandStatus]]
    correlation_id: Optional[str]
    tool_info: Optional[ToolInfo]


def _pretrim_large_messages(
    messages: List[BaseMessage], token_counter
) -> List[BaseMessage]:
    processed_messages = []
    for message in messages:
        if token_counter.count_tokens([message]) > MAX_SINGLE_MESSAGE_TOKENS:
            message_copy = message.model_copy()
            message_copy.content = "Previous message was too large for context window and was omitted. Please respond based on the visible context."
            processed_messages.append(message_copy)
        else:
            processed_messages.append(message)
    return processed_messages


# reducers can be called multiple times by the LangGraph framework. One MUST assure
# that fully new object is returned from reducer function. If mutation happens instead,
# results might be broken !!!!!!
def _conversation_history_reducer(
    current: Dict[str, List[BaseMessage]], new: Optional[Dict[str, List[BaseMessage]]]
) -> Dict[str, List[BaseMessage]]:
    reduced = {**current}

    if new is None:
        return reduced

    for agent_name, new_messages in new.items():
        if not new_messages:
            continue

        token_counter = ApproximateTokenCounter(agent_name)
        processed_messages = _pretrim_large_messages(new_messages, token_counter)

        if not processed_messages:
            continue

        existing_messages = []

        if agent_name in reduced:
            existing_messages = reduced[agent_name]
            reduced[agent_name] = reduced[agent_name] + processed_messages
        else:
            reduced[agent_name] = processed_messages

        try:
            trimmed_messages = trim_messages(
                reduced[agent_name],
                max_tokens=MAX_CONTEXT_TOKENS,
                strategy="last",
                token_counter=token_counter.count_tokens,
                start_on="human",
                include_system=True,
                allow_partial=False,
            )

            reduced[agent_name] = _restore_message_consistency(trimmed_messages)

            # If trimming resulted in empty list, keep at least the last few messages along with the system message
            if not reduced[agent_name]:
                all_messages = current.get(agent_name, []) + processed_messages
                system_messages = [
                    msg for msg in all_messages if isinstance(msg, SystemMessage)
                ]
                non_system_messages = [
                    msg for msg in all_messages if not isinstance(msg, SystemMessage)
                ]

                min_non_system = min(3, len(non_system_messages))
                fallback_messages = (
                    system_messages + non_system_messages[-min_non_system:]
                )

                reduced[agent_name] = _restore_message_consistency(fallback_messages)

                logger.warning(
                    "Trim resulted in empty messages - falling back to minimal context",
                    agent_name=agent_name,
                )

            # Detect potential conversation loops or trimming failures
            post_trimmed_messages = reduced[agent_name]
            if (
                existing_messages == post_trimmed_messages
                and len(processed_messages) > 0
            ):
                logger.warning(
                    "Trimming resulted in identical message state - possible conversation loop",
                    agent_name=agent_name,
                )

        except Exception as e:
            logger.error(
                f"Error during message trimming: {str(e)}",
                agent_name=agent_name,
                exc_info=True,
            )
            # Keep the system messages plus a few recent messages as fallback
            all_messages = current.get(agent_name, []) + processed_messages
            system_messages = [
                msg for msg in all_messages if isinstance(msg, SystemMessage)
            ]
            non_system_messages = [
                msg for msg in all_messages if not isinstance(msg, SystemMessage)
            ]

            fallback_messages = system_messages + non_system_messages[-10:]
            reduced[agent_name] = _restore_message_consistency(fallback_messages)

    return reduced


def _restore_message_consistency(messages: List[BaseMessage]) -> List[BaseMessage]:
    if not messages:
        return []

    # Identify all AIMessages with tool calls
    tool_call_indices = {}
    for i, msg in enumerate(messages):
        if isinstance(msg, AIMessage) and hasattr(msg, "tool_calls") and msg.tool_calls:
            for tool_call in msg.tool_calls:
                tool_call_id = tool_call.get("id")
                if tool_call_id:
                    tool_call_indices[tool_call_id] = i

    # Process the messages to ensure consistency
    result: List[BaseMessage] = []
    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage):
            tool_call_id = getattr(msg, "tool_call_id", None)
            # Check if this tool message has a corresponding AIMessage with tool_calls
            # AND if the tool message appears after its parent
            if (
                tool_call_id
                and tool_call_id in tool_call_indices
                and i > tool_call_indices[tool_call_id]
            ):
                result.append(msg)
            else:
                # Convert invalid ToolMessage to HumanMessage
                if msg.content:
                    result.append(HumanMessage(content=msg.content))
        else:
            result.append(msg)

    return result


def _ui_chat_log_reducer(
    current: List[UiChatLog], new: Optional[List[UiChatLog]]
) -> List[UiChatLog]:
    if new is None:
        return current.copy()

    return current + new


class WorkflowState(TypedDict):
    plan: Plan
    status: WorkflowStatusEnum
    conversation_history: Annotated[
        Dict[str, List[BaseMessage]], _conversation_history_reducer
    ]
    ui_chat_log: Annotated[List[UiChatLog], _ui_chat_log_reducer]
    handover: List[BaseMessage]
    last_human_input: Union[WorkflowEvent, None]


class ReplacementRule(BaseModel):
    element: str
    rules: str


class SearchAndReplaceConfig(BaseModel):
    file_types: list[str]
    domain_speciality: str
    assignment_description: str
    replacement_rules: list[ReplacementRule]


class SearchAndReplaceWorkflowState(TypedDict):
    plan: Plan  # TODO remove not used once ToolExecutor is being refactored
    handover: List[
        BaseMessage
    ]  # TODO remove not used once HandoverAgent is being refactored
    status: WorkflowStatusEnum
    # conversation_history always is going to look as follow, and is going to be rewritten in each graph cycle
    # {
    #    'replace_agent': [
    #        SystemMessage(…),
    #        HumanMessage(…),
    #        AIMessage(…)
    #     ]
    # }
    conversation_history: Dict[str, List[BaseMessage]]
    ui_chat_log: Annotated[List[UiChatLog], _ui_chat_log_reducer]
    config: Optional[SearchAndReplaceConfig]
    directory: str
    pending_files: List[str]


class ChatWorkflowState(TypedDict):
    plan: Plan
    status: WorkflowStatusEnum
    conversation_history: Annotated[
        Dict[str, List[BaseMessage]], _conversation_history_reducer
    ]
    ui_chat_log: Annotated[List[UiChatLog], _ui_chat_log_reducer]
    last_human_input: Union[WorkflowEvent, None]


DuoWorkflowStateType = Union[
    WorkflowState, SearchAndReplaceWorkflowState, ChatWorkflowState
]


class WorkflowContext(TypedDict):
    id: int
    plan: Plan
    goal: str
    summary: str


class Context(TypedDict):
    workflow: WorkflowContext
