from enum import StrEnum
from typing import (
    Annotated,
    Any,
    Dict,
    List,
    NotRequired,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

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
from duo_workflow_service.gitlab.gitlab_api import Namespace, Project
from duo_workflow_service.token_counter.approximate_token_counter import (
    ApproximateTokenCounter,
)
from duo_workflow_service.workflows.type_definitions import AdditionalContext

# max content tokens is 400K but adding a buffer of 10% just in case
MAX_CONTEXT_TOKENS = int(400_000 * 0.90)
MAX_SINGLE_MESSAGE_TOKENS = int(MAX_CONTEXT_TOKENS * 0.65)

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
    delete: NotRequired[bool]  # Used to signal deletion in state updates


class Plan(TypedDict):
    steps: List[Task]
    reset: NotRequired[bool]  # Used in updates to discard previous steps


class WorkflowStatusEnum(StrEnum):
    CREATED = "created"
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
    APPROVAL_ERROR = "approval_error"
    FINISHED = "finished"
    STOPPED = "stopped"


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
    tool_response: NotRequired[Any]


class UiChatLog(TypedDict):
    message_type: MessageTypeEnum
    message_sub_type: Optional[str]
    content: str
    timestamp: str
    status: Optional[Union[ToolStatus, SlashCommandStatus]]
    correlation_id: Optional[str]
    tool_info: Optional[ToolInfo]
    additional_context: Optional[List[AdditionalContext]]


def _pretrim_large_messages(
    messages: List[BaseMessage], token_counter: ApproximateTokenCounter
) -> List[BaseMessage]:
    processed_messages = []
    for message in messages:
        msg_token = token_counter.count_tokens([message])
        if msg_token > MAX_SINGLE_MESSAGE_TOKENS:
            logger.debug(
                f"Message with role: {message.type} token size: {msg_token} "
                f"exceeds the single message token limit: {MAX_SINGLE_MESSAGE_TOKENS}."
                f"Replacing its content with a placeholder."
            )
            message_copy = message.model_copy()
            message_copy.content = (
                "Previous message was too large for context window and was omitted. Please respond "
                "based on the visible context."
            )
            processed_messages.append(message_copy)
        else:
            processed_messages.append(message)
    return processed_messages


def _plan_reducer(current: Plan, new: Optional[Plan]) -> Plan:
    if new is None:
        return current

    if current is None or "steps" not in current:
        current = Plan(steps=[])

    # Discard existing steps if asked to reset
    if new.get("reset"):
        current["steps"] = new["steps"]
        return current

    for step in new["steps"]:
        # Find existing step with same id
        existing_step = next(
            (item for item in current["steps"] if item["id"] == step["id"]), None
        )

        # Check if incoming step is marked for deletion
        delete = step.get("delete", False)

        # If step doesn't exist, add it
        if existing_step is None:
            # ... unless it's marked for deletion, in which case skip it
            if not delete:
                current["steps"].append(step)
        else:
            # If step exists and is marked for deletion, remove it
            if delete:
                current["steps"].remove(existing_step)
            else:
                # Update existing step with new values
                existing_step.update(step)

    return current


def get_messages_profile(
    messages: List[BaseMessage],
    token_counter: ApproximateTokenCounter,
    include_tool_tokens: bool = True,
) -> Tuple[List[str], int]:

    roles = [msg.type for msg in messages]
    token_size = (
        token_counter.count_tokens(messages, include_tool_tokens=include_tool_tokens)
        if messages
        else 0
    )
    return roles, token_size


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

        current_msg_roles, current_msg_token = get_messages_profile(
            messages=reduced.get(agent_name, []),
            token_counter=token_counter,
            include_tool_tokens=False,
        )

        new_msg_roles, new_msg_token = get_messages_profile(
            messages=new_messages,
            token_counter=token_counter,
            include_tool_tokens=False,
        )

        logger.debug(
            f"Starting trimming conversation history for {agent_name} with "
            f"current messages roles: {current_msg_roles}, token size: {current_msg_token}; "
            f"new messages roles: {new_msg_roles}, token size: {new_msg_token}; "
            f"total token size including tool specs: {current_msg_token + new_msg_token + token_counter.tool_tokens}",
            current_msg_tokens=current_msg_token,
            new_msg_token=new_msg_token,
            total_tokens_before_trimming=current_msg_token
            + new_msg_token
            + token_counter.tool_tokens,
        )

        processed_messages = _pretrim_large_messages(new_messages, token_counter)

        if not processed_messages:
            continue

        existing_messages = reduced.get(agent_name, [])
        reduced[agent_name] = existing_messages + processed_messages

        pretrimmed_msg_roles, pretrimmed_msg_token = get_messages_profile(
            messages=reduced[agent_name],
            token_counter=token_counter,
            include_tool_tokens=False,
        )

        logger.debug(
            f"Finished pretrim with messages roles: {pretrimmed_msg_roles}, message token: {pretrimmed_msg_token}, "
            f"estimated token size including tool specs: {pretrimmed_msg_token + token_counter.tool_tokens}",
            total_tokens_after_pretrimming=pretrimmed_msg_token
            + token_counter.tool_tokens,
        )

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
            if not reduced[agent_name] or len(reduced[agent_name]) == 1:
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
                    "Trim resulted in empty messages/invalid messages - falling back to minimal context",
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

            fallback_messages = system_messages + non_system_messages[-5:]
            reduced[agent_name] = _restore_message_consistency(fallback_messages)

        posttrimmed_msg_roles, posttrimmed_msg_token = get_messages_profile(
            messages=reduced[agent_name],
            token_counter=token_counter,
            include_tool_tokens=False,
        )

        logger.debug(
            f"Finished posttrim with messages roles: {posttrimmed_msg_roles}, message token: {posttrimmed_msg_token}, "
            f"estimated token size including tool specs: {posttrimmed_msg_token + token_counter.tool_tokens}",
            total_tokens_after_posttrimming=posttrimmed_msg_token
            + token_counter.tool_tokens,
        )

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
    plan: Annotated[Plan, _plan_reducer]
    status: WorkflowStatusEnum
    conversation_history: Annotated[
        Dict[str, List[BaseMessage]], _conversation_history_reducer
    ]
    ui_chat_log: Annotated[List[UiChatLog], _ui_chat_log_reducer]
    handover: List[BaseMessage]
    last_human_input: Union[WorkflowEvent, None]
    project: Project | None
    goal: str | None
    additional_context: list[AdditionalContext] | None


class ApprovalStateRejection(BaseModel):
    message: Optional[str]


class ChatWorkflowState(TypedDict):
    plan: Plan
    status: WorkflowStatusEnum
    conversation_history: Annotated[
        Dict[str, List[BaseMessage]], _conversation_history_reducer
    ]
    ui_chat_log: Annotated[List[UiChatLog], _ui_chat_log_reducer]
    last_human_input: Union[WorkflowEvent, None]
    goal: str | None
    project: Project | None
    namespace: Namespace | None
    approval: ApprovalStateRejection | None
    preapproved_tools: list[str] | None


DuoWorkflowStateType = Union[WorkflowState, ChatWorkflowState]


class WorkflowContext(TypedDict):
    id: int
    plan: Plan
    goal: str
    summary: str


class Context(TypedDict):
    workflow: WorkflowContext
