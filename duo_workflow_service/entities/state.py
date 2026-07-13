from enum import StrEnum
from typing import Annotated, Any, Dict, List, NotRequired, Optional, TypedDict, Union

import structlog
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from duo_workflow_service.conversation.trimmer import LEGACY_MAX_CONTEXT_TOKENS
from duo_workflow_service.entities.event import WorkflowEvent
from duo_workflow_service.gitlab.gitlab_api import Namespace, Project
from duo_workflow_service.security.secret_redaction import redact_secrets_for_ui
from duo_workflow_service.workflows.type_definitions import AdditionalContext
from lib.context import ModelSizeBucket, get_model_metadata

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
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"


class SlashCommandStatus(StrEnum):
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"


# Display only first 4KB of a tool response on UI to avoid duplicating large responses twice in a checkpoint
TOOL_RESPONSE_MAX_DISPLAY_MSG = 4 * 1024

# Shared sub_type / JSON-payload error code used to tag tier-access-denied
# events. Read by ChatAgent when tagging the AGENT UiChatLog and by
# ToolsExecutor when serializing the tool_response payload.
TIER_ACCESS_DENIED_SUB_TYPE = "tier_access_denied"


class ToolInfo(TypedDict):
    name: str
    args: dict[str, Any]
    tool_response: NotRequired[Any]
    suggested_patterns: NotRequired[list[str]]
    # Semantic version of the tool that produced this entry (from
    # ``DuoBaseTool.tool_version``). Lets the client version the tool→component
    # contract for generative UI. Source/server identity for MCP tools is a
    # separate follow-up (needs a `server` field on the McpTool proto).
    tool_version: NotRequired[str]


class UiChatLog(TypedDict):
    message_type: MessageTypeEnum
    message_sub_type: Optional[str]
    content: str
    timestamp: str
    status: Optional[Union[ToolStatus, SlashCommandStatus]]
    correlation_id: Optional[str]
    tool_info: Optional[ToolInfo]
    additional_context: Optional[List[AdditionalContext]]
    message_id: Optional[str]
    required_plan: NotRequired[Optional[str]]
    component_name: NotRequired[Optional[str]]
    subsession_id: NotRequired[Optional[str]]


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
        # If step exists and is marked for deletion, remove it
        elif delete:
            current["steps"].remove(existing_step)
        else:
            # Update existing step with new values
            existing_step.update(step)

    return current


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

        existing_messages = reduced.get(agent_name, [])
        reduced[agent_name] = existing_messages + new_messages

    return reduced


def get_model_max_context_token_limit(
    model_size: Optional[ModelSizeBucket] = None,
) -> int:
    # Returns the context-window limit for the given model size (None = the current
    # default model), or LEGACY_MAX_CONTEXT_TOKENS when no model metadata is set.
    model_metadata = get_model_metadata(model_size)
    token_limit = (
        model_metadata.llm_definition.max_context_tokens
        if model_metadata is not None
        else LEGACY_MAX_CONTEXT_TOKENS
    )
    logger.info(
        "Model context window limit.",
        model_size=model_size,
        token_limit=token_limit,
    )
    return token_limit


def get_current_model_max_context_token_limit() -> int:
    # Convenience wrapper for the current default model (the None = current-model
    # convention of get_model_max_context_token_limit).
    return get_model_max_context_token_limit(None)


def _ui_chat_log_reducer(
    current: List[UiChatLog], new: Optional[List[UiChatLog]]
) -> List[UiChatLog]:
    if new is None:
        return current.copy()

    return current + new


def build_tool_info(
    name: str, args: dict[str, Any], tool_response: Any = None
) -> ToolInfo:
    """Build a ToolInfo dict for UiChatLog display.

    Applies two transformations to ``tool_response`` before storing it.
    First, ``redact_secrets_for_ui`` replaces structured secrets (GitLab tokens,
    JWTs, AWS keys, etc.) and high-entropy strings (Azure storage keys, generic
    API blobs) with ``[REDACTED]``, using raised entropy thresholds so that git
    SHAs, checksums, and UUIDs are left intact.
    Second, string responses are capped at ``TOOL_RESPONSE_MAX_DISPLAY_MSG``
    characters to prevent bloating ui_chat_log payloads and checkpoints with
    large tool outputs.
    """
    info = ToolInfo(name=name, args=args)
    if tool_response is not None:
        tool_response = redact_secrets_for_ui(tool_response, tool_name=name)
        if (
            isinstance(tool_response, str)
            and len(tool_response) > TOOL_RESPONSE_MAX_DISPLAY_MSG
        ):
            tool_response = tool_response[:TOOL_RESPONSE_MAX_DISPLAY_MSG]
        info["tool_response"] = tool_response
    return info


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
    denied_tools: list[str] | None


DuoWorkflowStateType = Union[WorkflowState, ChatWorkflowState]
