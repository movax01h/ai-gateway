from enum import StrEnum
from typing import Annotated, Any, Dict, List, NotRequired, Optional, TypedDict, Union

import structlog
from langchain_core.messages import BaseMessage
from pydantic import BaseModel

from ai_gateway.model_metadata import current_model_metadata_context
from duo_workflow_service.conversation.trimmer import (
    LEGACY_MAX_CONTEXT_TOKENS,
    trim_conversation_history,
)
from duo_workflow_service.entities.event import WorkflowEvent
from duo_workflow_service.gitlab.gitlab_api import Namespace, Project
from duo_workflow_service.workflows.type_definitions import AdditionalContext
from lib.feature_flags import FeatureFlag, is_feature_enabled

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
    message_id: Optional[str]


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
        combined_messages = existing_messages + new_messages

        # If feature flag is enabled, trim to max context window for the specific model
        # Otherwise trim to the old 400K context window size for all models.
        model_metadata = current_model_metadata_context.get()

        reduced[agent_name] = trim_conversation_history(
            messages=combined_messages,
            component_name=agent_name,
            max_context_tokens=(
                model_metadata.llm_definition.max_context_tokens
                if is_feature_enabled(FeatureFlag.AI_PER_MODEL_CONTEXT_WINDOW)
                and model_metadata is not None
                else LEGACY_MAX_CONTEXT_TOKENS
            ),
            model_name=(
                model_metadata.llm_definition.name
                if model_metadata is not None
                else "claude"  # Claude for safer token counting (~+30%)
            ),
        )

    return reduced


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
