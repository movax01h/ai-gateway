from enum import StrEnum
from typing import NotRequired, TypedDict

from duo_workflow_service.agent_platform.v1.state.base import (
    BaseIOKey,
    FlowState,
    FlowStateKeys,
    IOKey,
    IOKeyFactory,
    IOKeyTemplate,
    RuntimeIOKey,
    conversation_history_replace_reducer,
    create_nested_dict,
    get_vars_from_state,
    merge_nested_dict,
    merge_nested_dict_reducer,
)

__all__ = [
    "BaseIOKey",
    "FlowEvent",
    "FlowEventType",
    "FlowState",
    "FlowStateKeys",
    "IOKey",
    "IOKeyFactory",
    "IOKeyTemplate",
    "RuntimeIOKey",
    "conversation_history_replace_reducer",
    "create_nested_dict",
    "get_vars_from_state",
    "merge_nested_dict",
    "merge_nested_dict_reducer",
]


class FlowEventType(StrEnum):
    RESPONSE = "response"  # Agent sends response via NewCheckpoint protobuf message
    APPROVE = "approve"  # User approval via Approval.Approved protobuf message
    REJECT = "reject"  # User rejection via Approval.Rejected without message field
    MODIFY = "modify"  # User rejection via Approval.Rejected with message field


class FlowEvent(TypedDict):
    event_type: FlowEventType
    message: NotRequired[str]
