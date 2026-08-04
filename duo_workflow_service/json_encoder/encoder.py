import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Send

from duo_workflow_service.entities.state import (
    AdditionalContext,
    ApprovalStateRejection,
)


class CustomEncoder(json.JSONEncoder):
    """Custom JSON encoder class that extends json.JSONEncoder to handle langchain object types."""

    def default(self, o: Any) -> Any:
        """Overrides the default method to provide custom encoding for specific types.

        Args:
            o: The object to encode.

        Returns:
            JSON-serializable representation of the object.
        """
        if isinstance(
            o,
            (
                SystemMessage,
                HumanMessage,
                AIMessage,
                ToolMessage,
                ApprovalStateRejection,
                AdditionalContext,
            ),
        ):
            data = o.model_dump()
            data.update({"type": o.__class__.__name__})
            return data
        if isinstance(o, Send):
            # Raw `Send` packets legitimately end up in a checkpoint's
            # `channel_values["__pregel_tasks"]` whenever a native-`Send`
            # dispatch (e.g. concurrent `delegate_task` fan-out) is
            # pending/un-consumed at the moment the checkpoint is persisted
            # (see langgraph.pregel._loop.LoopProtocol._put_checkpoint).
            # LangGraph's own serde knows how to round-trip `Send`, but this
            # custom JSON path doesn't -- without this branch, persisting
            # such a checkpoint raises
            # `TypeError: Object of type Send is not JSON serializable`.
            return {"type": "Send", "node": o.node, "arg": o.arg}
        return super().default(o)
