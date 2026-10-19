import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Send

from duo_workflow_service.entities.image_blocks import strip_image_payloads
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
            if "content" in data:
                # Applies to every message role: an image can arrive as a user
                # attachment or as a tool result, and either would otherwise be
                # persisted (and re-read) on every checkpoint for the rest of
                # the session.
                #
                # Note this fires on *every* write, so an image does not survive
                # its own turn: any interrupt checkpoints here and then resumes
                # by reloading the stripped history. A mid-turn tool approval is
                # enough. Known v1 limitation -- see the `image_blocks` module
                # docstring for the reasoning and the durable fix.
                data["content"] = strip_image_payloads(data["content"])
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


# A `\u0000` escape that is not itself escaped: it must be preceded by an even
# number of backslashes (the `(\\\\)*` group), or the backslash belongs to a
# literal `\` in the string, e.g. the text `\u0000` is encoded as `\\u0000`.
_NUL_ESCAPE_RE = re.compile(r"(?<!\\)((?:\\\\)*)\\u0000")


def dumps_checkpoint(obj: Any) -> str:
    """Encode checkpoint data as JSON that PostgreSQL can store.

    Tool output can contain NUL characters, for example terminal escape sequences in a
    command's output. `jsonb` cannot represent them, so GitLab rejects the whole
    checkpoint with `PG::UntranslatableCharacter` and the workflow fails. They carry no
    meaning for the model, so they are dropped.
    """
    return _NUL_ESCAPE_RE.sub(r"\1", json.dumps(obj, cls=CustomEncoder))
