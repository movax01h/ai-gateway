"""Checkpoint write mode, shared by the checkpointer and the graph nodes."""

from contextvars import ContextVar

from langgraph.types import Overwrite

from duo_workflow_service.client_capabilities import is_client_capable
from duo_workflow_service.entities.state import UiChatLog

incremental_checkpoints_enabled: ContextVar[bool] = ContextVar(
    "incremental_checkpoints_enabled", default=False
)


def write_incremental_only(incremental_enabled: bool | None = None) -> bool:
    """True when the instance stores only the checkpoint header and the channel blobs.

    Every channel value is then written once as a blob delta, so a node can shrink a list channel without losing
    history. The checkpointer passes the workflow flag explicitly; nodes fall back to the value published for the
    current run.
    """
    if incremental_enabled is None:
        incremental_enabled = incremental_checkpoints_enabled.get()
    return incremental_enabled and is_client_capable("incremental_checkpoints_only")


def compaction_ui_chat_log_update(
    entries: list[UiChatLog],
) -> list[UiChatLog] | Overwrite:
    """``ui_chat_log`` update for a step that rewrote the conversation history.

    Once the blobs hold the history, the channel is replaced instead of appended to, so state keeps only this step's
    entries and the group-start snapshot stops growing with the session (gitlab-org/gitlab#628017). Rails folds the full
    history back from the blobs. Older instances keep the appended list.
    """
    if write_incremental_only():
        return Overwrite(entries)
    return entries
