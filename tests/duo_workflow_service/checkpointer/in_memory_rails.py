"""A `GitLabWorkflow` saver for each request against an in-memory GitLab, for tests that run a graph over several
requests.

`GitLabWorkflow` stores a task's interrupt writes and no output, and GitLab only adds rows. Every request builds a
fresh saver, so nothing survives between requests but what GitLab keeps.
"""

import base64
import json
from typing import Any, Callable, Optional, cast
from unittest.mock import MagicMock
from urllib.parse import parse_qs, urlparse

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from duo_workflow_service.checkpointer.gitlab_workflow import GitLabWorkflow
from duo_workflow_service.checkpointer.gitlab_workflow_utils import (
    uncompress_checkpoint,
)
from duo_workflow_service.gitlab.gitlab_api import WorkflowConfig
from duo_workflow_service.gitlab.http_client import GitlabHttpClient, GitLabHttpResponse
from duo_workflow_service.json_encoder.encoder import CustomEncoder

__all__ = [
    "GraphQLTipRails",
    "InMemoryRails",
    "SaverFactory",
    "gitlab_savers",
    "memory_savers",
]

# Makes the saver one request runs on.
SaverFactory = Callable[[], BaseCheckpointSaver]


class InMemoryRails:
    """Serves the checkpoint API over REST, with each checkpoint's writes."""

    # GitLab refuses a longer batch: `MAX_WRITES_PER_BATCH` in its `CreateCheckpointWriteBatchService`.
    MAX_WRITES_PER_BATCH = 10

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []
        self._last_write_id = 0

    def workflow_config(self) -> dict[str, Any]:
        top = [row for row in self._rows if row["checkpoint_ns"] is None]
        first = min(top, key=lambda row: row["thread_ts"]) if top else None
        return {
            "first_checkpoint": {"thread_ts": first["thread_ts"]} if first else None,
            # Unset, so the tip is read over REST with its writes.
            "latest_checkpoint": None,
            "workflow_status": "running",
            "incremental_checkpoints_enabled": False,
            "agent_privileges_names": [],
            "pre_approved_agent_privileges_names": [],
            "mcp_enabled": False,
            "allow_agent_to_request_user": True,
            "archived": False,
        }

    async def aget(
        self, path: str, object_hook: Optional[Callable] = None, **_kwargs: Any
    ) -> GitLabHttpResponse:
        if "/checkpoints" not in path:
            return GitLabHttpResponse(status_code=200, body={"status": "running"})
        query = parse_qs(urlparse(path).query, keep_blank_values=True)
        rows = sorted(self._rows, key=lambda row: row["thread_ts"], reverse=True)
        if "checkpoint_ns" in query:
            wanted = query["checkpoint_ns"][0] or None
            rows = [row for row in rows if row["checkpoint_ns"] == wanted]
        per_page = int(query.get("per_page", ["20"])[0])
        page = int(query.get("page", ["1"])[0])
        served = [
            # GitLab reads a checkpoint's writes with no order, so they come back newest first here.
            {**row, "checkpoint_writes": row["writes"][::-1]}
            for row in rows[(page - 1) * per_page : page * per_page]
        ]
        body = json.loads(
            json.dumps(served, cls=CustomEncoder), object_hook=object_hook
        )
        next_page = str(page + 1) if len(rows) > page * per_page else ""
        return GitLabHttpResponse(
            status_code=200, body=body, headers={"X-Next-Page": next_page}
        )

    async def apost(
        self, path: str, body: Optional[str] = None, **_kwargs: Any
    ) -> GitLabHttpResponse:
        payload = json.loads(body or "{}")
        if path.endswith("/checkpoint_writes_batch"):
            if len(payload["checkpoint_writes"]) > self.MAX_WRITES_PER_BATCH:
                return GitLabHttpResponse(
                    status_code=400, body={"message": "Too many checkpoint writes"}
                )
            row = next(
                row for row in self._rows if row["thread_ts"] == payload["thread_ts"]
            )
            # GitLab inserts in bulk, so a task's later interrupt is added and never replaces its earlier one.
            for write in payload["checkpoint_writes"]:
                self._last_write_id += 1
                row["writes"].append({**write, "id": self._last_write_id})
            return GitLabHttpResponse(status_code=200, body={})
        checkpoint = uncompress_checkpoint(payload["compressed_checkpoint"])
        self._rows.append(
            {
                "id": len(self._rows) + 1,
                "thread_ts": payload["thread_ts"],
                "parent_ts": payload.get("parent_ts"),
                "checkpoint_ns": payload.get("checkpoint_ns") or None,
                "checkpoint": json.loads(json.dumps(checkpoint, cls=CustomEncoder)),
                "metadata": payload.get("metadata") or {},
                "writes": [],
            }
        )
        return GitLabHttpResponse(status_code=200, body={})

    async def apatch(self, _path: str, **_kwargs: Any) -> GitLabHttpResponse:
        return GitLabHttpResponse(status_code=200, body={})

    def stored_interrupts(self) -> list[tuple[str, bool]]:
        """Every interrupt row GitLab holds, oldest first, as its task and whether it has an interrupt in it.

        A row with none is what the saver stores for a task that finished after it paused.
        """
        writes = sorted(
            (write for row in self._rows for write in row["writes"]),
            key=lambda write: write["id"],
        )
        return [
            # An empty msgpack array is the one byte 0x90.
            (write["task"], base64.b64decode(write["data"]) != b"\x90")
            for write in writes
            if write["channel"] == "__interrupt__"
        ]


class GraphQLTipRails(InMemoryRails):
    """Hands each request the tip as GraphQL's `latestCheckpoint` does, which carries no `checkpoint_writes`."""

    def workflow_config(self) -> dict[str, Any]:
        config = super().workflow_config()
        top = [row for row in self._rows if row["checkpoint_ns"] is None]
        if top:
            tip = max(top, key=lambda row: row["thread_ts"])
            config["latest_checkpoint"] = {
                "threadTs": tip["thread_ts"],
                "parentTs": tip["parent_ts"],
                "checkpoint": json.dumps(tip["checkpoint"], cls=CustomEncoder),
                "metadata": json.dumps(tip["metadata"], cls=CustomEncoder),
            }
        return config


def gitlab_savers(rails: Optional[InMemoryRails] = None) -> SaverFactory:
    """A fresh `GitLabWorkflow` per request, all on one in-memory GitLab."""
    served_by = rails or InMemoryRails()

    def saver() -> BaseCheckpointSaver:
        return GitLabWorkflow(
            client=cast(GitlabHttpClient, served_by),
            workflow_id="1",
            workflow_type=MagicMock(),
            workflow_config=cast(WorkflowConfig, served_by.workflow_config()),
            internal_event_client=MagicMock(),
            billing_event_service=MagicMock(),
        )

    return saver


def memory_savers() -> SaverFactory:
    """LangGraph's `InMemorySaver`, which stores every task's writes."""
    saver = InMemorySaver()
    return lambda: saver
