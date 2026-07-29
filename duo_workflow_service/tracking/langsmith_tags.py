import asyncio
from typing import Any

import structlog
from langsmith import get_current_run_tree
from langsmith.run_trees import RunTree

__all__ = ["tag_current_run"]

log = structlog.stdlib.get_logger("langsmith_tags")


async def tag_current_run(tag: str, metadata: dict[str, Any]) -> None:
    """Attach a tag and metadata to the current LangSmith run tree and its trace root, if any.

    No-op when LangSmith tracing is disabled or no active run tree exists.

    Tagging only the current run (e.g. a specific agent-node run several levels
    deep in a LangGraph trace) doesn't surface in the top-level traces list in
    the LangSmith UI, which only reflects the root run's tags — so the root run
    is tagged too, in addition to the current one, so limit events are
    filterable/searchable at the trace level as well as visible on the specific
    run that triggered them.

    Args:
        tag: The tag to add to the run (e.g. "soft_limit_reached").
        metadata: Additional metadata fields to attach to the run.
    """
    run_tree = get_current_run_tree()
    if run_tree is None:
        return

    _tag_run(run_tree, tag, metadata)

    if run_tree.trace_id and run_tree.trace_id != run_tree.id:
        await _tag_root_run(run_tree, tag, metadata)


def _tag_run(run_tree: RunTree, tag: str, metadata: dict[str, Any]) -> None:
    # add_tags() extends the tag list unconditionally; guard against duplicate
    # entries if the same run gets tagged more than once (e.g. a node that
    # re-runs after already crossing its limit once).
    if tag not in (run_tree.tags or []):
        run_tree.add_tags([tag])
    run_tree.add_metadata(metadata)


async def _tag_root_run(run_tree: RunTree, tag: str, metadata: dict[str, Any]) -> None:
    """Tag the trace's root run via the LangSmith API.

    Unlike `_tag_run` (a free in-memory mutation that rides along with the
    SDK's own batched flush of `run_tree`), there's no in-memory object
    reference to the root run to mutate here: LangGraph's node-tracing path
    only populates `RunTree.parent_run_id`/`trace_id` (plain IDs), never
    `RunTree.parent_run` (the object reference `RunTree.create_child()` sets,
    which only the `@traceable`-decorator path uses). Reaching the root
    therefore needs a real read-modify-write API round trip. `Client` is
    synchronous, so both calls run in a thread to avoid blocking the event
    loop. This is a rare, non-hot-path event (fires once per session when a
    limit is actually hit), so the small race window against another
    concurrent tag write on the same root is accepted rather than engineering
    around it.
    """
    client = run_tree.client
    try:
        root = await asyncio.to_thread(client.read_run, run_tree.trace_id)
        tags = list(root.tags or [])
        if tag not in tags:
            tags.append(tag)
        extra = dict(root.extra or {})
        root_metadata = dict(extra.get("metadata") or {})
        root_metadata.update(metadata)
        extra["metadata"] = root_metadata
        await asyncio.to_thread(
            client.update_run, run_tree.trace_id, tags=tags, extra=extra
        )
    except Exception:  # pylint: disable=broad-except
        # Tagging the root is a best-effort observability nicety; never let a
        # LangSmith API hiccup break the actual agent workflow.
        log.warning(
            "Failed to tag trace root run in LangSmith",
            trace_id=str(run_tree.trace_id),
            tag=tag,
            exc_info=True,
        )
