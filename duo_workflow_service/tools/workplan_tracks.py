"""Turn the independent tracks of a Workplan into child work items.

The workplan flow runs this tool as a deterministic step after the planner.
The planner decides what the tracks are. This tool decides nothing about the
plan: it checks that the tracks can run at the same time, and then writes them
to GitLab in an order that lets every dependency become a link.

A flow can link a work item only when it creates the item
(``linkedItemsWidget`` on ``workItemCreate``): ``workItemAddLinkedItems`` does
not accept the ``ai_workflows`` token scope. So an item must exist before an
item that it blocks, and this tool creates the tracks in dependency order.
"""

import json
import re
from typing import Any, ClassVar, Dict, List, NamedTuple, Optional, Type

import structlog
from langchain_core.tools import ToolException
from packaging.version import Version
from pydantic import BaseModel, Field

from duo_workflow_service.security.tool_output_security import ToolTrustLevel
from duo_workflow_service.tools.version_compatibility import (
    supports_agent_plan_widget,
)
from duo_workflow_service.tools.work_item import CreateWorkItemNote
from duo_workflow_service.tools.work_items.base_tool import (
    LINKED_ITEMS_MAX_IDS,
    WorkItemBaseTool,
)
from duo_workflow_service.tools.work_items.queries.graphql_queries import (
    GET_WORKPLAN_TRACKS_LINKS_QUERY,
    GET_WORKPLAN_TRACKS_PARENT_QUERY,
)

__all__ = [
    "WorkplanPublishTracks",
    "WorkplanPublishTracksInput",
    "WorkplanTrack",
    "plan_tracks",
]

log = structlog.stdlib.get_logger(__name__)

# With at most this many tracks, a track has at most _MAX_TRACKS - 1 blockers,
# which fits in the LINKED_ITEMS_MAX_IDS links one workItemCreate accepts.
_MAX_TRACKS = LINKED_ITEMS_MAX_IDS
_TRACK_KEY_PATTERN = re.compile(r"[a-z0-9]([a-z0-9-]{0,38}[a-z0-9])?")
# The marker is an HTML comment, so it does not show in the rendered
# description. A later run reads it to find the child of a track.
_TRACK_MARKER_TEMPLATE = "<!-- workplan-track: {key} -->"
_TRACK_MARKER_PATTERN = re.compile(r"<!-- workplan-track: ([a-z0-9-]+) -->")
# The workplan flow plans Issues, and the child of an Issue is a Task.
_PARENT_TYPE = "Issue"
_CHILD_TYPE = "Task"


class WorkplanTrack(BaseModel):
    key: str = Field(
        description="Stable identifier of the track: lowercase letters, digits and hyphens."
    )
    title: str = Field(description="Title of the child work item.")
    workplan: str = Field(
        description="The Workplan of the child work item. It must stand on its own."
    )
    files: List[str] = Field(
        default_factory=list,
        description="Every file the track creates or edits. A path that ends in '/' is a directory.",
    )
    depends_on: List[str] = Field(
        default_factory=list,
        description="Keys of the tracks that must close before this track starts.",
    )


class WorkplanPublishTracksInput(BaseModel):
    url: Optional[str] = Field(
        default=None,
        description="URL of the work item that the plan belongs to. Without it, the tool creates nothing.",
    )
    tracks: Optional[List[WorkplanTrack]] = Field(
        default=None,
        description="The tracks of the plan. Fewer than two tracks creates nothing.",
    )


class TrackPlan(NamedTuple):
    """The result of the checks.

    ``order`` is empty when ``problems`` is not.
    """

    order: List[str]
    concurrent: Dict[str, List[str]]
    problems: List[str]


def _normalize_path(path: str) -> str:
    path = path.strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    path = path.lstrip("/")
    return "" if path == "." else path


def _paths_overlap(first: str, second: str) -> bool:
    """True when the paths are the same, or one is inside the other.

    One path cannot be both a file and the parent directory of another path, so a directory needs no trailing slash. An
    empty path is the whole repository, so it overlaps every path.
    """
    first, second = first.rstrip("/"), second.rstrip("/")
    return (
        not first
        or not second
        or first == second
        or second.startswith(first + "/")
        or first.startswith(second + "/")
    )


def _ancestors(key: str, deps: Dict[str, List[str]]) -> set[str]:
    """Every track that must finish before ``key`` starts, directly or not."""
    seen: set[str] = set()
    stack = list(deps.get(key, []))
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        stack.extend(deps.get(current, []))
    return seen


def _topological_order(keys: List[str], deps: Dict[str, List[str]]) -> List[str]:
    """Kahn's algorithm. Keeps the input order among tracks that are ready together.

    Returns fewer keys than it was given when the dependencies have a cycle.
    """
    remaining = {key: set(deps.get(key, [])) for key in keys}
    order: List[str] = []
    while True:
        ready = [key for key in keys if key in remaining and not remaining[key]]
        if not ready:
            break
        for key in ready:
            order.append(key)
            del remaining[key]
        for pending in remaining.values():
            pending.difference_update(ready)
    return order


def plan_tracks(tracks: List[WorkplanTrack]) -> TrackPlan:
    """Check that the tracks can be published, and find the creation order.

    The checks are the parts of the split that must not depend on a model:

    - The keys are valid and unique.
    - Every dependency names another track, and the dependencies have no cycle.
    - Every track names the files it changes.
    - Two tracks that change the same file are ordered: one of them depends on the other, directly or not. Two agents
        that edit one file at the same time make merge requests that conflict.

    Returns:
        A TrackPlan. ``concurrent`` maps each key to the keys of the tracks
        that can run at the same time as it.
    """
    problems: List[str] = []
    keys = [track.key for track in tracks]
    by_key = {track.key: track for track in tracks}

    for key in keys:
        # fullmatch: with match, `$` also matches before a final newline, and
        # the marker of such a key does not read back.
        if not _TRACK_KEY_PATTERN.fullmatch(key):
            problems.append(f"Track key '{_key(key)}' is not valid.")
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        problems.append(
            f"Track keys are not unique: {', '.join(_key(key) for key in duplicates)}."
        )

    deps: Dict[str, List[str]] = {}
    for track in tracks:
        unique_deps = list(dict.fromkeys(track.depends_on))
        for dep in unique_deps:
            if dep == track.key:
                problems.append(f"Track '{_key(track.key)}' depends on itself.")
            elif dep not in by_key:
                problems.append(
                    f"Track '{_key(track.key)}' depends on unknown track '{_key(dep)}'."
                )
        deps[track.key] = [
            dep for dep in unique_deps if dep in by_key and dep != track.key
        ]

        if not [path for path in track.files if path.strip()]:
            problems.append(f"Track '{_key(track.key)}' names no files.")

    if problems:
        return TrackPlan(order=[], concurrent={}, problems=problems)

    order = _topological_order(keys, deps)
    if len(order) != len(keys):
        cycle = sorted(set(keys) - set(order))
        return TrackPlan(
            order=[],
            concurrent={},
            problems=[
                f"The dependencies of these tracks form a cycle: {', '.join(_key(key) for key in cycle)}."
            ],
        )

    ancestors = {key: _ancestors(key, deps) for key in keys}
    files = {
        track.key: [_normalize_path(path) for path in track.files if path.strip()]
        for track in tracks
    }
    concurrent: Dict[str, List[str]] = {key: [] for key in keys}

    for index, first in enumerate(keys):
        for second in keys[index + 1 :]:
            ordered = first in ancestors[second] or second in ancestors[first]
            if ordered:
                continue
            concurrent[first].append(second)
            concurrent[second].append(first)
            shared = sorted(
                {
                    path
                    for path in files[first]
                    for other in files[second]
                    if _paths_overlap(path, other)
                }
            )
            if shared:
                problems.append(
                    f"Tracks '{_key(first)}' and '{_key(second)}' both change {', '.join(shared)}, "
                    "and neither depends on the other."
                )

    if problems:
        return TrackPlan(order=[], concurrent={}, problems=problems)

    return TrackPlan(order=order, concurrent=concurrent, problems=[])


def _child_url(parent_url: str, iid: Any) -> str:
    return re.sub(
        r"/-/(?:issues|work_items)/\d+.*$", f"/-/work_items/{iid}", parent_url
    )


def _marker_key(description: Optional[str]) -> Optional[str]:
    match = _TRACK_MARKER_PATTERN.search(description or "")
    return match.group(1) if match else None


def _child_description(parent_url: str, key: str) -> str:
    return (
        f"One track of the Workplan of {parent_url}.\n\n"
        f"{_TRACK_MARKER_TEMPLATE.format(key=key)}"
    )


def _child_workplan(
    track: WorkplanTrack,
    tracks: Dict[str, WorkplanTrack],
    concurrent: List[str],
    blockers: List[Dict[str, Any]],
    parent_url: str,
) -> str:
    """The track's own Workplan, followed by the parts that code can state exactly."""
    own = [_normalize_path(path) for path in track.files if path.strip()]
    others = sorted(
        {
            _normalize_path(path)
            for key in concurrent
            for path in tracks[key].files
            if path.strip()
        }
        - set(own)
    )

    sections = [track.workplan.rstrip(), "", "## Files", ""]
    sections.append("This track creates or edits these files:")
    sections.append("")
    sections.extend(f"- `{path}`" for path in own)

    if others:
        sections.extend(
            [
                "",
                "Other tracks change these files at the same time. Do not edit them:",
                "",
            ]
        )
        sections.extend(f"- `{path}`" for path in others)

    sections.extend(["", "## Order", ""])
    if blockers:
        sections.append("Start this track after these work items close:")
        sections.append("")
        sections.extend(f"- [{item['title']}]({item['url']})" for item in blockers)
        sections.append("")
    sections.append(
        f"This track is one part of {parent_url}. Implement this track only."
    )

    return "\n".join(sections) + "\n"


def _short(text: Any, limit: int = 300) -> str:
    """Error text for a public note: the client can put a whole HTTP body in it."""
    text = str(text)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _key(key: Any) -> str:
    """A track key for a public note: the schema does not limit its length."""
    return _short(key, 60)


def _track_name(key: Any, iid: Any = None) -> str:
    """A track and its child item for a public note, for example `go` (#12)."""
    return f"`{_key(key)}`" + (f" (#{iid})" if iid else "")


def _skip_reason(url: Optional[str], tracks: List[WorkplanTrack]) -> Optional[str]:
    """Why the tool creates nothing before it reads the parent, if it does not."""
    if not url:
        # The flow passes the URL of the session's work item, which is empty
        # when the session has no work item.
        return "The session has no work item, so there is no parent for the tracks."
    if len(tracks) < 2:
        return "The plan has fewer than two tracks, so it stays one work item."
    if not supports_agent_plan_widget():
        return "This GitLab version has no Workplan widget for the child items."
    return None


def _status(entries: List[Dict[str, Any]], links_read: bool) -> str:
    """One word for the outcome.

    The order is from the worst outcome to the best.
    """
    if any(entry["result"] not in ("created", "exists") for entry in entries):
        return "partial"
    if not links_read:
        return "unverified"
    if any(entry.get("links_missing") for entry in entries):
        return "incomplete"
    return "published"


def _outcome_note(result: Dict[str, Any]) -> str:
    """The note that tells the author what to check in the child work items."""
    if result["status"] == "skipped":
        lines = [
            "Generate Workplan did not split the plan into child work items. "
            "The result is `skipped`."
        ]
    else:
        lines = [
            "Check the child work items of this plan. Generate Workplan split the "
            f"plan with this result: `{result['status']}`."
        ]
    if result.get("reason"):
        lines.append("")
        lines.append(result["reason"])
    for problem in result.get("problems", []):
        lines.append(f"- {problem}")

    details = []
    for entry in result.get("tracks", []):
        name = _track_name(entry["key"], entry.get("iid"))
        if entry["result"] in ("failed", "not_attempted"):
            error = _short(entry["error"]) if entry.get("error") else ""
            details.append(f"- {name}: not created. {error}".rstrip())
        if entry.get("links_missing"):
            details.append(
                f"- {name}: GitLab has no BLOCKED_BY link to "
                f"{', '.join(_track_name(key) for key in entry['links_missing'])}. "
                "Add the link by hand."
            )
    details.extend(f"- {_short(warning)}" for warning in result.get("warnings", []))
    if details:
        lines.append("")
        lines.extend(details)

    if result["status"] == "partial":
        lines.append("")
        lines.append(
            "Run Generate Workplan again to create the missing child work items. "
            "A second run does not create a child item twice."
        )
    return "\n".join(lines)


class WorkplanPublishTracks(WorkItemBaseTool):
    name: str = "workplan_publish_tracks"
    description: str = (
        "Create one child work item per track of a Workplan, in dependency "
        "order, with a BLOCKED_BY link for each dependency. Creates nothing "
        "when the tracks fail a check, and never creates a track twice. "
        "Deterministic, no LLM call."
    )
    args_schema: Type[BaseModel] = WorkplanPublishTracksInput
    tool_version: ClassVar[Version] = Version("0.1.0")
    # No `allowed_flows`: the dry-run flow validator builds its registry with
    # no flow id, so it would drop a flow-limited tool from a v1 flow. The
    # tool can do nothing that `create_work_item` cannot.
    # The result carries the titles of existing child items and GitLab error
    # text, which users can write.
    trust_level: ToolTrustLevel = ToolTrustLevel.UNTRUSTED_USER_CONTENT

    def format_display_message(
        self, args: Any, tool_response: Any = None
    ) -> Optional[str]:
        tracks = getattr(args, "tracks", None) or []
        try:
            status = json.loads(tool_response).get("status") if tool_response else None
        except (TypeError, ValueError):
            status = None
        if status is None:
            return f"Publish {len(tracks)} tracks of the Workplan as child work items"
        return f"Publish the tracks of the Workplan as child work items: {status}"

    async def _execute(
        self,
        url: Optional[str] = None,
        tracks: Optional[List[Any]] = None,
        **_kwargs: Any,
    ) -> str:
        parsed = [
            track
            if isinstance(track, WorkplanTrack)
            else WorkplanTrack.model_validate(track)
            for track in (tracks or [])
        ]

        reason = _skip_reason(url, parsed)
        # `not url` only narrows the type: _skip_reason gives a reason for it.
        if reason or not url:
            return json.dumps({"status": "skipped", "reason": reason})

        resolved = self._parse_work_item_url(url)
        if resolved.parent.type != "project":
            raise ToolException(
                "The workplan flow splits only a work item in a project."
            )

        parent = await self._fetch_parent(
            resolved.parent.full_path, resolved.work_item_iid
        )
        parent_url = parent.get("webUrl") or url
        parent_type = (parent.get("workItemType") or {}).get("name", "")

        if parent_type != _PARENT_TYPE:
            # The planner returned two or more tracks, so its announcement
            # promised child items: the note tells the author why there are
            # none. The type is read by name: a flow token cannot read
            # allowedChildTypes yet, so a custom type that works as an Issue
            # is not split (a follow-up). A backtick would end the code span.
            type_name = _short(parent_type or "unknown", 60).replace("`", "")
            return await self._finish(
                parent_url,
                {
                    "status": "skipped",
                    "reason": "The planner returned tracks, but Generate Workplan "
                    f"splits only a work item of the type named `{_PARENT_TYPE}`. "
                    f"The type of this work item is `{type_name}`, so the plan "
                    "stays one work item.",
                },
            )

        by_key = {track.key: track for track in parsed}
        children, total = self._children(parent)
        existing = self._existing_children(children)
        warnings: List[str] = []

        plan = self._plan(parsed, parent, children, total)
        if plan.problems:
            return await self._finish(
                parent_url,
                {
                    "status": "refused",
                    "reason": "The split failed a check, so no child work item was created.",
                    "problems": plan.problems,
                },
            )

        published: Dict[str, Dict[str, Any]] = {}
        entries: List[Dict[str, Any]] = []
        failure: Optional[str] = None

        for key in plan.order:
            track = by_key[key]

            if failure:
                entries.append({"key": key, "result": "not_attempted"})
                continue

            if key in existing:
                child = existing[key]
                published[key] = self._published(child, parent_url, track.title)
                entries.append(
                    {
                        "key": key,
                        "result": "exists",
                        "iid": child.get("iid"),
                        **published[key],
                    }
                )
                continue

            blockers = [published[dep] for dep in dict.fromkeys(track.depends_on)]
            kwargs: Dict[str, Any] = {
                "title": track.title,
                "description": _child_description(parent_url, key),
                "agent_plan": _child_workplan(
                    track, by_key, plan.concurrent[key], blockers, parent_url
                ),
                "hierarchy_widget": {"parent_id": parent["id"]},
            }
            if blockers:
                kwargs["linked_items_widget"] = {
                    "link_type": "BLOCKED_BY",
                    "work_items_ids": [item["id"] for item in blockers],
                }

            entry, failure = await self._create_track(
                resolved, track, kwargs, parent_url, warnings
            )
            entries.append(entry)
            if entry.get("id"):
                published[key] = {k: entry[k] for k in ("id", "title", "url")}

        links_read = await self._check_links(
            resolved.parent.full_path, by_key, entries, published, warnings
        )

        result: Dict[str, Any] = {
            "status": _status(entries, links_read),
            "tracks": entries,
        }
        if warnings:
            result["warnings"] = warnings
        return await self._finish(parent_url, result)

    def _plan(
        self,
        parsed: List[WorkplanTrack],
        parent: Dict[str, Any],
        children: List[Dict[str, Any]],
        total: int,
    ) -> TrackPlan:
        """Check the tracks against the parent and the children of an earlier run.

        Every check runs, so the note names every problem at once.
        """
        problems: List[str] = []
        if len(parsed) > _MAX_TRACKS:
            problems.append(
                f"The plan has {len(parsed)} tracks. The maximum is {_MAX_TRACKS}."
            )
        # create_work_item drops a parent id that fails this check, and it
        # creates the child with no parent. So create nothing.
        if (
            self._build_hierarchy_widget(
                {"hierarchy_widget": {"parent_id": parent["id"]}}, []
            )
            is None
        ):
            problems.append(
                f"The work item id {parent['id']} is not a work item global id, "
                "so a child cannot get it as its parent."
            )
        # The guard below needs every child: a marked child on a later page
        # would get a second copy.
        if total > len(children):
            problems.append(
                f"The work item has {total} child items, and the tool reads only the "
                f"first {len(children)}. So it cannot check all the tracks of an "
                "earlier plan."
            )
        # A run after a failed create has the same keys, and it creates the
        # missing tracks. An open child of a track that is not in this plan
        # means that the plan changed: create nothing, so no track gets a
        # second copy. A closed child does not stop a new split. A re-plan of
        # existing child items is a follow-up.
        # Read every child, not one per marker: a copy of a child has the same
        # marker, and the copy can be open while the first child is closed.
        # The note names only these children: a closed child counts as done
        # for its track, so closing a child of a track in the plan skips it.
        keys = {track.key for track in parsed}
        dropped: List[str] = []
        for child in children:
            marker = _marker_key(child.get("description"))
            if marker and marker not in keys and child.get("state") != "CLOSED":
                dropped.append(_track_name(marker, child.get("iid")))
        if dropped:
            problems.append(
                "The work item has open child items from an earlier plan. This plan "
                f"does not have their tracks: {', '.join(dropped)}. The plan changed. "
                "To split the new plan, close only these child items. A closed child "
                "item counts as done for its track, so Generate Workplan does not "
                "create that track again."
            )
        checked = plan_tracks(parsed)
        if problems or checked.problems:
            return TrackPlan(
                order=[], concurrent={}, problems=problems + checked.problems
            )
        return checked

    @staticmethod
    def _existing_children(children: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """The child of each track that already has one, found by its marker.

        A run that stopped after a failed create leaves some children, so the next run must not create them twice. A
        child without a marker is never a track's child: a human made it.
        """
        existing: Dict[str, Dict[str, Any]] = {}
        for child in children:
            marker = _marker_key(child.get("description"))
            if marker and marker not in existing:
                existing[marker] = child
        return existing

    @staticmethod
    def _published(
        child: Dict[str, Any], parent_url: str, title: str
    ) -> Dict[str, Any]:
        return {
            "id": child["id"],
            "title": child.get("title") or title,
            "url": child.get("webUrl") or _child_url(parent_url, child.get("iid")),
        }

    async def _fetch_parent(self, full_path: str, iid: Optional[int]) -> Dict[str, Any]:
        response = await self.gitlab_client.graphql(
            GET_WORKPLAN_TRACKS_PARENT_QUERY, {"fullPath": full_path, "iid": str(iid)}
        )
        nodes = (((response or {}).get("project") or {}).get("workItems") or {}).get(
            "nodes"
        ) or []
        if not nodes or not nodes[0].get("id"):
            raise ToolException(f"Work item {iid} not found in {full_path}")
        return nodes[0]

    @staticmethod
    def _children(parent: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        """The children that the read returned, and the count of all children."""
        for widget in parent.get("widgets") or []:
            children = (widget or {}).get("children")
            if children is None:
                continue
            nodes = children.get("nodes") or []
            return nodes, children.get("count") or len(nodes)
        return [], 0

    async def _create_track(
        self,
        resolved: Any,
        track: WorkplanTrack,
        kwargs: Dict[str, Any],
        parent_url: str,
        warnings: List[str],
    ) -> tuple[Dict[str, Any], Optional[str]]:
        """Create the child of one track.

        Returns:
            The entry of the track, and the reason to stop creating, if any. The
            tracks after this one may depend on it, so a failure stops creation.
        """
        key = track.key
        # Deliberately resilient, like the draft notes of `mr_review`: the
        # children created before a failure exist, and the result must say so.
        # Any exception counts: the client raises GitLabGraphQLError for a
        # GraphQL error and a plain Exception for a timeout.
        try:
            response = json.loads(
                await self._create_work_item(resolved.parent, _CHILD_TYPE, kwargs)
            )
        except Exception as e:
            log.warning("workplan track create failed", track=key, error=str(e))
            # GitLab writes the links after it commits the item, so a link
            # error comes back as a failure for an item that exists.
            saved = await self._saved_child(resolved, key, warnings)
            if saved is None:
                return {"key": key, "result": "failed", "error": str(e)}, str(e)
            warnings.append(
                f"{key}: GitLab saved the child but reported an error: {_short(e)}"
            )
            return {
                "key": key,
                "result": "created",
                "iid": saved.get("iid"),
                **self._published(saved, parent_url, track.title),
            }, None

        created = response.get("work_item") or {}
        create_warnings = response.get("warnings") or []
        warnings.extend(f"{key}: {warning}" for warning in create_warnings)
        entry = {
            "key": key,
            "result": "created",
            "iid": created.get("iid"),
            "id": created["id"],
            "title": created.get("title") or track.title,
            "url": _child_url(parent_url, created.get("iid")),
        }
        return entry, None

    async def _saved_child(
        self, resolved: Any, key: str, warnings: List[str]
    ) -> Optional[Dict[str, Any]]:
        """The child of ``key`` when GitLab saved it, read by its marker.

        This read only improves the report, so a failure of the read is a warning.
        """
        try:
            parent = await self._fetch_parent(
                resolved.parent.full_path, resolved.work_item_iid
            )
        except Exception as e:
            warnings.append(
                f"{key}: could not read the child items again after the failure: "
                f"{_short(e)}"
            )
            return None

        children, _ = self._children(parent)
        return next(
            (
                child
                for child in children
                if _marker_key(child.get("description")) == key
            ),
            None,
        )

    async def _check_links(
        self,
        full_path: str,
        by_key: Dict[str, WorkplanTrack],
        entries: List[Dict[str, Any]],
        published: Dict[str, Dict[str, Any]],
        warnings: List[str],
    ) -> bool:
        """Read back the BLOCKED_BY links of every track that has a child.

        This covers the children that exist from an earlier run too, because
        this tool cannot add a link to an item that exists. GitLab also writes
        the links in a callback after the save, and without the permission to
        set work item metadata it writes nothing and the mutation still
        succeeds. Only a read shows a missing link.

        Returns:
            False when the links could not be read.
        """
        expected = {
            entry["id"]: {
                published[dep]["id"]: dep
                for dep in by_key[entry["key"]].depends_on
                if dep in published
            }
            for entry in entries
            if entry.get("id") and by_key[entry["key"]].depends_on
        }
        if not expected:
            return True

        iids = [str(entry["iid"]) for entry in entries if entry.get("id") in expected]
        # Deliberately resilient: the children exist whatever this read does,
        # and the result must still say which ones.
        try:
            response = await self.gitlab_client.graphql(
                GET_WORKPLAN_TRACKS_LINKS_QUERY, {"fullPath": full_path, "iids": iids}
            )
        except Exception as e:
            warnings.append(f"Could not read back the BLOCKED_BY links: {_short(e)}")
            return False

        found: Dict[str, set[str]] = {}
        nodes = (((response or {}).get("project") or {}).get("workItems") or {}).get(
            "nodes"
        ) or []
        for node in nodes:
            links: set[str] = set()
            for widget in node.get("widgets") or []:
                linked = (widget or {}).get("linkedItems") or {}
                for link in linked.get("nodes") or []:
                    blocker_id = (link.get("workItem") or {}).get("id")
                    if link.get("linkType") == "is_blocked_by" and blocker_id:
                        links.add(blocker_id)
            found[str(node.get("id"))] = links

        for entry in entries:
            want = expected.get(entry.get("id"))
            if want is None:
                continue
            have = found.get(entry["id"], set())
            entry["links_missing"] = sorted(
                key for blocker_id, key in want.items() if blocker_id not in have
            )
        return True

    async def _finish(self, parent_url: str, result: Dict[str, Any]) -> str:
        """Tell the author when the outcome needs a human to check it.

        The planner announces the plan before this step runs, so a skip for the type, a refusal, a partial run or a
        missing link would otherwise show only in the session log.

        The note goes through ``create_work_item_note`` with the options the flow pins for every note it posts: not
        internal, and as a discussion a human can resolve. That tool also raises when GitLab rejects the note.
        """
        if result["status"] != "published":
            # The note only reports, and the result below carries the same facts.
            try:
                # Through ainvoke, so the note gets the input validation, secret
                # redaction and truncation of every tool call.
                await CreateWorkItemNote(metadata=self.metadata).ainvoke(
                    {
                        "body": _outcome_note(result),
                        "url": parent_url,
                        "internal": False,
                        "start_discussion": True,
                    }
                )
                result["note_posted"] = True
            except Exception as e:
                result.setdefault("warnings", []).append(
                    f"Could not post the outcome note: {_short(e)}"
                )
        return json.dumps(result)
