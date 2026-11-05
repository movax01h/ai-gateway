# pylint: disable=file-naming-for-tests
import json
from unittest.mock import Mock, patch

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.gitlab.executor_http_client import GitLabGraphQLError
from duo_workflow_service.tools.workplan_tracks import (
    WorkplanPublishTracks,
    WorkplanPublishTracksInput,
    WorkplanTrack,
    _child_description,
    _marker_key,
    plan_tracks,
)

PARENT_URL = "https://gitlab.com/group/project/-/work_items/43"
PARENT_ID = "gid://gitlab/WorkItem/900"


def _track(key, files, depends_on=None, title=None):
    return WorkplanTrack(
        key=key,
        title=title or key.title(),
        workplan=f"## Why\n\n{key} reason\n\n## What\n\n1. Build {key}.",
        files=files,
        depends_on=depends_on or [],
    )


# The shape GW 1.0.0 planned for the HTTP servers on 2026-11-03, cut to two
# language tracks: language tracks and root guards in parallel, then two
# shared-file tails.
SERVERS = [
    _track("go", ["go/"]),
    _track("rust", ["rust/"]),
    _track("guards", ["package.json", "eslint.config.js", ".gitignore"]),
    _track("ci", [".gitlab-ci.yml"], ["go", "rust"]),
    _track("readme", ["README.md"], ["go", "rust"]),
]


def _child(gid, iid, title, marker=None, state="OPEN"):
    description = f"<!-- workplan-track: {marker} -->" if marker else "Made by hand."
    return {
        "id": f"gid://gitlab/WorkItem/{gid}",
        "iid": str(iid),
        "title": title,
        "state": state,
        "webUrl": f"https://gitlab.com/group/project/-/work_items/{iid}",
        "description": description,
    }


class TestPlanTracks:
    def test_orders_blockers_first_and_keeps_input_order(self):
        plan = plan_tracks(SERVERS)

        assert not plan.problems
        assert plan.order == ["go", "rust", "guards", "ci", "readme"]

    def test_concurrent_tracks_are_the_unordered_pairs(self):
        concurrent = plan_tracks(SERVERS).concurrent

        assert concurrent["go"] == ["rust", "guards"]
        assert concurrent["ci"] == ["guards", "readme"]
        assert "go" not in concurrent["ci"]

    def test_tail_comes_after_a_blocker_listed_later(self):
        plan = plan_tracks([_track("tail", ["a"], ["head"]), _track("head", ["b"])])

        assert plan.order == ["head", "tail"]

    def test_refuses_two_unordered_tracks_on_one_file(self):
        plan = plan_tracks([_track("a", ["README.md"]), _track("b", ["./README.md"])])

        assert plan.order == []
        assert plan.problems == [
            "Tracks 'a' and 'b' both change README.md, and neither depends on the other."
        ]

    @pytest.mark.parametrize(
        "first,second",
        [
            ("go/", "go/main.go"),
            ("lib/auth", "lib/auth/token.py"),
            ("lib/auth/", "lib/auth"),
        ],
    )
    def test_refuses_a_path_inside_a_directory_another_track_owns(self, first, second):
        plan = plan_tracks([_track("a", [first]), _track("b", [second])])

        assert len(plan.problems) == 1
        assert "'a' and 'b'" in plan.problems[0]

    @pytest.mark.parametrize("whole", [".", "./", "/"])
    def test_the_whole_repository_overlaps_every_path(self, whole):
        plan = plan_tracks([_track("a", [whole]), _track("b", ["README.md"])])

        assert len(plan.problems) == 1

    def test_a_shared_name_prefix_is_not_an_overlap(self):
        plan = plan_tracks([_track("a", ["lib/auth"]), _track("b", ["lib/authz.py"])])

        assert not plan.problems

    def test_a_diamond_orders_the_shared_blocker_once(self):
        # d depends on b and c, which both depend on a: a is an ancestor of d
        # through two paths.
        plan = plan_tracks(
            [
                _track("a", ["a"]),
                _track("b", ["b"], ["a"]),
                _track("c", ["c"], ["a"]),
                _track("d", ["a"], ["b", "c"]),
            ]
        )

        assert not plan.problems
        assert plan.order == ["a", "b", "c", "d"]
        assert plan.concurrent["d"] == []

    def test_allows_one_file_in_tracks_ordered_through_another(self):
        plan = plan_tracks(
            [_track("a", ["x"]), _track("b", ["y"], ["a"]), _track("c", ["x"], ["b"])]
        )

        assert not plan.problems
        assert plan.order == ["a", "b", "c"]

    def test_refuses_a_cycle(self):
        plan = plan_tracks([_track("a", ["x"], ["b"]), _track("b", ["y"], ["a"])])

        assert plan.problems == ["The dependencies of these tracks form a cycle: a, b."]

    @pytest.mark.parametrize(
        "tracks,problem",
        [
            ([_track("a", ["x"], ["zzz"]), _track("b", ["y"])], "unknown track 'zzz'"),
            ([_track("a", ["x"], ["a"]), _track("b", ["y"])], "depends on itself"),
            ([_track("a", ["x"]), _track("a", ["y"])], "keys are not unique: a"),
            ([_track("Bad_Key", ["x"]), _track("b", ["y"])], "'Bad_Key' is not valid"),
            ([_track("a\n", ["x"]), _track("b", ["y"])], "'a\n' is not valid"),
            ([_track("a", []), _track("b", ["y"])], "'a' names no files"),
            ([_track("a", ["  "]), _track("b", ["y"])], "'a' names no files"),
        ],
    )
    def test_refuses_malformed_tracks(self, tracks, problem):
        plan = plan_tracks(tracks)

        assert plan.order == []
        assert any(problem in p for p in plan.problems), plan.problems


class FakeGitLab:
    """Answers the tool's GraphQL calls by query, like the real client.

    The real client (ExecutorGitLabHttpClient.graphql) returns only `data`, and
    raises GitLabGraphQLError on any top-level `errors`.
    """

    def __init__(
        self,
        children=None,
        parent_type="Issue",
        fail_titles=(),
        saved_but_failed_titles=(),
        drop_links_for=(),
        links_query_raises=False,
        notes_raise=False,
        note_rejected=False,
        parent_web_url=PARENT_URL,
        count=None,
        parent_id=PARENT_ID,
        hierarchy_widget=True,
        parent_reads_before_failure=None,
    ):
        self.parent_id = parent_id
        self.hierarchy_widget = hierarchy_widget
        # Raise on every read of the parent after this many.
        self.parent_reads_before_failure = parent_reads_before_failure
        self.parent_reads = 0
        self.note_rejected = note_rejected
        self.parent_web_url = parent_web_url
        self.count = count
        self.children = list(children or [])
        self.parent_type = parent_type
        self.fail_titles = set(fail_titles)
        self.saved_but_failed_titles = set(saved_but_failed_titles)
        self.drop_links_for = set(drop_links_for)
        self.links_query_raises = links_query_raises
        self.notes_raise = notes_raise
        self.creates = []
        self.links = {}
        self.notes = []
        self.link_queries = []
        self.next_id = 1000

    def _save(self, fields):
        self.next_id += 1
        gid = f"gid://gitlab/WorkItem/{self.next_id}"
        iid = str(self.next_id - 900)
        self.creates.append({**fields, "created_id": gid})
        widget = fields.get("linkedItemsWidget")
        if widget and fields["title"] not in self.drop_links_for:
            self.links[gid] = list(widget["workItemsIds"])
        self.children.append(
            {
                "id": gid,
                "iid": iid,
                "title": fields["title"],
                "state": "OPEN",
                "webUrl": f"https://gitlab.com/group/project/-/work_items/{iid}",
                "description": fields["descriptionWidget"]["description"],
            }
        )
        return {"id": gid, "iid": iid, "title": fields["title"]}

    async def graphql(self, query, variables):  # noqa: PLR0911 - one branch per query
        if "getWorkplanTracksParent" in query:
            self.parent_reads += 1
            limit = self.parent_reads_before_failure
            if limit is not None and self.parent_reads > limit:
                raise Exception("GraphQL request timed out after 30 seconds")
            widgets = [{}]
            if self.hierarchy_widget:
                widgets.append(
                    {
                        "children": {
                            "count": self.count or len(self.children),
                            "nodes": list(self.children),
                        }
                    }
                )
            return {
                "project": {
                    "workItems": {
                        "nodes": [
                            {
                                "id": self.parent_id,
                                "iid": "43",
                                "webUrl": self.parent_web_url,
                                "workItemType": {"name": self.parent_type},
                                "widgets": widgets,
                            }
                        ]
                    }
                }
            }
        if "workItemTypes" in query:
            return {
                "namespace": {
                    "workItemTypes": {
                        "nodes": [
                            {"id": "gid://gitlab/WorkItems::Type/1", "name": "Issue"},
                            {"id": "gid://gitlab/WorkItems::Type/5", "name": "Task"},
                        ]
                    }
                }
            }
        if "workItemCreate" in query:
            fields = variables["input"]
            if fields["title"] in self.fail_titles:
                raise GitLabGraphQLError([{"message": "boom"}])
            if fields["title"] in self.saved_but_failed_titles:
                self._save(fields)
                raise GitLabGraphQLError([{"message": "link callback failed"}])
            return {"workItemCreate": {"workItem": self._save(fields), "errors": []}}
        if "getWorkplanTracksLinks" in query:
            self.link_queries.append(variables["iids"])
            if self.links_query_raises:
                raise Exception("GraphQL request timed out after 30 seconds")
            nodes = [
                {
                    "id": child["id"],
                    "widgets": [
                        {
                            "linkedItems": {
                                "nodes": [
                                    {
                                        "linkType": "is_blocked_by",
                                        "workItem": {"id": gid},
                                    }
                                    for gid in self.links.get(child["id"], [])
                                ]
                            }
                        }
                    ],
                }
                for child in self.children
                if child["iid"] in variables["iids"]
            ]
            return {"project": {"workItems": {"nodes": nodes}}}
        if "GetProjectWorkItem" in query:
            # The work item lookup of create_work_item_note.
            return {
                "project": {"workItems": {"nodes": [{"id": PARENT_ID, "iid": "43"}]}}
            }
        if "createDiscussion" in query:
            if self.notes_raise:
                raise Exception("note failed")
            if self.note_rejected:
                return {"createDiscussion": {"note": None, "errors": ["rejected"]}}
            self.notes.append(variables["input"])
            return {
                "createDiscussion": {
                    "note": {"id": "gid://gitlab/Note/1"},
                    "errors": [],
                }
            }
        raise AssertionError(f"unexpected query: {query}")


@pytest.fixture(name="agent_plan_supported", autouse=True)
def agent_plan_supported_fixture():
    with (
        patch(
            "duo_workflow_service.tools.workplan_tracks.supports_agent_plan_widget",
            return_value=True,
        ),
        patch(
            "duo_workflow_service.tools.work_items.base_tool.supports_agent_plan_widget",
            return_value=True,
        ),
        patch(
            "duo_workflow_service.tools.work_items.version_compatibility.supports_agent_plan_widget",
            return_value=True,
        ),
        patch(
            "duo_workflow_service.tools.work_item.supports_create_discussion_mutation",
            return_value=True,
        ),
    ):
        yield


def _tool(fake):
    client = Mock()
    client.graphql = fake.graphql
    return WorkplanPublishTracks(
        metadata={"gitlab_client": client, "gitlab_host": "gitlab.com"}
    )


async def _run(fake, tracks, url=PARENT_URL):
    return json.loads(
        await _tool(fake)._arun(
            url=url, tracks=[t.model_dump() for t in tracks] if tracks else tracks
        )
    )


def _entry(result, key):
    return next(t for t in result["tracks"] if t["key"] == key)


class TestPublish:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("tracks", [None, [], [_track("only", ["x"])]])
    async def test_fewer_than_two_tracks_creates_nothing_and_reads_nothing(
        self, tracks
    ):
        fake = FakeGitLab()
        fake.graphql = None  # any call would fail

        result = json.loads(await _tool(fake)._arun(url=PARENT_URL, tracks=tracks))

        assert result["status"] == "skipped"
        # Quiet: the planner announced no child items. A note post would fail
        # here and leave a warning.
        assert "warnings" not in result
        assert "note_posted" not in result

    @pytest.mark.asyncio
    async def test_creates_tracks_in_order_with_blocked_by_links(self):
        fake = FakeGitLab()

        result = await _run(fake, SERVERS)

        assert result["status"] == "published"
        assert [c["title"] for c in fake.creates] == [
            "Go",
            "Rust",
            "Guards",
            "Ci",
            "Readme",
        ]
        ids = {c["title"]: c["created_id"] for c in fake.creates}
        for create in fake.creates:
            assert create["hierarchyWidget"] == {"parentId": PARENT_ID}
            assert create["workItemTypeId"] == "gid://gitlab/WorkItems::Type/5"

        independent = [
            c for c in fake.creates if c["title"] in ("Go", "Rust", "Guards")
        ]
        assert all("linkedItemsWidget" not in c for c in independent)
        ci = next(c for c in fake.creates if c["title"] == "Ci")
        assert ci["linkedItemsWidget"] == {
            "linkType": "BLOCKED_BY",
            "workItemsIds": [ids["Go"], ids["Rust"]],
        }
        assert _entry(result, "ci")["links_missing"] == []
        assert _entry(result, "readme")["links_missing"] == []
        assert "links_missing" not in _entry(result, "go")
        assert fake.notes == []

    @pytest.mark.asyncio
    async def test_reads_back_only_the_tracks_with_dependencies(self):
        fake = FakeGitLab()

        result = await _run(fake, SERVERS)

        assert fake.link_queries == [
            [_entry(result, "ci")["iid"], _entry(result, "readme")["iid"]]
        ]

    @pytest.mark.asyncio
    async def test_child_description_carries_the_marker(self):
        fake = FakeGitLab()

        await _run(fake, SERVERS)

        description = fake.creates[0]["descriptionWidget"]["description"]
        assert description.endswith("<!-- workplan-track: go -->")
        assert PARENT_URL in description

    @pytest.mark.parametrize("key", ["a", "0", "a-b", "9-z", "a" * 40])
    def test_the_marker_of_every_valid_key_reads_back(self, key):
        # A key that passes the check but whose marker does not read back
        # would get a second child on the next run.
        assert not plan_tracks([_track(key, ["x"]), _track("other", ["y"])]).problems
        assert _marker_key(_child_description(PARENT_URL, key)) == key

    @pytest.mark.asyncio
    async def test_a_key_with_a_final_newline_creates_nothing(self):
        # Its marker would not read back, so a second run would create the
        # track again.
        fake = FakeGitLab()

        result = await _run(fake, [_track("go\n", ["go/"]), _track("rust", ["rust/"])])

        assert result["status"] == "refused"
        assert fake.creates == []

    @pytest.mark.asyncio
    async def test_child_workplan_lists_own_and_concurrent_files(self):
        fake = FakeGitLab()

        await _run(fake, SERVERS)

        by_title = {c["title"]: c["agentPlanWidget"]["content"] for c in fake.creates}
        go = by_title["Go"]
        assert go.startswith("## Why\n\ngo reason")
        assert "Build go.\n\n## Files\n" in go
        assert "This track creates or edits these files:\n\n- `go/`" in go
        # Rust and the guards run at the same time as Go; the tails do not.
        do_not_edit = go.split("Do not edit them:")[1].split("## Order")[0]
        assert "- `rust/`" in do_not_edit
        assert "- `package.json`" in do_not_edit
        assert "`.gitlab-ci.yml`" not in do_not_edit
        assert (
            f"This track is one part of {PARENT_URL}. Implement this track only." in go
        )

        ci = by_title["Ci"]
        assert "Start this track after these work items close:" in ci
        assert "[Go](https://gitlab.com/group/project/-/work_items/" in ci
        do_not_edit = ci.split("Do not edit them:")[1].split("## Order")[0]
        assert "- `README.md`" in do_not_edit
        assert "`go/`" not in do_not_edit

    @pytest.mark.asyncio
    async def test_child_urls_work_from_an_issues_url(self):
        issues_url = "https://gitlab.com/group/project/-/issues/43"
        fake = FakeGitLab(parent_web_url=issues_url)

        result = await _run(fake, SERVERS, url=issues_url)

        assert _entry(result, "go")["url"].startswith(
            "https://gitlab.com/group/project/-/work_items/"
        )

    @pytest.mark.asyncio
    async def test_refused_tracks_create_nothing_and_tell_the_author(self):
        fake = FakeGitLab()

        result = await _run(fake, [_track("a", ["x"]), _track("b", ["x"])])

        assert result["status"] == "refused"
        assert result["problems"]
        assert fake.creates == []
        assert len(fake.notes) == 1
        assert fake.notes[0]["noteableId"] == PARENT_ID
        # The flow's note policy: visible to everyone, as a resolvable thread.
        assert fake.notes[0]["internal"] is False
        assert "with this result: `refused`" in fake.notes[0]["body"]
        assert "both change x" in fake.notes[0]["body"]

    @pytest.mark.asyncio
    async def test_skips_a_parent_that_is_not_an_issue_with_a_note(self):
        # The planner already announced the child items, so the author must
        # learn why there are none.
        fake = FakeGitLab(parent_type="Story")

        result = await _run(fake, SERVERS)

        assert result["status"] == "skipped"
        assert "The type of this work item is `Story`" in result["reason"]
        assert fake.creates == []
        assert result["note_posted"] is True
        assert fake.notes[0]["internal"] is False
        body = fake.notes[0]["body"]
        assert body.startswith(
            "Generate Workplan did not split the plan into child work items."
        )
        assert "Check the child work items" not in body
        assert "splits only a work item of the type named `Issue`" in body

    @pytest.mark.asyncio
    @pytest.mark.parametrize("parent_type", ["", None])
    async def test_a_parent_with_no_type_name_is_skipped_as_unknown(self, parent_type):
        fake = FakeGitLab(parent_type=parent_type)

        result = await _run(fake, SERVERS)

        assert result["status"] == "skipped"
        assert "`unknown`" in result["reason"]
        assert "`unknown`" in fake.notes[0]["body"]

    @pytest.mark.asyncio
    async def test_a_type_name_stays_in_its_code_span(self):
        fake = FakeGitLab(parent_type="Bug` @all `x")

        result = await _run(fake, SERVERS)

        assert "The type of this work item is `Bug @all x`," in result["reason"]

    @pytest.mark.asyncio
    async def test_more_than_ten_tracks_creates_nothing(self):
        fake = FakeGitLab()

        result = await _run(fake, [_track(f"t{i}", [f"f{i}"]) for i in range(11)])

        assert result["status"] == "refused"
        assert fake.creates == []

    @pytest.mark.asyncio
    async def test_a_refusal_names_every_problem_at_once(self):
        # The schema no longer limits the count or the keys, so the note is
        # the only feedback: one run must show every problem.
        fake = FakeGitLab()
        tracks = [_track(f"t{i}", [f"f{i}"]) for i in range(10)]
        tracks.append(_track("Bad_Key", ["f0"]))

        result = await _run(fake, tracks)

        problems = " ".join(result["problems"])
        assert "The maximum is 10." in problems
        assert "Track key 'Bad_Key' is not valid." in problems

    @pytest.mark.asyncio
    async def test_a_long_key_is_cut_in_the_problem(self):
        fake = FakeGitLab()

        result = await _run(fake, [_track("x" * 5000, ["a"]), _track("b", ["b"])])

        assert len(result["problems"][0]) < 200

    @pytest.mark.asyncio
    async def test_a_parent_it_cannot_read_raises_before_any_create(self):
        fake = FakeGitLab()

        async def missing(query, variables):
            if "getWorkplanTracksParent" in query:
                return {"project": {"workItems": {"nodes": []}}}
            raise AssertionError("no other call")

        fake.graphql = missing

        with pytest.raises(ToolException, match="not found"):
            await _run(fake, SERVERS)

    @pytest.mark.asyncio
    async def test_a_group_work_item_raises_before_any_read(self):
        fake = FakeGitLab()
        fake.graphql = None  # any call would fail

        with pytest.raises(ToolException, match="only a work item in a project"):
            await _run(
                fake, SERVERS, url="https://gitlab.com/groups/group/-/work_items/43"
            )

    @pytest.mark.asyncio
    async def test_a_parent_without_a_hierarchy_widget_has_no_children(self):
        fake = FakeGitLab(hierarchy_widget=False)

        result = await _run(fake, SERVERS)

        assert result["status"] == "published"
        assert len(fake.creates) == 5

    @pytest.mark.asyncio
    async def test_without_the_workplan_widget_creates_nothing(self):
        fake = FakeGitLab()
        with patch(
            "duo_workflow_service.tools.workplan_tracks.supports_agent_plan_widget",
            return_value=False,
        ):
            result = await _run(fake, SERVERS)

        assert result["status"] == "skipped"
        assert fake.creates == []
        assert fake.notes == []
        assert "note_posted" not in result

    @pytest.mark.asyncio
    async def test_a_session_without_a_work_item_creates_nothing(self):
        fake = FakeGitLab()
        fake.graphql = None  # any call would fail

        result = json.loads(
            await _tool(fake)._arun(tracks=[t.model_dump() for t in SERVERS])
        )

        assert result["status"] == "skipped"
        assert "warnings" not in result
        assert "note_posted" not in result
        assert WorkplanPublishTracksInput(tracks=None).url is None

    def test_input_accepts_a_missing_track_list(self):
        assert WorkplanPublishTracksInput(url=PARENT_URL).tracks is None


class TestFailures:
    @pytest.mark.asyncio
    async def test_a_graphql_error_stops_creation_and_reports_what_exists(self):
        fake = FakeGitLab(fail_titles={"Guards"})

        result = await _run(fake, SERVERS)

        assert result["status"] == "partial"
        assert [c["title"] for c in fake.creates] == ["Go", "Rust"]
        assert _entry(result, "go")["result"] == "created"
        assert _entry(result, "guards")["result"] == "failed"
        assert "boom" in _entry(result, "guards")["error"]
        # A tail may depend on the failed track, so nothing after it is created.
        assert _entry(result, "ci")["result"] == "not_attempted"
        assert "Run Generate Workplan again" in fake.notes[0]["body"]

    @pytest.mark.asyncio
    async def test_an_item_saved_despite_an_error_is_reported_as_created(self):
        fake = FakeGitLab(saved_but_failed_titles={"Go"})

        result = await _run(fake, SERVERS)

        go = _entry(result, "go")
        assert go["result"] == "created"
        assert go["id"] == fake.creates[0]["created_id"]
        assert any(
            "GitLab saved the child but reported an error" in w
            for w in result["warnings"]
        )
        # The item exists, so the tracks after it are still created.
        assert [entry["result"] for entry in result["tracks"]] == ["created"] * 5
        ci = next(c for c in fake.creates if c["title"] == "Ci")
        assert fake.creates[0]["created_id"] in ci["linkedItemsWidget"]["workItemsIds"]

    @pytest.mark.asyncio
    async def test_a_second_run_completes_a_partial_tree_without_duplicates(self):
        fake = FakeGitLab(fail_titles={"Guards"})
        await _run(fake, SERVERS)
        fake.fail_titles = set()

        result = await _run(fake, SERVERS)

        assert result["status"] == "published"
        assert [c["title"] for c in fake.creates] == [
            "Go",
            "Rust",
            "Guards",
            "Ci",
            "Readme",
        ]

    @pytest.mark.asyncio
    async def test_a_link_read_that_fails_keeps_the_result(self):
        fake = FakeGitLab(links_query_raises=True)

        result = await _run(fake, SERVERS)

        assert result["status"] == "unverified"
        assert len(fake.creates) == 5
        assert any(
            "Could not read back the BLOCKED_BY links" in w for w in result["warnings"]
        )

    @pytest.mark.asyncio
    async def test_reports_links_that_gitlab_did_not_write(self):
        fake = FakeGitLab(drop_links_for={"Ci"})

        result = await _run(fake, SERVERS)

        assert result["status"] == "incomplete"
        assert _entry(result, "ci")["links_missing"] == ["go", "rust"]
        assert _entry(result, "readme")["links_missing"] == []
        assert "GitLab has no BLOCKED_BY link to `go`, `rust`" in fake.notes[0]["body"]

    @pytest.mark.asyncio
    async def test_a_note_gitlab_rejects_is_not_reported_as_posted(self):
        fake = FakeGitLab(drop_links_for={"Ci"}, note_rejected=True)

        result = await _run(fake, SERVERS)

        assert "note_posted" not in result
        assert any("Could not post the outcome note" in w for w in result["warnings"])

    @pytest.mark.asyncio
    async def test_a_long_error_is_cut_in_the_note(self):
        fake = FakeGitLab()
        original = fake.graphql

        async def failing_with_body(query, variables):
            if "workItemCreate" in query:
                raise Exception("GraphQL request failed with HTTP 502: " + "x" * 5000)
            return await original(query, variables)

        fake.graphql = failing_with_body

        await _run(fake, SERVERS)

        assert len(fake.notes[0]["body"]) < 2000

    @pytest.mark.asyncio
    async def test_a_parent_id_the_hierarchy_widget_refuses_creates_nothing(self):
        # create_work_item would drop the parent of such an id with a warning,
        # and create a child that a later run cannot see.
        fake = FakeGitLab(parent_id="gid://gitlab/Issue/900")

        result = await _run(fake, SERVERS)

        assert result["status"] == "refused"
        assert fake.creates == []
        assert "is not a work item global id" in result["problems"][0]

    @pytest.mark.asyncio
    async def test_a_failed_read_after_a_failed_create_is_a_warning(self):
        fake = FakeGitLab(fail_titles={"Go"}, parent_reads_before_failure=1)

        result = await _run(fake, SERVERS)

        assert result["status"] == "partial"
        assert _entry(result, "go")["result"] == "failed"
        assert any(
            "go: could not read the child items again after the failure" in w
            for w in result["warnings"]
        )

    @pytest.mark.asyncio
    async def test_a_note_that_fails_keeps_the_result(self):
        fake = FakeGitLab(drop_links_for={"Ci"}, notes_raise=True)

        result = await _run(fake, SERVERS)

        assert result["status"] == "incomplete"
        assert any("Could not post the outcome note" in w for w in result["warnings"])


class TestExistingChildren:
    @pytest.mark.asyncio
    async def test_a_marked_child_is_not_created_again(self):
        fake = FakeGitLab(
            children=[
                _child(501, 50, "Go server", marker="go"),
                _child(503, 52, "Human task"),
            ]
        )

        result = await _run(fake, SERVERS)

        assert "Go" not in [c["title"] for c in fake.creates]
        assert _entry(result, "go")["result"] == "exists"
        ci = next(c for c in fake.creates if c["title"] == "Ci")
        assert "gid://gitlab/WorkItem/501" in ci["linkedItemsWidget"]["workItemsIds"]
        assert result["status"] == "published"
        assert fake.notes == []

    @pytest.mark.asyncio
    async def test_a_changed_plan_creates_nothing_next_to_earlier_children(self):
        # The planner does not read the markers, so a re-run can pick other keys.
        fake = FakeGitLab(
            children=[
                _child(501, 50, "Go server", marker="go"),
                _child(502, 51, "Old split", marker="old-split"),
            ]
        )

        result = await _run(fake, SERVERS)

        assert result["status"] == "refused"
        assert fake.creates == []
        # Only the child of the dropped track: closing #50 would skip `go`.
        problem = result["problems"][0]
        assert "This plan does not have their tracks: `old-split` (#51)." in problem
        assert "#50" not in problem
        assert "close only these child items" in problem
        assert "counts as done for its track" in problem
        assert "`old-split` (#51)" in fake.notes[0]["body"]

    @pytest.mark.asyncio
    async def test_closing_the_named_children_lets_the_new_plan_split(self):
        # The author follows the refusal note: #51 is closed, #50 stays open.
        fake = FakeGitLab(
            children=[
                _child(501, 50, "Go server", marker="go"),
                _child(502, 51, "Old split", marker="old-split", state="CLOSED"),
            ]
        )

        result = await _run(fake, SERVERS)

        assert result["status"] == "published"
        assert _entry(result, "go")["result"] == "exists"
        assert [c["title"] for c in fake.creates] == ["Rust", "Guards", "Ci", "Readme"]
        assert fake.notes == []

    @pytest.mark.asyncio
    async def test_a_closed_child_of_a_track_in_the_plan_is_not_created_again(self):
        # Finished work stays finished: the closed child counts as the track's.
        fake = FakeGitLab(
            children=[_child(501, 50, "Go server", marker="go", state="CLOSED")]
        )

        result = await _run(fake, SERVERS)

        assert _entry(result, "go")["result"] == "exists"
        assert "Go" not in [c["title"] for c in fake.creates]

    @pytest.mark.asyncio
    async def test_an_open_copy_of_a_closed_child_still_stops_a_changed_plan(self):
        # A copy of a child carries the same marker.
        fake = FakeGitLab(
            children=[
                _child(502, 51, "Old split", marker="old-split", state="CLOSED"),
                _child(503, 52, "Old split copy", marker="old-split"),
                _child(504, 53, "Old split copy 2", marker="old-split"),
            ]
        )

        result = await _run(fake, SERVERS)

        assert result["status"] == "refused"
        # The closed child is already done, so the note names only the open
        # copies, and each of them: the author must close both.
        problem = result["problems"][0]
        assert "their tracks: `old-split` (#52), `old-split` (#53)." in problem
        assert "#51" not in problem

    @pytest.mark.asyncio
    async def test_a_closed_child_of_an_earlier_plan_does_not_stop_a_new_split(self):
        fake = FakeGitLab(
            children=[_child(502, 51, "Old split", marker="old-split", state="CLOSED")]
        )

        result = await _run(fake, SERVERS)

        assert result["status"] == "published"
        assert len(fake.creates) == 5

    @pytest.mark.asyncio
    async def test_a_child_without_a_marker_is_not_a_track(self):
        fake = FakeGitLab(children=[_child(503, 52, "Go")])

        result = await _run(fake, SERVERS)

        assert _entry(result, "go")["result"] == "created"
        assert "Go" in [c["title"] for c in fake.creates]

    @pytest.mark.asyncio
    async def test_a_first_run_posts_no_note(self):
        fake = FakeGitLab()

        result = await _run(fake, SERVERS)

        assert result["status"] == "published"
        assert fake.notes == []

    @pytest.mark.asyncio
    async def test_more_children_than_one_page_creates_nothing(self):
        # A marked child on a later page would get a second copy.
        fake = FakeGitLab(count=150)

        result = await _run(fake, SERVERS)

        assert result["status"] == "refused"
        assert fake.creates == []
        assert "has 150 child items" in result["problems"][0]


class TestDisplayMessage:
    def test_reports_the_status_when_there_is_a_response(self):
        tool = WorkplanPublishTracks(metadata={})
        args = WorkplanPublishTracksInput(url=PARENT_URL, tracks=SERVERS)

        assert tool.format_display_message(args, json.dumps({"status": "refused"})) == (
            "Publish the tracks of the Workplan as child work items: refused"
        )
        assert tool.format_display_message(args) == (
            "Publish 5 tracks of the Workplan as child work items"
        )

    def test_a_response_that_is_not_json_reads_as_no_status(self):
        tool = WorkplanPublishTracks(metadata={})
        args = WorkplanPublishTracksInput(url=PARENT_URL, tracks=SERVERS)

        assert tool.format_display_message(args, "not json") == (
            "Publish 5 tracks of the Workplan as child work items"
        )
