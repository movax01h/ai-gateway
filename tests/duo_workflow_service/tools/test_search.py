import json
from unittest.mock import AsyncMock

import pytest

from duo_workflow_service.tools.search import (
    BaseSearchInput,
    BlobSearch,
    CommitSearch,
    GroupProjectSearch,
    IssueSearch,
    IssueSearchInput,
    MergeRequestSearch,
    MergeRequestSearchInput,
    MilestoneSearch,
    NoteSearch,
    RefSearchInput,
    UserSearch,
    WikiBlobSearch,
)


class TestSearch:
    @pytest.fixture
    def gitlab_client_mock(self):
        return AsyncMock()

    @pytest.fixture
    def metadata(self, gitlab_client_mock):
        return {"gitlab_client": gitlab_client_mock}

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_class, scope, search_type, search_params, optional_params, mock_response",
        [
            (
                IssueSearch,
                "issues",
                "projects",
                {"confidential": False, "state": "open"},
                {"order_by": None, "sort": None},
                [{"id": 1, "title": "found issue"}],
            ),
            (
                MergeRequestSearch,
                "merge_requests",
                "groups",
                {"state": "opened"},
                {"order_by": "created_at", "sort": None},
                [{"id": 1, "title": "found merge request"}],
            ),
            (
                MilestoneSearch,
                "milestones",
                "projects",
                {},
                {"order_by": None, "sort": "desc"},
                [{"id": 1, "title": "found milestone"}],
            ),
            (
                UserSearch,
                "users",
                "groups",
                {},
                {"order_by": "created_at", "sort": "asc"},
                [{"id": 1, "title": "found user"}],
            ),
            (
                WikiBlobSearch,
                "wiki_blobs",
                "projects",
                {"ref": "main"},
                {"order_by": None, "sort": None},
                [{"id": 1, "title": "found wiki blob"}],
            ),
            (
                CommitSearch,
                "commits",
                "projects",
                {"ref": "main"},
                {"order_by": "created_at", "sort": "desc"},
                [{"id": "abc123", "title": "found commit"}],
            ),
            (
                BlobSearch,
                "blobs",
                "projects",
                {"ref": "main"},
                {"order_by": None, "sort": "asc"},
                [{"id": 1, "title": "found blob"}],
            ),
            (
                NoteSearch,
                "notes",
                "projects",
                {},
                {"order_by": "created_at", "sort": None},
                [{"id": 1, "title": "found note"}],
            ),
        ],
    )
    async def test_search(
        self,
        tool_class,
        scope,
        search_type,
        search_params,
        optional_params,
        mock_response,
        metadata,
        gitlab_client_mock,
    ):
        gitlab_client_mock.aget.return_value = mock_response

        tool = tool_class(metadata=metadata)  # type: ignore

        base_params = {
            "id": "1",
            "search": "test search",
            "search_type": search_type,
        }

        all_params = {**base_params, **search_params, **optional_params}
        response = await tool._arun(**all_params)

        expected_response = json.dumps({"search_results": mock_response})
        assert response == expected_response

        expected_params = {"scope": scope, **all_params}
        expected_params = {k: v for k, v in expected_params.items() if v is not None}
        expected_params.pop("id")
        expected_params.pop("search_type")
        if "confidential" in expected_params:
            expected_params["confidential"] = str(
                expected_params["confidential"]
            ).lower()

        gitlab_client_mock.aget.assert_called_once_with(
            path=f"/api/v4/{search_type}/1/search",
            params=expected_params,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_class, scope, search_type, extra_params",
        [
            (IssueSearch, "issues", "projects", {"confidential": False}),
            (MergeRequestSearch, "merge_requests", "groups", {}),
            (MilestoneSearch, "milestones", "projects", {}),
            (UserSearch, "users", "groups", {}),
            (WikiBlobSearch, "wiki_blobs", "projects", {}),
            (CommitSearch, "commits", "projects", {}),
            (BlobSearch, "blobs", "projects", {}),
            (NoteSearch, "notes", "projects", {}),
            (GroupProjectSearch, "projects", "groups", {}),
        ],
    )
    async def test_search_no_results(
        self, tool_class, scope, search_type, extra_params, metadata, gitlab_client_mock
    ):
        gitlab_client_mock.aget.return_value = []

        tool = tool_class(metadata=metadata)  # type: ignore

        base_params = {
            "id": "1",
            "search": "not found search term",
        }
        search_type_param = {"search_type": search_type}
        all_params = {**base_params, **extra_params}
        if tool_class != GroupProjectSearch:
            all_params = {**all_params, **search_type_param}

        response = await tool._arun(**all_params)

        expected_response = json.dumps({"search_results": []})
        assert response == expected_response

        expected_params = {"scope": scope, **all_params}
        expected_params.pop("id")
        if "search_type" in expected_params:
            expected_params.pop("search_type")
        if "confidential" in expected_params:
            expected_params["confidential"] = str(
                expected_params["confidential"]
            ).lower()

        gitlab_client_mock.aget.assert_called_once_with(
            path=f"/api/v4/{search_type}/1/search",
            params=expected_params,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "search_params, optional_params, mock_response",
        [
            (
                {"id": "123", "search": "test project"},
                {"order_by": None, "sort": None},
                [{"id": 1, "name": "Test Project", "description": "A test project"}],
            ),
            (
                {"id": "456", "search": "workflow"},
                {"order_by": "created_at", "sort": "desc"},
                [
                    {
                        "id": 2,
                        "name": "Workflow Project",
                        "description": "A workflow project",
                    },
                    {
                        "id": 3,
                        "name": "Another Workflow",
                        "description": "Another workflow project",
                    },
                ],
            ),
        ],
    )
    async def test_group_project_search(
        self,
        search_params,
        optional_params,
        mock_response,
        metadata,
        gitlab_client_mock,
    ):
        gitlab_client_mock.aget.return_value = mock_response

        tool = GroupProjectSearch(
            name="gitlab_group_project_search",
            description="Search for projects within a specified GitLab group.",
            metadata=metadata,
        )

        all_params = {**search_params, **optional_params}
        response = await tool._arun(**all_params)

        expected_response = json.dumps({"search_results": mock_response})
        assert response == expected_response

        expected_params = {"scope": "projects", **all_params}
        expected_params.pop("id")
        expected_params = {k: v for k, v in expected_params.items() if v is not None}

        gitlab_client_mock.aget.assert_called_once_with(
            path=f"/api/v4/groups/{search_params['id']}/search",
            params=expected_params,
        )

    @pytest.mark.asyncio
    async def test_group_project_search_no_results(self, metadata, gitlab_client_mock):
        gitlab_client_mock.aget.return_value = []

        tool = GroupProjectSearch(
            name="gitlab_group_project_search",
            description="Search for projects within a specified GitLab group.",
            metadata=metadata,
        )

        search_params = {
            "id": "789",
            "search": "nonexistent project",
        }

        response = await tool._arun(**search_params)

        expected_response = json.dumps({"search_results": []})
        assert response == expected_response

        expected_params = {"scope": "projects", "search": "nonexistent project"}

        gitlab_client_mock.aget.assert_called_once_with(
            path="/api/v4/groups/789/search",
            params=expected_params,
        )


def test_blob_search_format_display_message():
    tool = BlobSearch(description="Blob search description")

    input_data = RefSearchInput(
        id="123", search="test search", search_type="projects", ref="main"
    )

    message = tool.format_display_message(input_data)

    expected_message = "Search for files with term 'test search' in projects 123"
    assert message == expected_message


def test_commit_search_format_display_message():
    tool = CommitSearch(description="Commit search description")

    input_data = RefSearchInput(
        id="123", search="test search", search_type="projects", ref="main"
    )

    message = tool.format_display_message(input_data)

    expected_message = "Search for commits with term 'test search' in projects 123"
    assert message == expected_message


def test_group_project_search_format_display_message():
    tool = GroupProjectSearch(description="Group project search description")

    input_data = BaseSearchInput(id="123", search="test search", search_type="groups")

    message = tool.format_display_message(input_data)

    expected_message = "Search for projects with term 'test search' in groups 123"
    assert message == expected_message


def test_issue_search_format_display_message():
    tool = IssueSearch(description="Issue search description")

    input_data = IssueSearchInput(
        id="123",
        search="test search",
        search_type="projects",
        state="opened",
        confidential=False,
    )

    message = tool.format_display_message(input_data)

    expected_message = "Search for issues with term 'test search' in projects 123"
    assert message == expected_message


def test_merge_request_search_format_display_message():
    tool = MergeRequestSearch(description="Merge request search description")

    input_data = MergeRequestSearchInput(
        id="123", search="test search", search_type="groups", state="opened"
    )

    message = tool.format_display_message(input_data)

    expected_message = "Search for merge requests with term 'test search' in groups 123"
    assert message == expected_message


def test_milestone_search_format_display_message():
    tool = MilestoneSearch(description="Milestone search description")

    input_data = BaseSearchInput(id="123", search="test search", search_type="projects")

    message = tool.format_display_message(input_data)

    expected_message = "Search for milestones with term 'test search' in projects 123"
    assert message == expected_message


def test_note_search_format_display_message():
    tool = NoteSearch(description="Note search description")

    input_data = BaseSearchInput(id="123", search="test search", search_type="projects")

    message = tool.format_display_message(input_data)

    expected_message = "Search for comments with term 'test search' in projects 123"
    assert message == expected_message


def test_user_search_format_display_message():
    tool = UserSearch(description="User search description")

    input_data = BaseSearchInput(id="123", search="test search", search_type="groups")

    message = tool.format_display_message(input_data)

    expected_message = "Search for users with term 'test search' in groups 123"
    assert message == expected_message


def test_wiki_blob_search_format_display_message():
    tool = WikiBlobSearch(description="Wiki blob search description")

    input_data = RefSearchInput(
        id="123", search="test search", search_type="projects", ref="main"
    )

    message = tool.format_display_message(input_data)

    expected_message = "Search for files with term 'test search' in projects 123"
    assert message == expected_message
