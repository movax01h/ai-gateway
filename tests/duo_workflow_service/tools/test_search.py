import json
from unittest.mock import AsyncMock, patch

import pytest

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.policies.file_exclusion_policy import FileExclusionPolicy
from duo_workflow_service.tools.search import (
    AdvanceBlobSearch,
    BaseSearchInput,
    BlobSearch,
    CommitSearch,
    GroupProjectSearch,
    IssueSearch,
    IssueSearchInput,
    MilestoneSearch,
    NoteSearch,
    RefSearchInput,
    UserSearch,
    WikiBlobSearch,
)


def create_mock_aget(response_data):
    """Create a mock aget function that returns GitLabHttpResponse."""

    async def mock_aget(*args, **kwargs):
        return GitLabHttpResponse(
            status_code=200,
            body=response_data,
        )

    return mock_aget


class TestSearch:
    @pytest.fixture(name="gitlab_client_mock")
    def gitlab_client_mock_fixture(self):
        return AsyncMock()

    @pytest.fixture(name="metadata")
    def metadata_fixture(self, gitlab_client_mock):
        return {"gitlab_client": gitlab_client_mock}

    @pytest.fixture(name="project_mock")
    def project_mock_fixture(self):
        return {"exclusion_rules": ["*.log", "secrets/*", "*.secret"]}

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
        # Mock return values based on input parameters using side_effect
        mock_aget = create_mock_aget(mock_response)

        gitlab_client_mock.aget.side_effect = mock_aget

        tool = tool_class(metadata=metadata)

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

        # BlobSearch uses different parameters for aget call
        if tool_class == BlobSearch:
            gitlab_client_mock.aget.assert_called_once_with(
                path=f"/api/v4/{search_type}/1/search",
                params=expected_params,
                parse_json=True,
            )
        else:
            gitlab_client_mock.aget.assert_called_once_with(
                path=f"/api/v4/{search_type}/1/search",
                params=expected_params,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "tool_class, scope, search_type, extra_params",
        [
            (IssueSearch, "issues", "projects", {"confidential": False}),
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
        # Mock return values based on input parameters using side_effect
        mock_aget = create_mock_aget([])

        gitlab_client_mock.aget.side_effect = mock_aget

        tool = tool_class(metadata=metadata)

        base_params = {
            "id": "1",
            "search": "not found search term",
            "search_type": "groups",
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

        # BlobSearch uses different parameters for aget call
        if tool_class == BlobSearch:
            gitlab_client_mock.aget.assert_called_once_with(
                path=f"/api/v4/{search_type}/1/search",
                params=expected_params,
                parse_json=True,
            )
        else:
            gitlab_client_mock.aget.assert_called_once_with(
                path=f"/api/v4/{search_type}/1/search",
                params=expected_params,
            )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "search_params, optional_params, mock_response",
        [
            (
                {"id": "123", "search": "test project", "search_type": "groups"},
                {"order_by": None, "sort": None},
                [{"id": 1, "name": "Test Project", "description": "A test project"}],
            ),
            (
                {"id": "456", "search": "workflow", "search_type": "groups"},
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
        gitlab_client_mock.aget.side_effect = create_mock_aget(mock_response)

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
        expected_params.pop("search_type")
        expected_params = {k: v for k, v in expected_params.items() if v is not None}

        gitlab_client_mock.aget.assert_called_once_with(
            path=f"/api/v4/groups/{search_params['id']}/search",
            params=expected_params,
        )

    @pytest.mark.asyncio
    async def test_group_project_search_no_results(self, metadata, gitlab_client_mock):
        gitlab_client_mock.aget.side_effect = create_mock_aget([])

        tool = GroupProjectSearch(
            name="gitlab_group_project_search",
            description="Search for projects within a specified GitLab group.",
            metadata=metadata,
        )

        search_params = {
            "id": "789",
            "search": "nonexistent project",
            "search_type": "groups",
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


class TestBlobSearchFileExclusion:
    """Test FileExclusionPolicy integration with BlobSearch."""

    @pytest.fixture(name="gitlab_client_mock")
    def gitlab_client_mock_fixture(self):
        return AsyncMock()

    @pytest.fixture(name="project_with_exclusions")
    def project_with_exclusions_fixture(self):
        return {"exclusion_rules": ["*.log", "secrets/*", "*.secret"]}

    @pytest.fixture(name="metadata_with_project")
    def metadata_with_project_fixture(
        self, gitlab_client_mock, project_with_exclusions
    ):
        return {
            "gitlab_client": gitlab_client_mock,
            "project": project_with_exclusions,
        }

    @pytest.fixture(name="blob_search_results")
    def blob_search_results_fixture(self):
        return [
            {
                "basename": "main",
                "data": "def main():\n print('hello')",
                "path": "src/main.py",
                "filename": "src/main.py",
                "id": None,
                "ref": "main",
                "startline": 1,
                "project_id": 6,
            },
            {
                "basename": "debug",
                "data": "DEBUG: Application started",
                "path": "logs/debug.log",
                "filename": "logs/debug.log",
                "id": None,
                "ref": "main",
                "startline": 1,
                "project_id": 6,
            },
            {
                "basename": "config",
                "data": "api_key = secret123",
                "path": "secrets/config.secret",
                "filename": "secrets/config.secret",
                "id": None,
                "ref": "main",
                "startline": 1,
                "project_id": 6,
            },
            {
                "basename": "readme",
                "data": "# Project README",
                "path": "README.md",
                "filename": "README.md",
                "id": None,
                "ref": "main",
                "startline": 1,
                "project_id": 6,
            },
        ]

    @pytest.mark.asyncio
    async def test_blob_search_filters_excluded_files(
        self,
        gitlab_client_mock,
        metadata_with_project,
        blob_search_results,
    ):
        """Test that BlobSearch filters out files matching exclusion patterns."""
        gitlab_client_mock.aget.side_effect = create_mock_aget(blob_search_results)

        tool = BlobSearch(metadata=metadata_with_project)

        response = await tool._arun(
            id="1",
            search="test search",
            search_type="projects",
            ref="main",
        )

        response_data = json.loads(response)
        search_results = response_data["search_results"]

        # Should only include main.py and README.md (excluded: debug.log, config.secret)
        assert len(search_results) == 2

        included_paths = [result["path"] for result in search_results]
        assert "src/main.py" in included_paths
        assert "README.md" in included_paths
        assert "logs/debug.log" not in included_paths
        assert "secrets/config.secret" not in included_paths

    @pytest.mark.asyncio
    async def test_blob_search_no_exclusions_when_no_project(
        self,
        gitlab_client_mock,
        blob_search_results,
    ):
        """Test that BlobSearch doesn't filter when no project is available."""
        gitlab_client_mock.aget.side_effect = create_mock_aget(blob_search_results)
        metadata_no_project = {"gitlab_client": gitlab_client_mock, "project": None}

        tool = BlobSearch(metadata=metadata_no_project)

        response = await tool._arun(
            id="1",
            search="test search",
            search_type="projects",
            ref="main",
        )

        response_data = json.loads(response)
        search_results = response_data["search_results"]

        # Should include all files when no project is available
        assert len(search_results) == 4

    @pytest.mark.asyncio
    async def test_blob_search_empty_results(
        self,
        gitlab_client_mock,
        metadata_with_project,
    ):
        """Test that BlobSearch handles empty results correctly."""
        gitlab_client_mock.aget.side_effect = create_mock_aget([])

        tool = BlobSearch(metadata=metadata_with_project)

        response = await tool._arun(
            id="1",
            search="nonexistent",
            search_type="projects",
            ref="main",
        )

        response_data = json.loads(response)
        search_results = response_data["search_results"]

        assert len(search_results) == 0

    @pytest.mark.asyncio
    async def test_blob_search_handles_missing_path_fields(
        self,
        gitlab_client_mock,
        metadata_with_project,
    ):
        """Test that BlobSearch handles results with missing path/filename fields."""
        results_with_missing_paths = [
            {
                "basename": "main",
                "data": "def main():\n print('hello')",
                "path": "src/main.py",
                "filename": "src/main.py",
                "id": None,
                "ref": "main",
                "startline": 1,
                "project_id": 6,
            },
            {
                "basename": "unknown",
                "data": "some content",
                # Missing path and filename fields
                "id": None,
                "ref": "main",
                "startline": 1,
                "project_id": 6,
            },
        ]

        gitlab_client_mock.aget.side_effect = create_mock_aget(
            results_with_missing_paths
        )

        tool = BlobSearch(metadata=metadata_with_project)

        response = await tool._arun(
            id="1",
            search="test search",
            search_type="projects",
            ref="main",
        )

        response_data = json.loads(response)
        search_results = response_data["search_results"]

        # Should include main.py and the result without path (since it can't be filtered)
        assert len(search_results) == 2

    @pytest.mark.asyncio
    async def test_blob_search_logs_error_on_failed_response(
        self, gitlab_client_mock, metadata_with_project
    ):
        """Test that BlobSearch logs error and returns empty results on failure."""
        from structlog.testing import capture_logs

        failed_response = GitLabHttpResponse(
            status_code=404,
            body={"message": "404 Project Not Found"},
            headers={"content-type": "application/json"},
        )
        gitlab_client_mock.aget.return_value = failed_response

        tool = BlobSearch(metadata=metadata_with_project)

        with capture_logs() as captured_logs:
            response = await tool._arun(
                id="999",
                search="test search",
                search_type="projects",
                ref="main",
            )

        response_data = json.loads(response)
        assert response_data == {"search_results": []}

        assert len(captured_logs) == 1
        log_entry = captured_logs[0]
        assert log_entry["event"] == "Blob search request failed"
        assert log_entry["status_code"] == 404
        assert log_entry["error"] == {"message": "404 Project Not Found"}

        gitlab_client_mock.aget.assert_called_once_with(
            path="/api/v4/projects/999/search",
            params={"scope": "blobs", "search": "test search", "ref": "main"},
            parse_json=True,
        )


class TestAdvanceBlobSearch:
    """Test suite for AdvanceBlobSearch tool."""

    @pytest.fixture(name="gitlab_client_mock")
    def gitlab_client_mock_fixture(self):
        return AsyncMock()

    @pytest.fixture(name="project_with_exclusions")
    def project_with_exclusions_fixture(self):
        return {"exclusion_rules": ["*.log", "secrets/*", "*.secret"]}

    @pytest.fixture(name="metadata_with_project")
    def metadata_with_project_fixture(
        self, gitlab_client_mock, project_with_exclusions
    ):
        return {
            "gitlab_client": gitlab_client_mock,
            "project": project_with_exclusions,
        }

    @pytest.fixture(name="blob_search_results")
    def blob_search_results_fixture(self):
        return [
            {
                "basename": "main",
                "data": "def main():\n print('hello')",
                "path": "src/main.py",
                "filename": "src/main.py",
                "id": None,
                "ref": "main",
                "startline": 1,
                "project_id": 6,
            },
            {
                "basename": "debug",
                "data": "DEBUG: Application started",
                "path": "logs/debug.log",
                "filename": "logs/debug.log",
                "id": None,
                "ref": "main",
                "startline": 1,
                "project_id": 6,
            },
            {
                "basename": "config",
                "data": "api_key = secret123",
                "path": "secrets/config.secret",
                "filename": "secrets/config.secret",
                "id": None,
                "ref": "main",
                "startline": 1,
                "project_id": 6,
            },
            {
                "basename": "readme",
                "data": "# Project README",
                "path": "README.md",
                "filename": "README.md",
                "id": None,
                "ref": "main",
                "startline": 1,
                "project_id": 6,
            },
        ]

    @pytest.mark.asyncio
    async def test_project_level_search(
        self, gitlab_client_mock, metadata_with_project, blob_search_results
    ):
        """Test AdvanceBlobSearch with project-level API endpoint."""
        gitlab_client_mock.aget.side_effect = create_mock_aget(blob_search_results)

        tool = AdvanceBlobSearch(metadata=metadata_with_project)

        response = await tool._arun(
            search="def main",
            api_url="/api/v4/projects/123/search",
            ref="main",
        )

        response_data = json.loads(response)
        search_results = response_data["search_results"]

        assert len(search_results) == 2
        included_paths = [result["path"] for result in search_results]
        assert "src/main.py" in included_paths
        assert "README.md" in included_paths

        gitlab_client_mock.aget.assert_called_once_with(
            path="/api/v4/projects/123/search",
            params={"scope": "blobs", "search": "def main", "ref": "main"},
            parse_json=True,
        )

    @pytest.mark.asyncio
    async def test_group_level_search(
        self, gitlab_client_mock, metadata_with_project, blob_search_results
    ):
        """Test AdvanceBlobSearch with group-level API endpoint."""
        gitlab_client_mock.aget.side_effect = create_mock_aget(blob_search_results)

        tool = AdvanceBlobSearch(metadata=metadata_with_project)

        response = await tool._arun(
            search="config",
            api_url="/api/v4/groups/456/search",
        )

        response_data = json.loads(response)
        search_results = response_data["search_results"]

        assert len(search_results) == 2

        gitlab_client_mock.aget.assert_called_once_with(
            path="/api/v4/groups/456/search",
            params={"scope": "blobs", "search": "config"},
            parse_json=True,
        )

    @pytest.mark.asyncio
    async def test_instance_level_search(
        self, gitlab_client_mock, metadata_with_project, blob_search_results
    ):
        """Test AdvanceBlobSearch with instance-wide API endpoint."""
        gitlab_client_mock.aget.side_effect = create_mock_aget(blob_search_results)

        tool = AdvanceBlobSearch(metadata=metadata_with_project)

        response = await tool._arun(
            search="rails",
            api_url="/api/v4/search",
        )

        response_data = json.loads(response)
        assert "search_results" in response_data

        gitlab_client_mock.aget.assert_called_once_with(
            path="/api/v4/search",
            params={"scope": "blobs", "search": "rails"},
            parse_json=True,
        )

    @pytest.mark.asyncio
    async def test_invalid_api_url_raises_exception(
        self, gitlab_client_mock, metadata_with_project
    ):
        """Test that AdvanceBlobSearch raises ToolException for invalid API URLs."""
        from langchain_core.tools.base import ToolException

        tool = AdvanceBlobSearch(metadata=metadata_with_project)

        with pytest.raises(ToolException) as exc_info:
            await tool._arun(
                search="test",
                api_url="/api/v4/invalid/endpoint",
            )

        assert "Invalid api_url" in str(exc_info.value)
        gitlab_client_mock.aget.assert_not_called()

    @pytest.mark.asyncio
    async def test_ref_only_applied_to_project_search(
        self, gitlab_client_mock, metadata_with_project, blob_search_results
    ):
        """Test that ref parameter is only applied for project-level searches."""
        gitlab_client_mock.aget.side_effect = create_mock_aget(blob_search_results)

        tool = AdvanceBlobSearch(metadata=metadata_with_project)

        await tool._arun(
            search="test",
            api_url="/api/v4/groups/123/search",
            ref="main",
        )

        gitlab_client_mock.aget.assert_called_once_with(
            path="/api/v4/groups/123/search",
            params={"scope": "blobs", "search": "test"},
            parse_json=True,
        )

    @pytest.mark.asyncio
    async def test_order_by_and_sort_params(
        self, gitlab_client_mock, metadata_with_project, blob_search_results
    ):
        """Test that order_by and sort parameters are passed correctly."""
        gitlab_client_mock.aget.side_effect = create_mock_aget(blob_search_results)

        tool = AdvanceBlobSearch(metadata=metadata_with_project)

        await tool._arun(
            search="test",
            api_url="/api/v4/projects/123/search",
            order_by="created_at",
            sort="desc",
        )

        gitlab_client_mock.aget.assert_called_once_with(
            path="/api/v4/projects/123/search",
            params={
                "scope": "blobs",
                "search": "test",
                "order_by": "created_at",
                "sort": "desc",
            },
            parse_json=True,
        )

    @pytest.mark.asyncio
    async def test_filters_excluded_files(
        self, gitlab_client_mock, metadata_with_project, blob_search_results
    ):
        """Test that AdvanceBlobSearch filters out files matching exclusion patterns."""
        gitlab_client_mock.aget.side_effect = create_mock_aget(blob_search_results)

        tool = AdvanceBlobSearch(metadata=metadata_with_project)

        response = await tool._arun(
            search="test",
            api_url="/api/v4/projects/123/search",
        )

        response_data = json.loads(response)
        search_results = response_data["search_results"]

        assert len(search_results) == 2
        included_paths = [result["path"] for result in search_results]
        assert "src/main.py" in included_paths
        assert "README.md" in included_paths
        assert "logs/debug.log" not in included_paths
        assert "secrets/config.secret" not in included_paths

    @pytest.mark.asyncio
    async def test_no_exclusions_when_no_project(
        self, gitlab_client_mock, blob_search_results
    ):
        """Test that AdvanceBlobSearch doesn't filter when no project is available."""
        gitlab_client_mock.aget.side_effect = create_mock_aget(blob_search_results)
        metadata_no_project = {"gitlab_client": gitlab_client_mock, "project": None}

        tool = AdvanceBlobSearch(metadata=metadata_no_project)

        response = await tool._arun(
            search="test",
            api_url="/api/v4/projects/123/search",
        )

        response_data = json.loads(response)
        search_results = response_data["search_results"]

        assert len(search_results) == 4

    @pytest.mark.asyncio
    async def test_empty_results(self, gitlab_client_mock, metadata_with_project):
        """Test that AdvanceBlobSearch handles empty results correctly."""
        gitlab_client_mock.aget.side_effect = create_mock_aget([])

        tool = AdvanceBlobSearch(metadata=metadata_with_project)

        response = await tool._arun(
            search="nonexistent",
            api_url="/api/v4/projects/123/search",
        )

        response_data = json.loads(response)
        assert response_data == {"search_results": []}

    @pytest.mark.asyncio
    async def test_raises_exception_on_failed_response(
        self, gitlab_client_mock, metadata_with_project
    ):
        """Test that AdvanceBlobSearch raises ToolException when response fails."""
        from langchain_core.tools.base import ToolException
        from structlog.testing import capture_logs

        failed_response = GitLabHttpResponse(
            status_code=500,
            body={"message": "Internal Server Error"},
            headers={"content-type": "application/json"},
        )
        gitlab_client_mock.aget.return_value = failed_response

        tool = AdvanceBlobSearch(metadata=metadata_with_project)

        with capture_logs() as captured_logs:
            with pytest.raises(ToolException) as exc_info:
                await tool._arun(
                    search="test",
                    api_url="/api/v4/projects/123/search",
                )

        assert "500" in str(exc_info.value)
        assert "Internal Server Error" in str(exc_info.value)

        assert len(captured_logs) == 1
        error_log = [
            log
            for log in captured_logs
            if log["event"] == "Advance Blob search request failed"
        ][0]
        assert error_log["status_code"] == 500

    @pytest.mark.asyncio
    async def test_url_encoded_project_path(
        self, gitlab_client_mock, metadata_with_project, blob_search_results
    ):
        """Test AdvanceBlobSearch with URL-encoded project path."""
        gitlab_client_mock.aget.side_effect = create_mock_aget(blob_search_results)

        tool = AdvanceBlobSearch(metadata=metadata_with_project)

        await tool._arun(
            search="test",
            api_url="/api/v4/projects/group%2Fsubgroup%2Fproject/search",
        )

        gitlab_client_mock.aget.assert_called_once_with(
            path="/api/v4/projects/group%2Fsubgroup%2Fproject/search",
            params={"scope": "blobs", "search": "test"},
            parse_json=True,
        )

    @pytest.mark.asyncio
    async def test_api_url_without_leading_slash(
        self, gitlab_client_mock, metadata_with_project, blob_search_results
    ):
        """Test AdvanceBlobSearch accepts API URL without leading slash."""
        gitlab_client_mock.aget.side_effect = create_mock_aget(blob_search_results)

        tool = AdvanceBlobSearch(metadata=metadata_with_project)

        response = await tool._arun(
            search="test",
            api_url="api/v4/projects/123/search",
        )

        response_data = json.loads(response)
        assert "search_results" in response_data

    @pytest.mark.asyncio
    async def test_handles_missing_path_fields(
        self, gitlab_client_mock, metadata_with_project
    ):
        """Test that AdvanceBlobSearch handles results with missing path/filename."""
        results_with_missing_paths = [
            {
                "basename": "main",
                "data": "def main():",
                "path": "src/main.py",
                "filename": "src/main.py",
                "id": None,
                "ref": "main",
                "startline": 1,
                "project_id": 6,
            },
            {
                "basename": "unknown",
                "data": "some content",
                "id": None,
                "ref": "main",
                "startline": 1,
                "project_id": 6,
            },
        ]
        gitlab_client_mock.aget.side_effect = create_mock_aget(
            results_with_missing_paths
        )

        tool = AdvanceBlobSearch(metadata=metadata_with_project)

        response = await tool._arun(
            search="test",
            api_url="/api/v4/projects/123/search",
        )

        response_data = json.loads(response)
        search_results = response_data["search_results"]

        assert len(search_results) == 2

    def test_supersedes_blob_search(self):
        """Test that AdvanceBlobSearch declares it supersedes BlobSearch."""
        assert AdvanceBlobSearch.supersedes == BlobSearch

    def test_required_capability(self):
        """Test that AdvanceBlobSearch requires advanced_search capability."""
        assert AdvanceBlobSearch.required_capability == "advanced_search"


class TestValidateAndNormalizeApiUrl:
    """Test suite for _validate_and_normalize_api_url."""

    @pytest.fixture
    def tool(self):
        gitlab_client_mock = AsyncMock()
        metadata = {"gitlab_client": gitlab_client_mock, "project": None}
        return AdvanceBlobSearch(metadata=metadata)

    @pytest.mark.parametrize(
        "input_url,expected",
        [
            ("/api/v4/search", "/api/v4/search"),
            ("api/v4/search", "/api/v4/search"),
            ("/api/v4/projects/123/search", "/api/v4/projects/123/search"),
            ("api/v4/projects/123/search", "/api/v4/projects/123/search"),
            ("/api/v4/groups/456/search", "/api/v4/groups/456/search"),
            (
                "/api/v4/projects/gitlab-org%2Fgitlab/search",
                "/api/v4/projects/gitlab-org%2Fgitlab/search",
            ),
            (
                "/api/v4/projects/gitlab-org%2Fmodelops%2Fai-assist/search",
                "/api/v4/projects/gitlab-org%2Fmodelops%2Fai-assist/search",
            ),
            (
                "/api/v4/groups/gitlab-org%2Fmodelops/search",
                "/api/v4/groups/gitlab-org%2Fmodelops/search",
            ),
        ],
    )
    def test_valid_urls(self, tool, input_url, expected):
        assert tool._validate_and_normalize_api_url(input_url) == expected

    @pytest.mark.parametrize(
        "malicious_url",
        [
            "/api/v4/projects/1/search/../../../users",
            "/api/v4/projects/1/search/../../admin",
            "/api/v4/groups/1/search/../issues",
            "/api/v4/projects/gitlab-org%2Fmodelops%2Fai-assist/search/../issues",
            "/api/v4/groups/gitlab-org%2Fmodelops/search/../issues",
            "/api/v4/projects/../search",
            "/api/v4/projects/./search",
            "/api/v4/projects/123/search/extra",
        ],
    )
    def test_path_traversal_rejected(self, tool, malicious_url):
        from langchain_core.tools.base import ToolException

        with pytest.raises(ToolException) as exc_info:
            tool._validate_and_normalize_api_url(malicious_url)
        assert "Invalid api_url" in str(exc_info.value)

    @pytest.mark.parametrize(
        "invalid_url",
        [
            "/api/v4/users",
            "/api/v4/projects/123",
            "/api/v4/admin",
            "/api/v4/projects/123/issues",
            "/api/v3/search",
            "/api/v4/projects/gitlab-org/gitlab/search",
            "/api/v4/groups/gitlab-org/modelops/search",
        ],
    )
    def test_invalid_endpoints_rejected(self, tool, invalid_url):
        from langchain_core.tools.base import ToolException

        with pytest.raises(ToolException) as exc_info:
            tool._validate_and_normalize_api_url(invalid_url)
        assert "Invalid api_url" in str(exc_info.value)
