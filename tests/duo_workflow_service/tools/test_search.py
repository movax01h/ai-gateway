import json
from unittest.mock import AsyncMock, patch

import pytest

from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.policies.file_exclusion_policy import FileExclusionPolicy
from duo_workflow_service.tools.search import (
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

        with patch(
            "duo_workflow_service.policies.file_exclusion_policy.is_feature_enabled",
            return_value=True,
        ):
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
    async def test_blob_search_no_exclusions_when_feature_disabled(
        self,
        gitlab_client_mock,
        metadata_with_project,
        blob_search_results,
    ):
        """Test that BlobSearch doesn't filter when feature flag is disabled."""
        gitlab_client_mock.aget.side_effect = create_mock_aget(blob_search_results)

        tool = BlobSearch(metadata=metadata_with_project)

        with patch(
            "duo_workflow_service.policies.file_exclusion_policy.is_feature_enabled",
            return_value=False,
        ):
            response = await tool._arun(
                id="1",
                search="test search",
                search_type="projects",
                ref="main",
            )

        response_data = json.loads(response)
        search_results = response_data["search_results"]

        # Should include all files when feature is disabled
        assert len(search_results) == 4

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

        with patch(
            "duo_workflow_service.policies.file_exclusion_policy.is_feature_enabled",
            return_value=True,
        ):
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

        with patch(
            "duo_workflow_service.policies.file_exclusion_policy.is_feature_enabled",
            return_value=True,
        ):
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

        with patch(
            "duo_workflow_service.policies.file_exclusion_policy.is_feature_enabled",
            return_value=True,
        ):
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
        """Test that BlobSearch logs error details when response fails."""
        from structlog.testing import capture_logs

        from duo_workflow_service.gitlab.http_client import GitLabHttpResponse

        # Mock a failed response
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

        # Verify the response returns empty results
        response_data = json.loads(response)
        assert response_data == {"search_results": []}

        # Verify error was logged
        assert len(captured_logs) == 1
        log_entry = captured_logs[0]
        assert log_entry["event"] == "Blob search request failed"
        assert log_entry["status_code"] == 404
        assert log_entry["error"] == {"message": "404 Project Not Found"}

        # Verify the API was called correctly
        gitlab_client_mock.aget.assert_called_once_with(
            path="/api/v4/projects/999/search",
            params={"scope": "blobs", "search": "test search", "ref": "main"},
            parse_json=True,
        )
