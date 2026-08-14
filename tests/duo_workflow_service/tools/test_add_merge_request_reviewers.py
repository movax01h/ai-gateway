# pylint: disable=file-naming-for-tests
import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.tools.base import ToolException

from duo_workflow_service.gitlab.gitlab_api import Project
from duo_workflow_service.tools.merge_request import (
    SET_REVIEWERS_MUTATION,
    AddMergeRequestReviewers,
    AddMergeRequestReviewersInput,
)


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    return Mock()


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
        "project": Project(
            id=1,
            name="test-project",
            description="Test project",
            http_url_to_repo="http://example.com/repo.git",
            web_url="http://example.com/repo",
            languages=[],
            exclusion_rules=[],
        ),
    }


def _resolve_project_path_mock(path: str) -> AsyncMock:
    # resolve_identifier_to_path fetches path_with_namespace for a numeric id, so
    # mock the /api/v4/projects/{id} response it reads.
    response = Mock()
    response.is_success.return_value = True
    response.body = {"path_with_namespace": path}
    return AsyncMock(return_value=response)


def _set_reviewers_ok() -> AsyncMock:
    return AsyncMock(return_value={"mergeRequestSetReviewers": {"errors": []}})


def _expected_variables(project_path: str, usernames: list[str]) -> dict:
    return {
        "input": {
            "projectPath": project_path,
            "iid": "123",
            "reviewerUsernames": usernames,
            "operationMode": "APPEND",
        }
    }


@pytest.mark.asyncio
async def test_appends_reviewers_for_numeric_project_id(gitlab_client_mock, metadata):
    # A numeric project_id is resolved to its literal namespace path via
    # path_with_namespace (group/subgroup/project) — the canonical path, so slashes
    # are passed through verbatim and it is not %2F-encoded.
    gitlab_client_mock.aget = _resolve_project_path_mock("top-level/subgroup/project")
    gitlab_client_mock.graphql = _set_reviewers_ok()
    tool = AddMergeRequestReviewers(metadata=metadata)

    response = await tool.arun(
        {
            "project_id": 1,
            "merge_request_iid": 123,
            "reviewer_usernames": ["bob"],
        }
    )

    assert response == json.dumps({"requested_reviewers": ["bob"]})
    gitlab_client_mock.aget.assert_called_once_with("/api/v4/projects/1")
    gitlab_client_mock.graphql.assert_called_once_with(
        SET_REVIEWERS_MUTATION,
        _expected_variables("top-level/subgroup/project", ["bob"]),
    )


@pytest.mark.asyncio
async def test_appends_reviewers_from_url(gitlab_client_mock, metadata):
    # When the MR is identified by URL, the project path is parsed from that URL, so
    # no /projects/{id} lookup is needed.
    gitlab_client_mock.aget = AsyncMock()
    gitlab_client_mock.graphql = _set_reviewers_ok()
    tool = AddMergeRequestReviewers(metadata=metadata)

    response = await tool.arun(
        {
            "url": "https://gitlab.com/top-level/subgroup/project/-/merge_requests/123",
            "reviewer_usernames": ["bob", "carol"],
        }
    )

    assert response == json.dumps({"requested_reviewers": ["bob", "carol"]})
    gitlab_client_mock.graphql.assert_called_once_with(
        SET_REVIEWERS_MUTATION,
        _expected_variables("top-level/subgroup/project", ["bob", "carol"]),
    )
    gitlab_client_mock.aget.assert_not_called()


@pytest.mark.asyncio
async def test_empty_reviewer_list_is_a_no_op(gitlab_client_mock, metadata):
    # "Nobody to add" is a legal call, not an error: the assign step always makes
    # exactly one tool call, so a OneOffComponent never retries for want of one.
    gitlab_client_mock.aget = AsyncMock()
    gitlab_client_mock.graphql = AsyncMock()
    tool = AddMergeRequestReviewers(metadata=metadata)

    response = await tool.arun(
        {"project_id": 1, "merge_request_iid": 123, "reviewer_usernames": []}
    )

    assert json.loads(response) == {
        "requested_reviewers": [],
        "message": "No reviewers to add.",
    }
    gitlab_client_mock.aget.assert_not_called()
    gitlab_client_mock.graphql.assert_not_called()


@pytest.mark.asyncio
async def test_graphql_errors_raise_tool_exception(gitlab_client_mock, metadata):
    gitlab_client_mock.aget = _resolve_project_path_mock("namespace/project")
    gitlab_client_mock.graphql = AsyncMock(
        return_value={
            "mergeRequestSetReviewers": {
                "errors": ["User cannot be assigned as reviewer"]
            }
        }
    )
    tool = AddMergeRequestReviewers(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._execute(
            project_id=1, merge_request_iid=123, reviewer_usernames=["bob"]
        )

    assert "Failed to add reviewers" in str(exc_info.value)


@pytest.mark.asyncio
async def test_null_mutation_response_raises_tool_exception(
    gitlab_client_mock, metadata
):
    # A null mergeRequestSetReviewers means the mutation itself was rejected — for
    # example on an instance older than 19.2, where the ai_workflows scope is not
    # allowed for it. Do not report success.
    gitlab_client_mock.aget = _resolve_project_path_mock("namespace/project")
    gitlab_client_mock.graphql = AsyncMock(
        return_value={"mergeRequestSetReviewers": None}
    )
    tool = AddMergeRequestReviewers(metadata=metadata)

    with pytest.raises(ToolException):
        await tool._execute(
            project_id=1, merge_request_iid=123, reviewer_usernames=["bob"]
        )


@pytest.mark.asyncio
async def test_unresolvable_project_fails_before_mutating(gitlab_client_mock, metadata):
    # If the numeric project_id can't be resolved to a namespace path, appending
    # must fail loudly rather than issue the mutation with an unknown project.
    failed_response = Mock()
    failed_response.is_success.return_value = False
    failed_response.status_code = 404
    failed_response.body = "Not found"
    gitlab_client_mock.aget = AsyncMock(return_value=failed_response)
    gitlab_client_mock.graphql = AsyncMock()
    tool = AddMergeRequestReviewers(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._execute(
            project_id=999, merge_request_iid=123, reviewer_usernames=["bob"]
        )

    assert "Failed to resolve project" in str(exc_info.value)
    gitlab_client_mock.graphql.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["19.1.0", "15.3.0"])
async def test_instance_older_than_19_2_fails_without_calling_gitlab(
    gitlab_client_mock, metadata, version
):
    # 19.2 is when the ai_workflows scope was allowed for the mutation. Below it the
    # call cannot succeed, so neither the project lookup nor the mutation is issued.
    gitlab_client_mock.aget = AsyncMock()
    gitlab_client_mock.graphql = AsyncMock()
    tool = AddMergeRequestReviewers(metadata=metadata)

    with patch(
        "duo_workflow_service.tools.version_compatibility.gitlab_version"
    ) as mock_version:
        mock_version.get.return_value = version

        with pytest.raises(ToolException) as exc_info:
            await tool._execute(
                project_id=1, merge_request_iid=123, reviewer_usernames=["bob"]
            )

    assert "requires GitLab 19.2 or later" in str(exc_info.value)
    gitlab_client_mock.aget.assert_not_called()
    gitlab_client_mock.graphql.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["19.2.0", "19.3.1", None])
async def test_supported_or_unknown_version_proceeds(
    gitlab_client_mock, metadata, version
):
    # An unreported version must not block the call: the version_compatibility
    # fallback predates 19.2, so gating on it would disable the tool on instances
    # that do support the mutation.
    gitlab_client_mock.aget = _resolve_project_path_mock("namespace/project")
    gitlab_client_mock.graphql = _set_reviewers_ok()
    tool = AddMergeRequestReviewers(metadata=metadata)

    with patch(
        "duo_workflow_service.tools.version_compatibility.gitlab_version"
    ) as mock_version:
        mock_version.get.return_value = version

        response = await tool._execute(
            project_id=1, merge_request_iid=123, reviewer_usernames=["bob"]
        )

    assert json.loads(response) == {"requested_reviewers": ["bob"]}
    gitlab_client_mock.graphql.assert_called_once_with(
        SET_REVIEWERS_MUTATION, _expected_variables("namespace/project", ["bob"])
    )


@pytest.mark.asyncio
async def test_empty_reviewer_list_is_a_no_op_on_an_old_instance(
    gitlab_client_mock, metadata
):
    # The no-op stays a no-op below 19.2: assign_reviewers must still be able to
    # report "nobody to add" without the step failing.
    gitlab_client_mock.aget = AsyncMock()
    gitlab_client_mock.graphql = AsyncMock()
    tool = AddMergeRequestReviewers(metadata=metadata)

    with patch(
        "duo_workflow_service.tools.version_compatibility.gitlab_version"
    ) as mock_version:
        mock_version.get.return_value = "19.1.0"

        response = await tool._execute(
            project_id=1, merge_request_iid=123, reviewer_usernames=[]
        )

    assert json.loads(response)["requested_reviewers"] == []
    gitlab_client_mock.graphql.assert_not_called()


@pytest.mark.asyncio
async def test_missing_merge_request_identifier_is_rejected(
    gitlab_client_mock, metadata
):
    gitlab_client_mock.graphql = AsyncMock()
    tool = AddMergeRequestReviewers(metadata=metadata)

    with pytest.raises(ToolException) as exc_info:
        await tool._execute(project_id=1, reviewer_usernames=["bob"])

    assert "merge_request_iid" in str(exc_info.value)
    gitlab_client_mock.graphql.assert_not_called()


def test_format_display_message(metadata):
    tool = AddMergeRequestReviewers(metadata=metadata)

    message = tool.format_display_message(
        AddMergeRequestReviewersInput(
            project_id=1, merge_request_iid=123, reviewer_usernames=["bob", "carol"]
        )
    )

    assert message == ("Add reviewers (bob, carol) to merge request !123 in project 1")


def test_format_display_message_with_url(metadata):
    tool = AddMergeRequestReviewers(metadata=metadata)

    message = tool.format_display_message(
        AddMergeRequestReviewersInput(
            url="https://gitlab.com/namespace/project/-/merge_requests/123",
            reviewer_usernames=["bob"],
        )
    )

    assert message == (
        "Add reviewers (bob) to merge request "
        "https://gitlab.com/namespace/project/-/merge_requests/123"
    )


def test_format_display_message_with_no_reviewers(metadata):
    # The empty list is a legal call, so its display message has to read sensibly
    # rather than rendering an empty pair of brackets.
    tool = AddMergeRequestReviewers(metadata=metadata)

    message = tool.format_display_message(
        AddMergeRequestReviewersInput(
            project_id=1, merge_request_iid=123, reviewer_usernames=[]
        )
    )

    assert message == (
        "Add reviewers (no reviewers) to merge request !123 in project 1"
    )
