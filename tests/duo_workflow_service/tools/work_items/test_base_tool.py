"""Tests for base tool methods shared across work item tools."""

import json
from unittest.mock import AsyncMock, Mock, patch

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.tools.work_item import (
    CreateWorkItem,
    GetWorkItem,
    UpdateWorkItem,
)
from duo_workflow_service.tools.work_items.base_tool import (
    ResolvedParent,
    ResolvedWorkItem,
)
from duo_workflow_service.tools.work_items.queries.work_items import (
    GET_GROUP_LABELS_QUERY,
    GET_PROJECT_LABELS_QUERY,
)


@pytest.fixture(name="gitlab_client_mock")
def gitlab_client_mock_fixture():
    mock = Mock()
    mock.graphql = AsyncMock()
    return mock


@pytest.fixture(name="metadata")
def metadata_fixture(gitlab_client_mock):
    return {
        "gitlab_client": gitlab_client_mock,
        "gitlab_host": "gitlab.com",
    }


@pytest.mark.asyncio
async def test_validate_parent_url_with_group_id(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    result = await tool._validate_parent_url(
        url=None,
        group_id="namespace/group",
        project_id=None,
    )
    assert isinstance(result, ResolvedParent)
    assert result.type == "group"
    assert result.full_path == "namespace/group"


@pytest.mark.asyncio
async def test_validate_parent_url_with_project_id(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    result = await tool._validate_parent_url(
        url=None,
        group_id=None,
        project_id="namespace/project",
    )
    assert isinstance(result, ResolvedParent)
    assert result.type == "project"
    assert result.full_path == "namespace/project"


@pytest.mark.asyncio
async def test_validate_parent_url_with_group_url(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    result = await tool._validate_parent_url(
        url="https://gitlab.com/groups/namespace/group",
        group_id=None,
        project_id=None,
    )
    assert isinstance(result, ResolvedParent)
    assert result.type == "group"
    assert result.full_path == "namespace/group"


@pytest.mark.asyncio
async def test_validate_parent_url_with_project_url(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    result = await tool._validate_parent_url(
        url="https://gitlab.com/namespace/project",
        group_id=None,
        project_id=None,
    )
    assert isinstance(result, ResolvedParent)
    assert result.type == "project"
    assert result.full_path == "namespace/project"


@pytest.mark.asyncio
async def test_validate_parent_url_with_invalid_url(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    with pytest.raises(ToolException) as exc_info:
        await tool._validate_parent_url(
            url="https://example.com/not-gitlab",
            group_id=None,
            project_id=None,
        )
    assert "Failed to parse parent work item URL" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validate_parent_url_with_no_params(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    with pytest.raises(ToolException) as exc_info:
        await tool._validate_parent_url(url=None, group_id=None, project_id=None)
    assert "Must provide either URL, group_id, or project_id" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validate_work_item_url_with_group_id_and_iid(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    resolved_parent = ResolvedParent(type="group", full_path="namespace/group")
    tool._validate_parent_url = AsyncMock(return_value=resolved_parent)

    result = await tool._validate_work_item_url(
        url=None,
        group_id="namespace/group",
        project_id=None,
        work_item_iid=42,
    )
    assert result.parent.type == "group"
    assert result.parent.full_path == "namespace/group"
    assert result.work_item_iid == 42


@pytest.mark.asyncio
async def test_validate_work_item_url_with_project_id_and_iid(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    resolved_parent = ResolvedParent(type="project", full_path="namespace/project")
    tool._validate_parent_url = AsyncMock(return_value=resolved_parent)

    result = await tool._validate_work_item_url(
        url=None,
        group_id=None,
        project_id="namespace/project",
        work_item_iid=42,
    )
    assert result.parent.type == "project"
    assert result.parent.full_path == "namespace/project"
    assert result.work_item_iid == 42


@pytest.mark.asyncio
async def test_validate_work_item_url_with_group_url(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    result = await tool._validate_work_item_url(
        url="https://gitlab.com/groups/namespace/group/-/work_items/42",
        group_id=None,
        project_id=None,
        work_item_iid=None,
    )
    assert result.parent.type == "group"
    assert result.parent.full_path == "namespace/group"
    assert result.work_item_iid == 42


@pytest.mark.asyncio
async def test_validate_work_item_url_with_project_url(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    result = await tool._validate_work_item_url(
        url="https://gitlab.com/namespace/project/-/work_items/42",
        group_id=None,
        project_id=None,
        work_item_iid=None,
    )
    assert result.parent.type == "project"
    assert result.parent.full_path == "namespace/project"
    assert result.work_item_iid == 42


@pytest.mark.asyncio
async def test_validate_work_item_url_with_no_iid(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    with pytest.raises(ToolException) as exc_info:
        await tool._validate_work_item_url(
            url=None,
            group_id="namespace/group",
            project_id=None,
            work_item_iid=None,
        )
    assert "Must provide work_item_iid if no URL is given" in str(exc_info.value)


@pytest.mark.asyncio
async def test_validate_work_item_url_with_invalid_url_without_work_item_iid(metadata):
    tool = GetWorkItem(description="test tool", metadata=metadata)
    with pytest.raises(ToolException) as exc_info:
        await tool._validate_work_item_url(
            url="https://example.com/namespace/project/-/work_items/42",
            group_id=None,
            project_id=None,
            work_item_iid=None,
        )
    assert "Failed to parse work item URL" in str(exc_info.value)


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.tools.work_items.base_tool.get_query_variables_for_version"
)
async def test_fetch_work_item_data_calls_version_compatibility(
    mock_get_query_variables,
    gitlab_client_mock,
    metadata,
):
    mock_get_query_variables.return_value = {
        "includeHierarchyWidget": True,
        "includeDevelopmentWidget": True,
    }
    work_item_data = {
        "id": "gid://gitlab/WorkItem/123",
        "iid": "42",
        "title": "Test Work Item",
    }
    graphql_response = {"project": {"workItems": {"nodes": [work_item_data]}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = UpdateWorkItem(description="update work item", metadata=metadata)

    resolved = ResolvedWorkItem(
        parent=ResolvedParent(type="project", full_path="namespace/project"),
        work_item_iid=42,
    )

    await tool._fetch_work_item_data(resolved)

    mock_get_query_variables.assert_called_once_with(
        "includeHierarchyWidget", "includeDevelopmentWidget"
    )
    gitlab_client_mock.graphql.assert_called_once()
    call_args = gitlab_client_mock.graphql.call_args
    query_variables = call_args[0][1]
    assert query_variables["includeHierarchyWidget"] is True
    assert query_variables["includeDevelopmentWidget"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "graphql_return, parent_type, full_path, expected_message",
    [
        (
            None,
            "project",
            "namespace/project",
            "Project 'namespace/project' not found or not accessible",
        ),
        (
            {},
            "project",
            "namespace/project",
            "Project 'namespace/project' not found or not accessible",
        ),
        (
            {"namespace": None},
            "group",
            "namespace/group",
            "Group 'namespace/group' not found or not accessible",
        ),
    ],
)
async def test_get_work_item_data_raises_on_bad_response(
    gitlab_client_mock,
    metadata,
    graphql_return,
    parent_type,
    full_path,
    expected_message,
):
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_return)

    tool = GetWorkItem(description="get work item", metadata=metadata)

    resolved = ResolvedWorkItem(
        parent=ResolvedParent(type=parent_type, full_path=full_path),
        work_item_iid=42,
    )

    with pytest.raises(ToolException) as exc_info:
        await tool._get_work_item_data(resolved)

    message = str(exc_info.value)
    assert expected_message in message
    assert "None" not in message


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.tools.work_items.base_tool.get_query_variables_for_version"
)
async def test_get_work_item_data_calls_version_compatibility(
    mock_get_query_variables,
    gitlab_client_mock,
    metadata,
):
    mock_get_query_variables.return_value = {
        "includeHierarchyWidget": True,
        "includeDevelopmentWidget": True,
    }
    work_item_data = {
        "id": "gid://gitlab/WorkItem/123",
        "iid": "42",
        "title": "Test Work Item",
    }
    graphql_response = {"project": {"workItems": {"nodes": [work_item_data]}}}
    gitlab_client_mock.graphql = AsyncMock(return_value=graphql_response)

    tool = GetWorkItem(description="get work item", metadata=metadata)

    resolved = ResolvedWorkItem(
        parent=ResolvedParent(type="project", full_path="namespace/project"),
        work_item_iid=42,
    )

    await tool._get_work_item_data(resolved)

    mock_get_query_variables.assert_called_once_with(
        "includeHierarchyWidget", "includeDevelopmentWidget"
    )
    gitlab_client_mock.graphql.assert_called_once()
    call_args = gitlab_client_mock.graphql.call_args
    query_variables = call_args[0][1]
    assert query_variables["includeHierarchyWidget"] is True
    assert query_variables["includeDevelopmentWidget"] is True


def exact_labels(root_key, *labels):
    """Response of the first lookup, which filters on the exact title."""
    return {root_key: {"exact": {"nodes": list(labels)}}}


def similar_labels(root_key, *labels):
    """Response of the follow-up lookup, which searches the titles."""
    return {root_key: {"similar": {"nodes": list(labels)}}}


def unknown_label(root_key, *similar):
    """A name that misses the title filter and falls through to the search."""
    return [exact_labels(root_key), similar_labels(root_key, *similar)]


TESTING_LABEL = {"id": "gid://gitlab/ProjectLabel/789", "title": "testing"}
FLAKY_LABEL = {"id": "gid://gitlab/GroupLabel/1", "title": "testing::flaky"}
ANCESTOR_GROUP_LABEL = {"id": "gid://gitlab/GroupLabel/2", "title": "testing"}
OWN_GROUP_LABEL = {"id": "gid://gitlab/GroupLabel/9", "title": "testing"}
BUG_LABEL = {"id": "gid://gitlab/Label/3", "title": "bug"}
BUG_LABELS = [
    {"id": "gid://gitlab/Label/1", "title": "Bug"},
    {"id": "gid://gitlab/Label/2", "title": "bug"},
]


@pytest.fixture(name="update_tool")
def update_tool_fixture(metadata):
    tool = UpdateWorkItem(description="update work item", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(
        return_value=ResolvedWorkItem(
            id="gid://gitlab/WorkItem/123",
            full_data={"workItemType": {"name": "Issue"}},
            parent=ResolvedParent(type="project", full_path="namespace/project"),
        )
    )
    return tool


@pytest.mark.asyncio
async def test_update_resolves_label_names(gitlab_client_mock, update_tool):
    gitlab_client_mock.graphql.side_effect = [
        exact_labels("project", TESTING_LABEL),
        exact_labels("project", {"id": "gid://gitlab/GroupLabel/5", "title": "stale"}),
        {"workItemUpdate": {"workItem": {"id": "gid://gitlab/WorkItem/123"}}},
    ]

    response = await update_tool._arun(
        project_id="namespace/project",
        work_item_iid=42,
        add_labels=["testing"],
        remove_labels=["stale"],
    )

    assert "warnings" not in json.loads(response)

    lookup_query, lookup_vars = gitlab_client_mock.graphql.call_args_list[0][0]
    assert lookup_query == GET_PROJECT_LABELS_QUERY
    assert lookup_vars == {
        "fullPath": "namespace/project",
        "name": "testing",
        "withSimilar": False,
        "includeAncestors": True,
    }

    gql_input = gitlab_client_mock.graphql.call_args_list[2][0][1]["input"]
    assert gql_input["labelsWidget"] == {
        "addLabelIds": ["gid://gitlab/ProjectLabel/789"],
        "removeLabelIds": ["gid://gitlab/GroupLabel/5"],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "found", "expected_widget"),
    [
        pytest.param(
            {"add_labels": ["testing", "testing"], "remove_labels": ["testing"]},
            [TESTING_LABEL],
            {
                "addLabelIds": [TESTING_LABEL["id"]],
                "removeLabelIds": [TESTING_LABEL["id"]],
            },
            id="a repeated name is looked up once",
        ),
        pytest.param(
            {"add_labels": ["testing"]},
            [ANCESTOR_GROUP_LABEL, TESTING_LABEL],
            {"addLabelIds": [TESTING_LABEL["id"]]},
            id="the project label beats its ancestor twin",
        ),
        pytest.param(
            {"add_labels": ["testing"], "add_label_ids": [101]},
            [TESTING_LABEL],
            {"addLabelIds": ["gid://gitlab/Label/101", TESTING_LABEL["id"]]},
            id="resolved names extend explicit IDs",
        ),
    ],
)
async def test_update_resolves_names_with_one_lookup(
    gitlab_client_mock, update_tool, kwargs, found, expected_widget
):
    gitlab_client_mock.graphql.side_effect = [
        exact_labels("project", *found),
        {"workItemUpdate": {"workItem": {"id": "gid://gitlab/WorkItem/123"}}},
    ]

    await update_tool._arun(project_id="namespace/project", work_item_iid=42, **kwargs)

    assert gitlab_client_mock.graphql.call_count == 2
    assert gitlab_client_mock.graphql.call_args_list[0][0][1]["name"] == "testing"

    gql_input = gitlab_client_mock.graphql.call_args_list[1][0][1]["input"]
    assert gql_input["labelsWidget"] == expected_widget


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lookups",
    [
        # The instance does not let an ai_workflows token read labels.
        pytest.param([{"project": {"exact": None}}] * 2, id="labels resolve to null"),
        # The parent itself is gone, which is not a reason to drop the update.
        pytest.param([{"project": None}] * 2, id="parent is inaccessible"),
        pytest.param([Exception("HTTP 500")] * 2, id="the lookup raises"),
        pytest.param(
            [exact_labels("project"), {"project": {"similar": None}}] * 2,
            id="only the search is unreadable",
        ),
    ],
)
async def test_update_reports_unreadable_labels_once(
    gitlab_client_mock, update_tool, lookups
):
    gitlab_client_mock.graphql.side_effect = [
        *lookups,
        {"workItemUpdate": {"workItem": {"id": "gid://gitlab/WorkItem/123"}}},
    ]

    response = json.loads(
        await update_tool._arun(
            project_id="namespace/project",
            work_item_iid=42,
            add_labels=["testing", "flaky"],
            title="New title",
        )
    )

    gql_input = gitlab_client_mock.graphql.call_args[0][1]["input"]
    assert gql_input["title"] == "New title"
    assert "labelsWidget" not in gql_input

    # One systemic cause, so the two names must not repeat it.
    assert len(response["warnings"]) == 1
    warning = response["warnings"][0]
    assert "the labels of project 'namespace/project' could not be read" in warning
    assert "Pass label IDs instead" in warning


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "similar", "expected_hint"),
    [
        ("testing::flak", [FLAKY_LABEL], "Similar labels: 'testing::flaky'."),
        ("BUG", BUG_LABELS, "Similar labels: 'Bug', 'bug'."),
        # Only an exact title resolves, so wrong casing is a hint too.
        ("TESTING", [TESTING_LABEL], "Similar labels: 'testing'."),
    ],
)
async def test_update_reports_unresolvable_label(
    gitlab_client_mock, update_tool, name, similar, expected_hint
):
    gitlab_client_mock.graphql.side_effect = [
        *unknown_label("project", *similar),
        {"workItemUpdate": {"workItem": {"id": "gid://gitlab/WorkItem/123"}}},
    ]

    response = json.loads(
        await update_tool._arun(
            project_id="namespace/project", work_item_iid=42, add_labels=[name]
        )
    )

    warning = response["warnings"][0]
    assert (
        f"Label '{name}' was not applied: it does not exist in "
        "project 'namespace/project'." in warning
    )
    assert expected_hint in warning


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kwargs", "lookups", "expected_widget", "reported"),
    [
        pytest.param(
            {"add_labels": ["bug", "project"]},
            [exact_labels("project", BUG_LABEL), *unknown_label("project")],
            {"addLabelIds": [BUG_LABEL["id"]]},
            ["Label 'project' was not applied"],
            id="a known name still applies",
        ),
        pytest.param(
            {"add_labels": ["nope", "also-nope"], "remove_labels": ["gone"]},
            unknown_label("project") * 3,
            None,
            [
                "Label 'nope' was not applied",
                "Label 'also-nope' was not applied",
                "Label 'gone' was not applied",
                "Ask the user which existing label to use.",
            ],
            id="every unknown name is reported at once",
        ),
    ],
)
async def test_update_reports_unknown_names_without_dropping_the_rest(
    gitlab_client_mock, update_tool, kwargs, lookups, expected_widget, reported
):
    gitlab_client_mock.graphql.side_effect = [
        *lookups,
        {"workItemUpdate": {"workItem": {"id": "gid://gitlab/WorkItem/123"}}},
    ]

    response = json.loads(
        await update_tool._arun(
            project_id="namespace/project", work_item_iid=42, **kwargs
        )
    )

    gql_input = gitlab_client_mock.graphql.call_args[0][1]["input"]
    assert gql_input.get("labelsWidget") == expected_widget
    assert gql_input.get("title") == kwargs.get("title")

    warnings = " ".join(response["warnings"])
    assert [fragment for fragment in reported if fragment in warnings] == reported


@pytest.mark.asyncio
async def test_update_redirects_label_names_passed_as_ids(
    gitlab_client_mock, update_tool
):
    """Skipping the malformed ID keeps the redirect without costing the rest of the update."""
    gitlab_client_mock.graphql.return_value = {
        "workItemUpdate": {"workItem": {"id": "gid://gitlab/WorkItem/123"}}
    }

    response = json.loads(
        await update_tool._arun(
            project_id="namespace/project",
            work_item_iid=42,
            add_label_ids=["testing"],
            title="New title",
        )
    )

    gql_input = gitlab_client_mock.graphql.call_args[0][1]["input"]
    assert gql_input["title"] == "New title"
    assert "labelsWidget" not in gql_input

    warning = response["warnings"][0]
    assert "add_label_ids accepts numeric IDs" in warning


@pytest.mark.asyncio
async def test_create_resolves_label_names_against_group(gitlab_client_mock, metadata):
    tool = CreateWorkItem(description="create work item", metadata=metadata)
    tool._validate_parent_url = AsyncMock(
        return_value=ResolvedParent(type="group", full_path="namespace/group")
    )
    gitlab_client_mock.graphql.side_effect = [
        exact_labels("group", {"id": "gid://gitlab/GroupLabel/7", "title": "testing"}),
        {
            "namespace": {
                "workItemTypes": {
                    "nodes": [{"id": "gid://gitlab/WorkItems::Type/1", "name": "Issue"}]
                }
            }
        },
        {
            "workItemCreate": {
                "workItem": {"id": "gid://gitlab/WorkItem/1", "title": "New"},
                "errors": [],
            }
        },
    ]

    await tool._arun(
        group_id="namespace/group",
        title="New",
        type_name="Issue",
        labels=["testing"],
    )

    assert gitlab_client_mock.graphql.call_args_list[0][0][0] == GET_GROUP_LABELS_QUERY

    gql_input = gitlab_client_mock.graphql.call_args_list[2][0][1]["input"]
    assert gql_input["labelsWidget"]["labelIds"] == ["gid://gitlab/GroupLabel/7"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("own_lookup", "expected_id"),
    [
        pytest.param([OWN_GROUP_LABEL], OWN_GROUP_LABEL["id"], id="own label"),
        pytest.param([], ANCESTOR_GROUP_LABEL["id"], id="ancestors only"),
    ],
)
async def test_update_prefers_the_group_label_over_an_ancestor_one(
    gitlab_client_mock, metadata, own_lookup, expected_id
):
    """A group and its ancestor share the GroupLabel scope, so the group is asked alone."""
    tool = UpdateWorkItem(description="update work item", metadata=metadata)
    tool._resolve_work_item_data = AsyncMock(
        return_value=ResolvedWorkItem(
            id="gid://gitlab/WorkItem/123",
            full_data={"workItemType": {"name": "Issue"}},
            parent=ResolvedParent(type="group", full_path="namespace/group"),
        )
    )
    gitlab_client_mock.graphql.side_effect = [
        exact_labels("group", ANCESTOR_GROUP_LABEL, OWN_GROUP_LABEL),
        exact_labels("group", *own_lookup),
        {"workItemUpdate": {"workItem": {"id": "gid://gitlab/WorkItem/123"}}},
    ]

    await tool._arun(
        group_id="namespace/group", work_item_iid=42, add_labels=["testing"]
    )

    assert gitlab_client_mock.graphql.call_args_list[1][0][1] == {
        "fullPath": "namespace/group",
        "name": "testing",
        "withSimilar": False,
        "includeAncestors": False,
    }

    gql_input = gitlab_client_mock.graphql.call_args_list[2][0][1]["input"]
    assert gql_input["labelsWidget"]["addLabelIds"] == [expected_id]


@pytest.mark.asyncio
@patch(
    "duo_workflow_service.tools.work_items.base_tool.supports_labels_by_name",
    return_value=False,
)
async def test_update_skips_label_names_on_an_older_instance(
    _mock_supports, gitlab_client_mock, update_tool
):
    """The lookup cannot succeed, so it is not attempted."""
    gitlab_client_mock.graphql.return_value = {
        "workItemUpdate": {"workItem": {"id": "gid://gitlab/WorkItem/123"}}
    }

    response = json.loads(
        await update_tool._arun(
            project_id="namespace/project",
            work_item_iid=42,
            add_labels=["testing"],
            title="New title",
        )
    )

    gitlab_client_mock.graphql.assert_awaited_once()
    gql_input = gitlab_client_mock.graphql.call_args[0][1]["input"]
    assert gql_input["title"] == "New title"
    assert "labelsWidget" not in gql_input

    assert "requires GitLab 19.4 or later" in response["warnings"][0]
