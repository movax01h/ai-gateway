import pytest

from duo_workflow_service.agents.project_utils import resolve_project_name_for_tool
from duo_workflow_service.gitlab.gitlab_api import Project


def _make_project(id: int = 42, name: str = "my-project") -> Project:
    return Project(
        id=id,
        name=name,
        description="",
        http_url_to_repo="",
        web_url="",
        default_branch="main",
        languages=[],
        exclusion_rules=[],
    )


@pytest.mark.parametrize(
    "project, tool_call, expected",
    [
        pytest.param(
            _make_project(id=42, name="my-project"),
            {"args": {"project_id": 42}},
            "my-project",
            id="match_int_int",
        ),
        pytest.param(
            _make_project(id=42, name="my-project"),
            {"args": {"project_id": "42"}},
            "my-project",
            id="match_string_int",
        ),
        pytest.param(
            _make_project(id=42, name="my-project"),
            {"args": {"project_id": 99}},
            None,
            id="different_project_id",
        ),
        pytest.param(
            _make_project(id=42, name="my-project"),
            {"args": {"project_id": "99"}},
            None,
            id="different_project_id_string",
        ),
        pytest.param(
            _make_project(id=21, name="other-project"),
            {"args": {"url": "https://gitlab.com/other/project"}},
            None,
            id="no_project_id_in_args",
        ),
        pytest.param(
            _make_project(id=42, name="my-project"),
            {"args": {}},
            None,
            id="empty_args",
        ),
        pytest.param(
            _make_project(id=42, name="my-project"),
            {},
            None,
            id="no_args_key",
        ),
        pytest.param(
            None,
            {"args": {"project_id": 42}},
            None,
            id="no_project",
        ),
        pytest.param(
            _make_project(id=42, name="my-project"),
            {"args": None},
            None,
            id="args_is_none",
        ),
    ],
)
def test_resolve_project_name_for_tool(project, tool_call, expected):
    assert resolve_project_name_for_tool(project, tool_call) == expected
