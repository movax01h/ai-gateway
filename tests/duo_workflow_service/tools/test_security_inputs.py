from typing import Any

import pytest
from pydantic import BaseModel, ValidationError, field_validator

from duo_workflow_service.tools.security_inputs import (
    MAX_GITLAB_GLOBAL_ID_LENGTH,
    MAX_PROJECT_FULL_PATH_LENGTH,
    MAX_VULNERABILITY_IDS,
    GitLabGlobalIdStr,
    ProjectFullPath,
    VulnerabilityIdList,
)


class _ProjectPathModel(BaseModel):
    project_full_path: ProjectFullPath


class _CoercedIdModel(BaseModel):
    id: GitLabGlobalIdStr

    @field_validator("id", mode="before")
    @classmethod
    def coerce_to_string(cls, v: Any) -> str:
        return str(v)


class _IdListModel(BaseModel):
    ids: VulnerabilityIdList


def test_limits_match_documented_values():
    assert MAX_PROJECT_FULL_PATH_LENGTH == 255
    assert MAX_GITLAB_GLOBAL_ID_LENGTH == 255
    assert MAX_VULNERABILITY_IDS == 100


@pytest.mark.parametrize(
    "path",
    [
        "abc",
        "namespace/project",
        "group/subgroup/project",
        "my-group_123/project.name",
        "g/" + "p" * (MAX_PROJECT_FULL_PATH_LENGTH - 2),
    ],
)
def test_project_full_path_accepts_valid_paths(path):
    assert _ProjectPathModel(project_full_path=path).project_full_path == path


@pytest.mark.parametrize(
    "path,expected_error",
    [
        ("ab", "at least 3 characters"),
        ("", "at least 3 characters"),
        (
            "g/" + "p" * (MAX_PROJECT_FULL_PATH_LENGTH - 1),
            f"at most {MAX_PROJECT_FULL_PATH_LENGTH} characters",
        ),
        ("../../../etc/passwd", "Invalid project path format"),
        ("group/../project", "Invalid project path format"),
        ("/absolute/path", "Invalid project path format"),
        ("path/to/project/", "Invalid project path format"),
    ],
)
def test_project_full_path_rejects_invalid_paths(path, expected_error):
    with pytest.raises(ValidationError, match=expected_error):
        _ProjectPathModel(project_full_path=path)


def test_project_full_path_length_checked_before_content():
    # An oversized value containing ".." fails on length, before the content check runs.
    path = ".." + "a" * MAX_PROJECT_FULL_PATH_LENGTH
    with pytest.raises(ValidationError) as exc_info:
        _ProjectPathModel(project_full_path=path)

    errors = exc_info.value.errors()
    assert [error["type"] for error in errors] == ["string_too_long"]


@pytest.mark.parametrize(
    "value,expected",
    [
        ("gid://gitlab/Vulnerability/123", "gid://gitlab/Vulnerability/123"),
        (123, "123"),
        ("9" * MAX_GITLAB_GLOBAL_ID_LENGTH, "9" * MAX_GITLAB_GLOBAL_ID_LENGTH),
        (int("9" * MAX_GITLAB_GLOBAL_ID_LENGTH), "9" * MAX_GITLAB_GLOBAL_ID_LENGTH),
    ],
)
def test_global_id_accepts_values_within_bound(value, expected):
    assert _CoercedIdModel(id=value).id == expected


@pytest.mark.parametrize(
    "value",
    [
        "9" * (MAX_GITLAB_GLOBAL_ID_LENGTH + 1),
        # The bound applies to the string produced by the before-validator.
        int("9" * (MAX_GITLAB_GLOBAL_ID_LENGTH + 1)),
    ],
)
def test_global_id_rejects_oversized_values(value):
    with pytest.raises(
        ValidationError, match=f"at most {MAX_GITLAB_GLOBAL_ID_LENGTH} characters"
    ):
        _CoercedIdModel(id=value)


def test_global_id_rejects_empty_string():
    with pytest.raises(ValidationError, match="at least 1 character"):
        _CoercedIdModel(id="")


@pytest.mark.parametrize("count", [0, 1, MAX_VULNERABILITY_IDS])
def test_vulnerability_id_list_accepts_up_to_limit(count):
    ids = [f"gid://gitlab/Vulnerability/{i}" for i in range(count)]
    assert _IdListModel(ids=ids).ids == ids


@pytest.mark.parametrize(
    "ids,expected_error",
    [
        (
            [str(i) for i in range(MAX_VULNERABILITY_IDS + 1)],
            f"at most {MAX_VULNERABILITY_IDS} items",
        ),
        (
            ["1", "x" * (MAX_GITLAB_GLOBAL_ID_LENGTH + 1)],
            f"at most {MAX_GITLAB_GLOBAL_ID_LENGTH} characters",
        ),
        (["1", ""], "at least 1 character"),
    ],
)
def test_vulnerability_id_list_rejects_oversized_input(ids, expected_error):
    with pytest.raises(ValidationError, match=expected_error):
        _IdListModel(ids=ids)
