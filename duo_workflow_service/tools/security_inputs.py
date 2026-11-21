"""Bounded input types shared by the security tools.

These types cap the size of tool arguments in the Pydantic ``args_schema`` so that
oversized values are rejected before any request reaches GitLab.
"""

from typing import Annotated

from pydantic import AfterValidator, Field, StringConstraints

__all__ = [
    "MAX_GITLAB_GLOBAL_ID_LENGTH",
    "MAX_PROJECT_FULL_PATH_LENGTH",
    "MAX_VULNERABILITY_IDS",
    "MIN_PROJECT_FULL_PATH_LENGTH",
    "GitLabGlobalIdStr",
    "ProjectFullPath",
    "VulnerabilityIdList",
]

# GitLab validates every route path (the full path of a project or namespace) with
# `length: { within: 1..255 }` in `app/models/route.rb`, so no real project has a
# longer full path.
MAX_PROJECT_FULL_PATH_LENGTH = 255
MIN_PROJECT_FULL_PATH_LENGTH = 3

# A GitLab global ID looks like `gid://gitlab/MergeRequest/123`. The longest model
# name plus a 64-bit integer ID is far below this bound.
MAX_GITLAB_GLOBAL_ID_LENGTH = 255

# Matches the "Up to 100" limit documented for the vulnerability ID list fields.
MAX_VULNERABILITY_IDS = 100


def _validate_project_full_path(value: str) -> str:
    if ".." in value or value.startswith("/") or value.endswith("/"):
        raise ValueError("Invalid project path format")
    return value


# The length constraints run before the AfterValidator, so an oversized value is
# rejected without scanning its contents.
ProjectFullPath = Annotated[
    str,
    StringConstraints(
        min_length=MIN_PROJECT_FULL_PATH_LENGTH,
        max_length=MAX_PROJECT_FULL_PATH_LENGTH,
    ),
    AfterValidator(_validate_project_full_path),
]

GitLabGlobalIdStr = Annotated[
    str,
    StringConstraints(max_length=MAX_GITLAB_GLOBAL_ID_LENGTH),
]

VulnerabilityIdList = Annotated[
    list[GitLabGlobalIdStr],
    Field(max_length=MAX_VULNERABILITY_IDS),
]
