"""Bounded input types shared by the security tools.

These types cap the size of tool arguments in the Pydantic ``args_schema`` so that
oversized values are rejected before any request reaches GitLab.
"""

from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, Field, StringConstraints

__all__ = [
    "MAX_GITLAB_GLOBAL_ID_LENGTH",
    "MAX_GIT_REF_LENGTH",
    "MAX_NUMERIC_ID_LENGTH",
    "MAX_PROJECT_FULL_PATH_LENGTH",
    "MAX_SCANNER_IDS",
    "MAX_SCANNER_ID_LENGTH",
    "MAX_SECURITY_FINDING_UUID_LENGTH",
    "MAX_VULNERABILITY_COMMENT_LENGTH",
    "MAX_VULNERABILITY_IDS",
    "MIN_PROJECT_FULL_PATH_LENGTH",
    "GitLabGlobalIdStr",
    "GitRefStr",
    "NumericIdStr",
    "ProjectFullPath",
    "ScannerIdList",
    "SecurityFindingUuid",
    "VulnerabilityComment",
    "VulnerabilityIdList",
    "VulnerabilitySeverity",
]

# GitLab validates every route path (the full path of a project or namespace) with
# `length: { within: 1..255 }` in `app/models/route.rb`, so no real project has a
# longer full path.
MAX_PROJECT_FULL_PATH_LENGTH = 255
MIN_PROJECT_FULL_PATH_LENGTH = 3

# A GitLab global ID looks like `gid://gitlab/MergeRequest/123`. The longest model
# name plus a 64-bit integer ID is far below this bound.
MAX_GITLAB_GLOBAL_ID_LENGTH = 255

# A numeric database ID such as a vulnerability ID. A 64-bit integer has at most 20
# digits.
MAX_NUMERIC_ID_LENGTH = 20

# Security finding UUIDs are canonical 36-character UUIDs, for example
# `1e9a2bf7-0450-5894-8db5-895c98e39deb`.
MAX_SECURITY_FINDING_UUID_LENGTH = 36

# Git does not cap the total length of a ref name, but each path component is limited
# to 255 bytes by common filesystems. This bound is generous for real branch and tag
# names while still rejecting oversized values.
MAX_GIT_REF_LENGTH = 1024

# Scanner IDs are short identifiers such as `semgrep` or `gemnasium`, and a pipeline
# runs only a handful of scanners.
MAX_SCANNER_ID_LENGTH = 255
MAX_SCANNER_IDS = 100

# Matches the "Up to 100" limit documented for the vulnerability ID list fields.
MAX_VULNERABILITY_IDS = 100

# GitLab rejects vulnerability state-transition comments (dismiss, confirm, revert to
# detected) longer than 50,000 characters, so a longer value can never succeed.
MAX_VULNERABILITY_COMMENT_LENGTH = 50_000


class VulnerabilitySeverity(StrEnum):
    """Valid vulnerability severity levels."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"
    UNKNOWN = "UNKNOWN"


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
    StringConstraints(min_length=1, max_length=MAX_GITLAB_GLOBAL_ID_LENGTH),
]

VulnerabilityIdList = Annotated[
    list[GitLabGlobalIdStr],
    Field(max_length=MAX_VULNERABILITY_IDS),
]

VulnerabilityComment = Annotated[
    str,
    StringConstraints(max_length=MAX_VULNERABILITY_COMMENT_LENGTH),
]

NumericIdStr = Annotated[
    str,
    StringConstraints(max_length=MAX_NUMERIC_ID_LENGTH),
]

SecurityFindingUuid = Annotated[
    str,
    StringConstraints(max_length=MAX_SECURITY_FINDING_UUID_LENGTH),
]

GitRefStr = Annotated[
    str,
    StringConstraints(max_length=MAX_GIT_REF_LENGTH),
]

ScannerIdList = Annotated[
    list[Annotated[str, StringConstraints(max_length=MAX_SCANNER_ID_LENGTH)]],
    Field(max_length=MAX_SCANNER_IDS),
]
