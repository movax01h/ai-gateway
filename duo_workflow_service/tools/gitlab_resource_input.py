from typing import Optional, Union

from pydantic import BaseModel, Field


class GitLabResourceInput(BaseModel):
    """Base model for GitLab resource inputs that can accept either a URL or resource IDs."""

    url: Optional[str] = Field(
        default=None,
        description="GitLab URL for the resource. If provided, other ID fields are not required.",
    )


class ProjectResourceInput(GitLabResourceInput):
    """Base input model for resources that belong to a project."""

    project_id: Optional[Union[int, str]] = Field(
        default=None,
        description="The ID or URL-encoded path of the project. Examples: 123, 'gitlab-org%2Fgitlab'. Required if URL is not provided.",
    )
