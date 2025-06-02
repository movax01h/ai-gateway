from typing import Optional

from pydantic import BaseModel

__all__ = ["AdditionalContext"]


# Note: additionaL_context is an alias for injected_context
class AdditionalContext(BaseModel):
    # One of "file", "snippet", "merge_request", "issue", "dependency", "local_git", "terminal", "repository", "directory"
    # The corresponding unit primitives must be registered with `include_{category}_context` format.
    # https://gitlab.com/gitlab-org/cloud-connector/gitlab-cloud-connector/-/tree/main/config/unit_primitives
    category: str
    id: Optional[str] = None
    content: Optional[str] = None
    metadata: Optional[dict] = None
