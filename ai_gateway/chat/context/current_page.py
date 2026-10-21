from typing import Annotated, Literal, Union

from pydantic import AfterValidator, BaseModel, ConfigDict

__all__ = [
    "MAX_TITLE_LENGTH",
    "CiBuildContext",
    "CommitContext",
    "Context",
    "CurrentPageContext",
    "EpicContext",
    "IssueContext",
    "MergeRequestContext",
    "WorkItemContext",
]

# Matches GitLab's Issuable::TITLE_LENGTH_MAX. Commit titles are already truncated by GitLab to under 100 characters.
MAX_TITLE_LENGTH = 255


def _truncate_title(title: str) -> str:
    # Truncate rather than reject: some GitLab import paths skip the title length validation, and rejecting would
    # break every chat request on that page.
    return title[:MAX_TITLE_LENGTH]


Title = Annotated[str, AfterValidator(_truncate_title)]


class Context(BaseModel):
    """Represents current page context and gets its prompt content from GitLab application.

    This class is deprecated but is needed to process requests from GitLab instances earlier than 17.9. This class
    should be deleted as soon as we stop supporting GitLab 17.9, that should happen after two major releases.
    """

    type: Literal["issue", "epic", "merge_request", "commit", "build"]
    content: str

    model_config = ConfigDict(frozen=True)


class CiBuildContext(BaseModel):
    type: Literal["build"]


class CommitContext(BaseModel):
    type: Literal["commit"]
    title: Title


class EpicContext(BaseModel):
    type: Literal["epic"]
    title: Title


class IssueContext(BaseModel):
    type: Literal["issue"]
    title: Title


class MergeRequestContext(BaseModel):
    type: Literal["merge_request"]
    title: Title


class WorkItemContext(BaseModel):
    type: Literal["work_item"]
    title: Title


# This Union allows to pass page context params and automatically
# define a correct instance of specific context
CurrentPageContext = Union[
    Context,
    CiBuildContext,
    CommitContext,
    EpicContext,
    IssueContext,
    MergeRequestContext,
    WorkItemContext,
]
