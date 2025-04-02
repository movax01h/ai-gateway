from typing import Literal, Union

from pydantic import BaseModel, ConfigDict

__all__ = [
    "Context",
    "CiBuildContext",
    "CommitContext",
    "EpicContext",
    "IssueContext",
    "MergeRequestContext",
    "CurrentPageContext",
]


class Context(BaseModel):
    """
    Represents current page context and gets its prompt content from GitLab application.
    This class is deprecated but is needed to process requests from
    GitLab instances earlier than 17.9. This class should be deleted as soon as we
    stop supporting GitLab 17.9, that should happen after two major releases.
    """

    type: Literal["issue", "epic", "merge_request", "commit", "build"]
    content: str

    model_config = ConfigDict(frozen=True)


class CiBuildContext(BaseModel):
    type: Literal["build"]


class CommitContext(BaseModel):
    type: Literal["commit"]
    title: str


class EpicContext(BaseModel):
    type: Literal["epic"]
    title: str


class IssueContext(BaseModel):
    type: Literal["issue"]
    title: str


class MergeRequestContext(BaseModel):
    type: Literal["merge_request"]
    title: str


# This Union allows to pass page context params and automatically
# define a correct instance of specific context
CurrentPageContext = Union[
    Context,
    CiBuildContext,
    CommitContext,
    EpicContext,
    IssueContext,
    MergeRequestContext,
]
