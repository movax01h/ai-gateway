from contextvars import ContextVar, Token
from typing import Optional

from duo_workflow_service.gitlab.gitlab_api import Namespace, Project
from duo_workflow_service.gitlab.gitlab_instance_info_service import (
    GitLabInstanceInfo,
    GitLabInstanceInfoService,
)

# Context variable for storing GitLab instance info per async task
_gitlab_context: ContextVar[Optional[GitLabInstanceInfo]] = ContextVar(
    "gitlab_context", default=None
)


class GitLabServiceContext:
    """Context manager for providing GitLab instance information within a scope.

    This context manager creates GitLab instance information from project/namespace
    data and makes it available to any code within the context via get_current_instance_info().

    Usage:
        with GitLabServiceContext(service, project, namespace):
            # Any code here can access GitLab info
            info = GitLabServiceContext.get_current_instance_info()
    """

    def __init__(
        self,
        service: GitLabInstanceInfoService,
        project: Optional[Project] = None,
        namespace: Optional[Namespace] = None,
    ):
        """Initialize the context manager.

        Args:
            service: GitLab instance info service to use
            project: Optional project data
            namespace: Optional namespace data
        """
        self.service = service
        self.project = project
        self.namespace = namespace
        self._token: Optional[Token[Optional[GitLabInstanceInfo]]] = None

    def __enter__(self) -> "GitLabServiceContext":
        """Enter the context and set up GitLab instance information."""
        instance_info = self.service.create_from_project_and_namespace(
            self.project, self.namespace
        )
        self._token = _gitlab_context.set(instance_info)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and clean up."""
        if self._token is not None:
            _gitlab_context.reset(self._token)

    @classmethod
    def get_current_instance_info(cls) -> Optional[GitLabInstanceInfo]:
        """Get the current GitLab instance information from context.

        Returns:
            GitLabInstanceInfo if within a GitLabServiceContext, None otherwise
        """
        return _gitlab_context.get()
