from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.chat.base import BaseToolsRegistry
from ai_gateway.chat.tools import BaseTool
from ai_gateway.chat.tools.gitlab import (
    BuildReader,
    CommitReader,
    EpicReader,
    GitlabDocumentation,
    IssueReader,
    MergeRequestReader,
    SelfHostedGitlabDocumentation,
)

__all__ = ["DuoChatToolsRegistry"]


class DuoChatToolsRegistry(BaseToolsRegistry):

    def __init__(self, self_hosted_documentation_enabled: bool = False):
        self.self_hosted_documentation_enabled = self_hosted_documentation_enabled

    @property
    def tools(self) -> list[BaseTool]:
        # We can also read the list of tools and associated unit primitives from the file
        # similar to what we implemented for the Prompt Registry
        tools: list[BaseTool] = [
            BuildReader(),
            EpicReader(),
            IssueReader(),
            MergeRequestReader(),
            CommitReader(),
        ]

        if self.self_hosted_documentation_enabled:
            tools.append(SelfHostedGitlabDocumentation())
        else:
            tools.append(GitlabDocumentation())

        return tools

    def get_on_behalf(self, user: StarletteUser, gl_version: str) -> list[BaseTool]:
        _tools = []

        for tool in self.tools:
            if not user.can(tool.unit_primitive):
                continue

            if not tool.is_compatible(gl_version):
                continue

            _tools.append(tool)

        return _tools

    def get_all(self) -> list[BaseTool]:
        return self.tools
