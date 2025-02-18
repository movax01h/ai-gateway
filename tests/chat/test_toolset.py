from typing import Type

import pytest
from gitlab_cloud_connector import CloudConnectorUser, GitLabUnitPrimitive, UserClaims

from ai_gateway.api.auth_utils import StarletteUser
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
from ai_gateway.chat.toolset import DuoChatToolsRegistry


class TestDuoChatToolRegistry:
    @pytest.mark.parametrize(
        "expected_tools",
        [
            {
                BuildReader,
                EpicReader,
                IssueReader,
                MergeRequestReader,
                GitlabDocumentation,
                CommitReader,
            }
        ],
    )
    def test_get_all_success(self, expected_tools: set[Type[BaseTool]]):
        tools = DuoChatToolsRegistry().get_all()
        actual_tools = {type(tool) for tool in tools}

        assert set(actual_tools) == set(expected_tools)

    def test_get_all_with_self_hosted_documentation(
        self,
    ):
        tools = DuoChatToolsRegistry(self_hosted_documentation_enabled=True).get_all()
        actual_tools = {type(tool) for tool in tools}

        assert actual_tools == {
            BuildReader,
            SelfHostedGitlabDocumentation,
            EpicReader,
            IssueReader,
            MergeRequestReader,
            CommitReader,
        }

    @pytest.mark.parametrize(
        ("unit_primitives", "expected_tools"),
        [
            ([GitLabUnitPrimitive.DOCUMENTATION_SEARCH], {GitlabDocumentation}),
            ([GitLabUnitPrimitive.ASK_EPIC], {EpicReader}),
            ([GitLabUnitPrimitive.ASK_ISSUE], {IssueReader}),
            (
                [
                    GitLabUnitPrimitive.DUO_CHAT,
                    GitLabUnitPrimitive.DOCUMENTATION_SEARCH,
                    GitLabUnitPrimitive.ASK_EPIC,
                    GitLabUnitPrimitive.ASK_ISSUE,
                ],
                {
                    GitlabDocumentation,
                    EpicReader,
                    IssueReader,
                },
            ),
            (
                [
                    GitLabUnitPrimitive.COMPLETE_CODE,
                    GitLabUnitPrimitive.GENERATE_CODE,
                ],
                set(),
            ),
        ],
    )
    def test_get_on_behalf_success(
        self,
        unit_primitives: list[GitLabUnitPrimitive],
        expected_tools: set[Type[BaseTool]],
    ):
        user = StarletteUser(
            CloudConnectorUser(
                authenticated=True,
                claims=UserClaims(scopes=[u.value for u in unit_primitives]),
            )
        )

        tools = DuoChatToolsRegistry().get_on_behalf(user, "")
        actual_tools = {type(tool) for tool in tools}

        assert set(actual_tools) == set(expected_tools)

    @pytest.mark.parametrize(
        "unit_primitives",
        [
            [
                GitLabUnitPrimitive.COMPLETE_CODE,
                GitLabUnitPrimitive.GENERATE_CODE,
                GitLabUnitPrimitive.EXPLAIN_CODE,
            ]
        ],
    )
    def test_get_on_behalf_empty(
        self,
        unit_primitives: list[GitLabUnitPrimitive],
    ):
        user = StarletteUser(
            CloudConnectorUser(
                authenticated=True,
                claims=UserClaims(scopes=[u.value for u in unit_primitives]),
            )
        )

        tools = DuoChatToolsRegistry().get_on_behalf(user, "")

        assert len(tools) == 0
