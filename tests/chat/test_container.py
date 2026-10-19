from typing import Type, cast
from unittest.mock import Mock

import pytest
from dependency_injector import containers, providers
from langchain_core.runnables import Runnable

from ai_gateway.chat.executor import GLAgentRemoteExecutor
from ai_gateway.chat.tools import BaseTool
from ai_gateway.chat.tools.gitlab import (
    GitlabDocumentation,
    SelfHostedGitlabDocumentation,
)


@pytest.fixture(name="config_values")
def config_values_fixture(custom_models_enabled: bool):
    return {"custom_models": {"enabled": custom_models_enabled}}


@pytest.fixture(name="mock_agent")
def mock_agent_fixture():
    return Mock(spec=Runnable)


@pytest.mark.parametrize("custom_models_enabled", [False])
def test_container(
    mock_ai_gateway_container: containers.DeclarativeContainer, mock_agent: Mock
):
    chat = cast(providers.Container, mock_ai_gateway_container.chat)

    assert isinstance(
        chat.gl_agent_remote_executor_factory(agent=mock_agent), GLAgentRemoteExecutor
    )


@pytest.mark.parametrize(
    ("custom_models_enabled", "expected_tool_type"),
    [(True, SelfHostedGitlabDocumentation), (False, GitlabDocumentation)],
)
def test_container_with_config(
    mock_ai_gateway_container: containers.DeclarativeContainer,
    expected_tool_type: Type[BaseTool],
    mock_agent: Runnable,
):
    chat = cast(providers.Container, mock_ai_gateway_container.chat)

    tool_types = {
        type(tool)
        for tool in chat.gl_agent_remote_executor_factory(agent=mock_agent).tools
    }

    assert expected_tool_type in tool_types
