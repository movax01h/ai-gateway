"""Fixtures specific to the analytics agent tests."""

# pylint: disable=redefined-outer-name,import-outside-toplevel

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import yaml

from agent_tests.conftest import make_prompt_adapter_class


def pytest_collection_modifyitems(items):
    """Auto-apply analytics marker to all tests in this directory."""
    for item in items:
        if "/agent_tests/analytics_agent/" in str(item.path):
            item.add_marker(pytest.mark.analytics)


@pytest.fixture
def mock_gitlab_client():
    """Mock GitLab client for GLQL responses.

    Tests must configure responses via mock_glql_response() from helpers.
    """
    client = AsyncMock()
    client.apost = AsyncMock()
    return client


@pytest.fixture(autouse=True)
def mock_gitlab_version_18_6():
    """Mock GitLab version as 18.6.0 (required for GLQL)."""
    with patch("duo_workflow_service.tools.run_glql_query.gitlab_version") as mock:
        mock.get.return_value = "18.6.0"
        yield mock


@pytest.fixture
def glql_tool(mock_gitlab_client):
    """RunGLQLQuery tool with mocked GitLab client."""
    from duo_workflow_service.tools.run_glql_query import RunGLQLQuery

    return RunGLQLQuery(metadata={"gitlab_client": mock_gitlab_client})


@pytest.fixture
def work_item_note_tool(mock_gitlab_client):
    """CreateWorkItemNote tool with mocked GitLab client."""
    from duo_workflow_service.tools.work_item import CreateWorkItemNote

    return CreateWorkItemNote(metadata={"gitlab_client": mock_gitlab_client})


@pytest.fixture
def analytics_system_template():
    """Load the analytics agent system template from YAML config file."""
    config_path = (
        Path(__file__).resolve().parents[2]
        / "duo_workflow_service"
        / "agent_platform"
        / "v1"
        / "flows"
        / "configs"
        / "analytics_agent.yml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["prompts"][0]["prompt_template"]["system"]


@pytest.fixture
def analytics_agent(
    real_llm,
    analytics_system_template,
    glql_tool,
    work_item_note_tool,
    mock_tools_registry,
):
    """Analytics agent with real LLM and mocked tools."""
    from duo_workflow_service.agents.chat_agent import ChatAgent

    RealLLMPromptAdapter = make_prompt_adapter_class()
    adapter = RealLLMPromptAdapter(
        model=real_llm,
        system_template=analytics_system_template,
        tools=[glql_tool, work_item_note_tool],
        agent_name="analytics_agent",
    )

    return ChatAgent(
        name="analytics_agent",
        prompt_adapter=adapter,
        tools_registry=mock_tools_registry,
        system_template_override=None,
    )
