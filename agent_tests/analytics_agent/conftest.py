"""Fixtures specific to the analytics agent tests."""

# pylint: disable=redefined-outer-name,import-outside-toplevel

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
import yaml

from agent_tests.conftest import make_prompt_adapter_class
from duo_workflow_service.conversation.history_optimizer.pipeline import (
    HistoryOptimizerPipeline,
)
from duo_workflow_service.conversation.history_optimizer.schema import (
    OptimizationResult,
)
from lib.context import gitlab_version


def _make_passthrough_pipeline() -> HistoryOptimizerPipeline:
    """Build a HistoryOptimizerPipeline mock that returns history unchanged."""
    mock_pipeline = Mock(spec=HistoryOptimizerPipeline)

    async def optimize(history):
        return history, [OptimizationResult(messages=history, was_modified=False)]

    mock_pipeline.optimize = AsyncMock(side_effect=optimize)
    return mock_pipeline


def pytest_collection_modifyitems(items):
    """Auto-apply analytics marker to all tests in this directory."""
    for item in items:
        if "/agent_tests/analytics_agent/" in str(item.path):
            item.add_marker(pytest.mark.analytics)


# Which flow config each test runs against. The agent is served from versioned
# configs, so a test has to say which prompt it is making claims about: they
# differ in the tools they offer and in what the schema they read looks like.
SCHEMA_TOOL_NAMES = {
    "1.0.0": "get_glql_schema",
    "2.0.0": "fetch_glql_schema",
}


def pytest_generate_tests(metafunc):
    """Run each test once per version in its `flow_versions` marker.

    Every test that uses `flow_version` (directly or via the agent fixtures) must carry
    the marker; a missing or empty marker fails collection instead of quietly skipping.
    """
    if "flow_version" not in metafunc.fixturenames:
        return

    marker = metafunc.definition.get_closest_marker("flow_versions")
    if marker is None or not marker.args:
        raise pytest.UsageError(
            f"{metafunc.definition.nodeid} needs a `flow_versions` marker naming "
            f"the flow configs it runs against, e.g. "
            f'@pytest.mark.flow_versions("1.0.0")'
        )

    unknown = set(marker.args) - SCHEMA_TOOL_NAMES.keys()
    if unknown:
        raise pytest.UsageError(
            f"{metafunc.definition.nodeid} names unknown flow "
            f"version(s) {sorted(unknown)}"
        )

    metafunc.parametrize("flow_version", marker.args, ids=str)


@pytest.fixture
def mock_gitlab_client():
    """Mock GitLab client for GLQL responses.

    Tests must configure responses via mock_glql_response() from helpers.
    """
    client = AsyncMock()
    client.apost = AsyncMock()
    return client


@pytest.fixture(autouse=True)
def mock_gitlab_version():
    """Set the GitLab version to 19.3.0, which clears both GLQL version floors.

    GLQL itself needs 18.6 and the schema endpoint 19.3. Every reader imports
    the same context var, so one set covers them all.
    """
    token = gitlab_version.set("19.3.0")
    yield
    gitlab_version.reset(token)


@pytest.fixture
def glql_tool(mock_gitlab_client):
    """RunGLQLQuery tool with mocked GitLab client."""
    from duo_workflow_service.tools.run_glql_query import RunGLQLQuery

    return RunGLQLQuery(metadata={"gitlab_client": mock_gitlab_client})


@pytest.fixture
def schema_tool_name(flow_version):
    """The name of the schema tool the version under test offers.

    Tests assert on the name rather than hardcoding it, since it differs between flow versions.
    """
    return SCHEMA_TOOL_NAMES[flow_version]


@pytest.fixture
def glql_schema_tool(flow_version, mock_gitlab_client):
    """The schema tool the version under test offers.

    1.0.0 answers from a copy bundled in this repo and needs no client. 2.0.0
    reads the instance, so it gets a stubbed endpoint serving
    `glql_schema_fake.json` - a trimmed down, representative version of the
    full schema. The stub answers on path, so a tool added to the toolset later
    cannot silently receive the schema for its own GET.
    """
    if flow_version == "1.0.0":
        from duo_workflow_service.tools.get_glql_schema import GetGlqlSchema

        tool = GetGlqlSchema(metadata={})
    else:
        from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
        from duo_workflow_service.tools.fetch_glql_schema import (
            SCHEMA_PATH,
            FetchGlqlSchema,
        )

        document = (Path(__file__).parent / "glql_schema_fake.json").read_text()

        async def aget(path, **_kwargs):
            if path != SCHEMA_PATH:
                return GitLabHttpResponse(status_code=404, body="404 Not Found")

            return GitLabHttpResponse(status_code=200, body=document)

        mock_gitlab_client.aget = AsyncMock(side_effect=aget)

        tool = FetchGlqlSchema(metadata={"gitlab_client": mock_gitlab_client})

    # Fail fast if a new flow version quietly inherits this branch's tool.
    assert tool.name == SCHEMA_TOOL_NAMES[flow_version]
    return tool


@pytest.fixture
def work_item_note_tool(mock_gitlab_client):
    """CreateWorkItemNote tool with mocked GitLab client."""
    from duo_workflow_service.tools.work_item import CreateWorkItemNote

    return CreateWorkItemNote(metadata={"gitlab_client": mock_gitlab_client})


@pytest.fixture
def merge_request_note_tool(mock_gitlab_client):
    """CreateMergeRequestNote tool with mocked GitLab client."""
    from duo_workflow_service.tools.merge_request_notes import CreateMergeRequestNote

    return CreateMergeRequestNote(
        metadata={"gitlab_client": mock_gitlab_client, "gitlab_host": "gitlab.com"}
    )


@pytest.fixture
def analytics_system_template(flow_version):
    """Load the analytics agent system template from YAML config file."""
    config_path = (
        Path(__file__).resolve().parents[2]
        / "duo_workflow_service"
        / "agent_platform"
        / "v1"
        / "flows"
        / "configs"
        / "analytics_agent"
        / f"{flow_version}.yml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config["prompts"][0]["prompt_template"]["system"]


@pytest.fixture
def analytics_agent(
    real_llm,
    analytics_system_template,
    glql_schema_tool,
    glql_tool,
    work_item_note_tool,
    merge_request_note_tool,
    orbit_list_commands_tool,
    orbit_invoke_command_tool,
    mock_tools_registry,
):
    """Analytics agent with the full toolset including mock Orbit MCP tools."""
    from duo_workflow_service.agents.chat_agent import ChatAgent
    from duo_workflow_service.tools.toolset import Toolset

    all_tools = [
        glql_schema_tool,
        glql_tool,
        work_item_note_tool,
        merge_request_note_tool,
        orbit_list_commands_tool,
        orbit_invoke_command_tool,
    ]

    RealLLMPromptAdapter = make_prompt_adapter_class()
    adapter = RealLLMPromptAdapter(
        model=real_llm,
        system_template=analytics_system_template,
        tools=all_tools,
        agent_name="analytics_agent",
    )

    tools_dict = {tool.name: tool for tool in all_tools}
    return ChatAgent(
        name="analytics_agent",
        prompt_adapter=adapter,
        tools_registry=mock_tools_registry,
        system_template_override=None,
        toolset=Toolset(pre_approved=set(), all_tools=tools_dict),
        optimizer_pipeline=_make_passthrough_pipeline(),
    )


@pytest.fixture
def analytics_agent_without_orbit(
    real_llm,
    analytics_system_template,
    glql_schema_tool,
    glql_tool,
    work_item_note_tool,
    merge_request_note_tool,
    mock_tools_registry,
):
    """Analytics agent with real LLM and mocked tools (no Orbit)."""
    from duo_workflow_service.agents.chat_agent import ChatAgent
    from duo_workflow_service.tools.toolset import Toolset

    all_tools = [
        glql_schema_tool,
        glql_tool,
        work_item_note_tool,
        merge_request_note_tool,
    ]

    RealLLMPromptAdapter = make_prompt_adapter_class()
    adapter = RealLLMPromptAdapter(
        model=real_llm,
        system_template=analytics_system_template,
        tools=all_tools,
        agent_name="analytics_agent",
    )

    tools_dict = {tool.name: tool for tool in all_tools}
    return ChatAgent(
        name="analytics_agent",
        prompt_adapter=adapter,
        tools_registry=mock_tools_registry,
        system_template_override=None,
        toolset=Toolset(pre_approved=set(), all_tools=tools_dict),
        optimizer_pipeline=_make_passthrough_pipeline(),
    )
