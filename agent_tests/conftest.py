"""Shared fixtures for agent tests.

These fixtures provide common setup for testing DWS agents with real LLMs.
The execution and validation models are configurable via CLI options.

To run agent tests in parallel for speed:
    pytest agent_tests/ -n auto

Requires ANTHROPIC_API_KEY environment variable.
"""

# pylint: disable=redefined-outer-name,import-outside-toplevel,super-init-not-called

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Type
from unittest.mock import MagicMock

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from duo_workflow_service.entities.state import ChatWorkflowState


def pytest_addoption(parser):
    """Add custom CLI options for model selection."""
    parser.addoption(
        "--execution-model",
        default="claude-sonnet-4-6",
        help="Anthropic model for agent execution (default: claude-sonnet-4-6)",
    )
    parser.addoption(
        "--validation-model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model for LLM-as-judge validation (default: claude-haiku-4-5-20251001)",
    )


def pytest_configure(config):
    """Register custom markers and set model defaults from CLI options."""
    config.addinivalue_line(
        "markers", "analytics: mark test as an analytics agent test"
    )
    validation_model = config.getoption("--validation-model", default=None)
    if validation_model:
        from agent_tests import helpers

        helpers.DEFAULT_VALIDATION_MODEL = validation_model


def _make_prompt_adapter_class():
    from duo_workflow_service.agents.prompt_adapter import BasePromptAdapter

    class RealLLMPromptAdapter(BasePromptAdapter):
        """Prompt adapter that uses a real LLM for testing."""

        def __init__(
            self,
            model: ChatAnthropic,
            system_template: str,
            tools: list[Any],
            agent_name: str = "agent",
        ):
            self._model = model.bind_tools(tools)
            self._system_template = system_template
            self._tools = tools
            self._agent_name = agent_name

        def get_model(self):
            return self._model

        async def get_response(
            self,
            input: ChatWorkflowState,
            **kwargs,  # noqa: A002
        ) -> AIMessage:
            from langchain_core.messages import BaseMessage, SystemMessage

            messages: list[BaseMessage] = [SystemMessage(content=self._system_template)]

            agent_name = kwargs.get("agent_name", self._agent_name)
            if agent_name in input.get("conversation_history", {}):
                messages.extend(input["conversation_history"][agent_name])

            response = await self._model.ainvoke(messages)
            return response

    return RealLLMPromptAdapter


make_prompt_adapter_class = _make_prompt_adapter_class


@pytest.fixture
def execution_model(request):
    """Model name for agent execution, from --execution-model CLI option."""
    return request.config.getoption("--execution-model")


@pytest.fixture
def validation_model(request):
    """Model name for LLM-as-judge validation, from --validation-model CLI option."""
    return request.config.getoption("--validation-model")


@pytest.fixture
def real_llm(execution_model):
    """Real Anthropic model for testing, configured via --execution-model."""
    return ChatAnthropic(  # type: ignore[call-arg]
        model=execution_model,
        max_tokens=4096,
    )


@pytest.fixture
def mock_tools_registry():
    """Tools registry that doesn't require approval."""
    from duo_workflow_service.components.tools_registry import ToolsRegistry

    registry = MagicMock(spec=ToolsRegistry)
    registry.approval_required.return_value = False
    return registry


@pytest.fixture
def initial_state():
    """Factory for creating initial workflow state.

    The agent_name parameter controls the conversation history key.
    """

    def _create_state(goal: str, agent_name: str = "agent") -> ChatWorkflowState:
        from langchain_core.messages import HumanMessage

        from duo_workflow_service.entities.state import (
            ChatWorkflowState,
            WorkflowStatusEnum,
        )

        return ChatWorkflowState(
            plan={"steps": []},
            status=WorkflowStatusEnum.EXECUTION,
            conversation_history={agent_name: [HumanMessage(content=goal)]},
            ui_chat_log=[],
            last_human_input=None,
            goal=goal,
            project=None,
            namespace=None,
            approval=None,
            preapproved_tools=[],
        )

    return _create_state


# ===== Mock Orbit MCP tools =====
# The real Orbit MCP tools (`orbit_list_commands`, `orbit_invoke_command`) are
# injected at runtime by Rails/Workhorse — no Python tool classes exist in
# this repo to import. These mocks mirror the real tool names, descriptions,
# and shapes so the LLM sees a realistic surface during tests. The canned
# graph schema reflects the live Orbit graph as of 2026-05-14 — 6 domains,
# the most-traversed edges — so routing decisions are made against
# representative data.


class OrbitListCommandsInput(BaseModel):
    command_names: list[str] | None = Field(default=None)
    format: str = Field(default="llm")


class MockOrbitListCommands(BaseTool):
    """Mock of orbit_list_commands — lists available Orbit commands."""

    name: str = "orbit_list_commands"
    description: str = (
        "List Orbit Knowledge Graph commands with descriptions and input "
        "schemas. Use this before invoke_command to discover available "
        "command details."
    )
    args_schema: Type[BaseModel] = OrbitListCommandsInput

    def _run(self, **_: Any) -> str:
        return json.dumps(
            {
                "commands": [
                    {
                        "name": "query_graph",
                        "description": (
                            "Execute a graph query. Before composing a "
                            "query, call get_query_dsl for the DSL and "
                            "get_graph_schema for the node and edge names."
                        ),
                    },
                    {
                        "name": "get_graph_schema",
                        "description": (
                            "Return the graph schema. Use expand_nodes for "
                            "node types to include properties and "
                            "relationships."
                        ),
                    },
                    {
                        "name": "get_query_dsl",
                        "description": "Return the query_graph JSON DSL grammar.",
                    },
                    {
                        "name": "get_response_format",
                        "description": (
                            "Return the JSON Schema for query_graph responses."
                        ),
                    },
                ]
            }
        )

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


class OrbitInvokeCommandInput(BaseModel):
    command_name: str
    parameters: dict[str, Any] | None = Field(default=None)


_ORBIT_COMMAND_RESPONSES: dict[str, dict[str, Any]] = {
    "get_graph_schema": {
        "domains": [
            {
                "name": "ci",
                "nodes": [
                    "Deployment",
                    "Environment",
                    "Job",
                    "JobMetadata",
                    "Pipeline",
                    "Runner",
                    "Stage",
                ],
            },
            {
                "name": "code_review",
                "nodes": [
                    "MergeRequest",
                    "MergeRequestDiff",
                    "MergeRequestDiffFile",
                ],
            },
            {"name": "core", "nodes": ["Group", "Note", "Project", "User"]},
            {"name": "plan", "nodes": ["Label", "Milestone", "WorkItem"]},
            {
                "name": "security",
                "nodes": [
                    "Finding",
                    "SecurityScan",
                    "Vulnerability",
                    "VulnerabilityIdentifier",
                    "VulnerabilityOccurrence",
                    "VulnerabilityScanner",
                ],
            },
            {
                "name": "source_code",
                "nodes": [
                    "Branch",
                    "Definition",
                    "Directory",
                    "File",
                    "ImportedSymbol",
                ],
            },
        ],
        "edges": [
            {"name": "APPROVED", "from": ["User"], "to": ["MergeRequest"]},
            {
                "name": "ASSIGNED",
                "from": ["User"],
                "to": ["MergeRequest", "WorkItem"],
            },
            {
                "name": "AUTHORED",
                "from": ["User"],
                "to": ["MergeRequest", "Note", "Vulnerability", "WorkItem"],
            },
            {
                "name": "CALLS",
                "from": ["Definition", "File"],
                "to": ["Definition", "ImportedSymbol"],
            },
            {
                "name": "CLOSED",
                "from": ["User"],
                "to": ["MergeRequest", "WorkItem"],
            },
            {"name": "CLOSES", "from": ["MergeRequest"], "to": ["WorkItem"]},
            {
                "name": "CONTAINS",
                "from": ["Branch", "Directory", "Group", "Project", "WorkItem"],
                "to": ["Branch", "Directory", "File", "Group", "Project", "WorkItem"],
            },
            {
                "name": "DEFINES",
                "from": ["File", "Definition"],
                "to": ["Definition"],
            },
            {"name": "DEPLOYED_TO", "from": ["MergeRequest"], "to": ["Deployment"]},
            {"name": "EXTENDS", "from": ["Definition"], "to": ["Definition"]},
            {"name": "FIXES", "from": ["MergeRequest"], "to": ["Vulnerability"]},
            {"name": "HAS_DIFF", "from": ["MergeRequest"], "to": ["MergeRequestDiff"]},
            {
                "name": "HAS_FILE",
                "from": ["MergeRequestDiff"],
                "to": ["MergeRequestDiffFile"],
            },
            {"name": "HAS_JOB", "from": ["Pipeline", "Stage"], "to": ["Job"]},
            {
                "name": "HAS_LABEL",
                "from": ["MergeRequest", "WorkItem"],
                "to": ["Label"],
            },
            {
                "name": "HAS_NOTE",
                "from": ["MergeRequest", "Vulnerability", "WorkItem"],
                "to": ["Note"],
            },
            {"name": "HAS_STAGE", "from": ["Pipeline"], "to": ["Stage"]},
            {
                "name": "IMPORTS",
                "from": ["File", "ImportedSymbol"],
                "to": ["Definition", "ImportedSymbol"],
            },
            {
                "name": "IN_GROUP",
                "from": ["Label", "Milestone", "WorkItem"],
                "to": ["Group"],
            },
            {
                "name": "IN_MILESTONE",
                "from": ["MergeRequest", "WorkItem"],
                "to": ["Milestone"],
            },
            {
                "name": "IN_PROJECT",
                "from": [
                    "Branch",
                    "Deployment",
                    "Environment",
                    "Job",
                    "Label",
                    "MergeRequest",
                    "Milestone",
                    "Pipeline",
                    "Vulnerability",
                    "WorkItem",
                ],
                "to": ["Project"],
            },
            {"name": "MEMBER_OF", "from": ["User"], "to": ["Group", "Project"]},
            {"name": "MERGED", "from": ["User"], "to": ["MergeRequest"]},
            {"name": "RELATED_TO", "from": ["WorkItem"], "to": ["WorkItem"]},
            {"name": "REVIEWER", "from": ["User"], "to": ["MergeRequest"]},
            {
                "name": "TRIGGERED",
                "from": ["MergeRequest", "User"],
                "to": ["Job", "Pipeline"],
            },
        ],
    },
    "get_query_dsl": {
        "version": "0.1",
        "grammar": {
            "type": "object",
            "properties": {
                "node_type": {"type": "string"},
                "filters": {"type": "object"},
                "expand": {"type": "array"},
                "node_ids": {"type": "array"},
            },
        },
    },
    "get_response_format": {
        "version": "0.1",
        "schema": {
            "type": "object",
            "properties": {
                "results": {"type": "array"},
                "page_info": {"type": "object"},
            },
        },
    },
    "query_graph": {
        "results": [
            {
                "id": "gid://gitlab/MergeRequest/1",
                "title": "Refactor GLQL frontend renderer",
            },
            {
                "id": "gid://gitlab/MergeRequest/2",
                "title": "Fix GLQL embedded view bug",
            },
        ]
    },
}


class MockOrbitInvokeCommand(BaseTool):
    """Mock of orbit_invoke_command — runs a named Orbit command."""

    name: str = "orbit_invoke_command"
    description: str = (
        "Execute an Orbit command. This is a wrapper tool: keep only "
        "command_name and parameters at the top level, and put downstream "
        "command inputs inside parameters."
    )
    args_schema: Type[BaseModel] = OrbitInvokeCommandInput

    def _run(self, command_name: str, parameters: dict | None = None) -> str:
        del parameters  # unused — mock ignores command-specific inputs
        response = _ORBIT_COMMAND_RESPONSES.get(
            command_name,
            {"error": f"Unknown orbit command: {command_name}"},
        )
        return json.dumps(response)

    async def _arun(self, command_name: str, parameters: dict | None = None) -> str:
        return self._run(command_name, parameters)


@pytest.fixture
def orbit_list_commands_tool():
    """Mock orbit_list_commands tool — reusable across agent test suites."""
    return MockOrbitListCommands()


@pytest.fixture
def orbit_invoke_command_tool():
    """Mock orbit_invoke_command tool — reusable across agent test suites."""
    return MockOrbitInvokeCommand()
