"""Shared fixtures for agent tests.

These fixtures provide common setup for testing DWS agents with real LLMs.
The execution and validation models are configurable via CLI options.

To run agent tests in parallel for speed:
    pytest agent_tests/ -n auto

Requires ANTHROPIC_API_KEY environment variable.
"""

# pylint: disable=redefined-outer-name,import-outside-toplevel,super-init-not-called

from __future__ import annotations

from typing import TYPE_CHECKING, Any
from unittest.mock import MagicMock

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage

if TYPE_CHECKING:
    from duo_workflow_service.entities.state import ChatWorkflowState


def pytest_addoption(parser):
    """Add custom CLI options for model selection."""
    parser.addoption(
        "--execution-model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model for agent execution (default: claude-haiku-4-5-20251001)",
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
        temperature=0.0,
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
