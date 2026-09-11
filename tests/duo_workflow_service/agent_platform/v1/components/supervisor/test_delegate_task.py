"""What the ``delegate_task`` tool tells the LLM about itself."""

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from duo_workflow_service.agent_platform.v1.components.supervisor.delegate_task import (
    DelegateTask,
    build_delegate_task_model,
)


@pytest.fixture(name="tool_schema")
def tool_schema_fixture():
    """The tool exactly as it is handed to the model."""
    return convert_to_openai_tool(
        build_delegate_task_model(
            [{"name": "developer", "description": "Writes code."}]
        )
    )["function"]


class TestToolDescription:
    def test_the_delegation_constraints_reach_the_llm(self, tool_schema):
        """The rules are only enforced by the model obeying them, so a tool description
        that never arrives silently drops them: the tool still works, and nothing fails.

        Asserted on the converted schema rather than on ``tool_description``, because
        that is the payload the provider is sent.
        """
        description = tool_schema["description"]

        assert "once per turn" in description
        assert "only** tool call in the turn" in description

    def test_the_description_is_declared_rather_than_taken_from_the_docstring(
        self, tool_schema
    ):
        """The docstring addresses the reader of the file; only ``tool_description`` is spent as prompt tokens and read
        as an instruction."""
        assert tool_schema["description"] == DelegateTask.tool_description
        assert tool_schema["description"] != DelegateTask.__doc__
