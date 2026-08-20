"""Test suite for the v2 delegate_task tool schema (DelegateTask, build_delegate_task_model)."""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from duo_workflow_service.agent_platform.v1.components.supervisor_v2.delegate_task import (
    DelegateTask,
    SubagentDescriptor,
    build_delegate_task_model,
)


class TestSubagentDescriptor:
    """Tests for the SubagentDescriptor TypedDict."""

    def test_holds_name_and_description(self):
        descriptor = SubagentDescriptor(name="developer", description="Writes code.")
        assert descriptor["name"] == "developer"
        assert descriptor["description"] == "Writes code."


class TestBuildDelegateTaskModel:
    """Tests for build_delegate_task_model."""

    def test_subagent_name_enum_matches_configured_names(self, subagent_descriptors):
        model = build_delegate_task_model(subagent_descriptors)
        field = model.model_fields["subagent_name"]
        enum_values = {member.value for member in field.annotation}
        assert enum_values == {"developer", "tester"}

    def test_valid_subagent_name_accepted(self, subagent_descriptors, developer_name):
        model = build_delegate_task_model(subagent_descriptors)
        instance = model(
            subagent_name=developer_name,
            description="Implement the feature",
            prompt="Implement the feature",
        )
        assert str(instance.subagent_name) == developer_name

    def test_unknown_subagent_name_rejected(self, subagent_descriptors):
        model = build_delegate_task_model(subagent_descriptors)
        with pytest.raises(ValidationError):
            model(
                subagent_name="unknown_agent",
                description="Do something",
                prompt="Do something",
            )

    def test_docstring_is_inherited_from_delegate_task(self, subagent_descriptors):
        """create_model does NOT inherit __doc__ automatically -- verify it's passed explicitly."""
        model = build_delegate_task_model(subagent_descriptors)
        assert model.__doc__ == DelegateTask.__doc__

    def test_subagent_name_field_description_lists_all_agents(
        self, subagent_descriptors, developer_name, developer_description
    ):
        model = build_delegate_task_model(subagent_descriptors)
        field_description = model.model_fields["subagent_name"].description
        assert developer_name in field_description
        assert developer_description in field_description

    def test_model_title_is_delegate_task(self, subagent_descriptors):
        model = build_delegate_task_model(subagent_descriptors)
        assert model.model_config.get("title") == "delegate_task"


class TestDelegateTask:
    """Tests for the base DelegateTask model."""

    def test_requires_subagent_name_description_and_prompt(self):
        with pytest.raises(ValidationError):
            DelegateTask()

    def test_has_no_subsession_id_parameter(self):
        """A subagent cannot be addressed after the fact, so the schema must not offer a way to try."""
        assert "subsession_id" not in DelegateTask.model_fields
        assert "subsession_id" not in str(DelegateTask.model_json_schema())

    def test_is_frozen(self):
        task = DelegateTask(
            subagent_name="developer",
            description="Implement the feature",
            prompt="Implement the feature",
        )
        with pytest.raises(ValidationError):
            task.prompt = "Something else"

    def test_from_ai_message_extracts_delegate_call(self, delegate_tool_call):
        ai_message = Mock(spec=AIMessage)
        ai_message.tool_calls = [delegate_tool_call]

        task = DelegateTask.from_ai_message(ai_message)

        assert task.subagent_name == delegate_tool_call["args"]["subagent_name"]
        assert task.prompt == delegate_tool_call["args"]["prompt"]

    def test_from_ai_message_raises_when_no_delegate_call_present(
        self, regular_tool_call
    ):
        ai_message = Mock(spec=AIMessage)
        ai_message.tool_calls = [regular_tool_call]

        with pytest.raises(ValueError, match="No delegate_task tool call found"):
            DelegateTask.from_ai_message(ai_message)
