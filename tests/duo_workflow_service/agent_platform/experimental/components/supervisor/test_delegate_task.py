"""Test suite for DelegateTask model and build_delegate_task_model factory."""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from duo_workflow_service.agent_platform.experimental.components.supervisor.delegate_task import (
    DelegateTask,
    ManagedAgentConfig,
    build_delegate_task_model,
)


class TestDelegateTaskBase:
    """Tests for the base DelegateTask model."""

    def test_tool_title_is_delegate_task(self):
        """Test that tool_title class variable is 'delegate_task'."""
        assert DelegateTask.tool_title == "delegate_task"

    def test_create_with_all_fields(self):
        """Test creating DelegateTask with all fields."""
        task = DelegateTask(
            subagent_type="developer",
            subsession_id=1,
            prompt="Implement the feature",
        )
        assert task.subagent_type == "developer"
        assert task.subsession_id == 1
        assert task.prompt == "Implement the feature"

    def test_create_with_no_subsession_id(self):
        """Test creating DelegateTask with subsession_id=None (new subsession)."""
        task = DelegateTask(
            subagent_type="tester",
            prompt="Write tests for the feature",
        )
        assert task.subagent_type == "tester"
        assert task.subsession_id is None
        assert task.prompt == "Write tests for the feature"

    def test_model_config_title(self):
        """Test that model config title is 'delegate_task'."""
        assert DelegateTask.model_config["title"] == "delegate_task"

    def test_from_ai_message(self):
        """Test extracting DelegateTask from an AIMessage."""
        ai_msg = Mock(spec=AIMessage)
        ai_msg.tool_calls = [
            {
                "id": "call_123",
                "name": "delegate_task",
                "args": {
                    "subagent_type": "developer",
                    "subsession_id": None,
                    "prompt": "Do the work",
                },
            }
        ]
        task = DelegateTask.from_ai_message(ai_msg)
        assert task.subagent_type == "developer"
        assert task.subsession_id is None
        assert task.prompt == "Do the work"

    def test_from_ai_message_no_delegate_call_raises(self):
        """Test that from_ai_message raises when no delegate_task call found."""
        ai_msg = Mock(spec=AIMessage)
        ai_msg.tool_calls = [
            {
                "id": "call_456",
                "name": "read_file",
                "args": {"file_path": "test.py"},
            }
        ]
        with pytest.raises(ValueError, match="No delegate_task tool call found"):
            DelegateTask.from_ai_message(ai_msg)

    def test_from_ai_message_empty_tool_calls_raises(self):
        """Test that from_ai_message raises when tool_calls is empty."""
        ai_msg = Mock(spec=AIMessage)
        ai_msg.tool_calls = []
        with pytest.raises(ValueError, match="No delegate_task tool call found"):
            DelegateTask.from_ai_message(ai_msg)


class TestBuildDelegateTaskModel:
    """Tests for the build_delegate_task_model factory function."""

    def test_builds_model_with_valid_agents(self, managed_agents_config):
        """Test building a DelegateTask model with valid agent configs."""
        model = build_delegate_task_model(managed_agents_config)
        assert model is not None
        assert issubclass(model, DelegateTask)

    def test_model_title_is_delegate_task(self, managed_agents_config):
        """Test that the built model has title 'delegate_task'."""
        model = build_delegate_task_model(managed_agents_config)
        assert model.model_config["title"] == "delegate_task"

    def test_subagent_type_constrained_to_enum(self, managed_agents_config):
        """Test that subagent_type is constrained to the provided agent names."""
        model = build_delegate_task_model(managed_agents_config)

        task = model(
            subagent_type="developer",
            prompt="Do work",
        )
        assert str(task.subagent_type) == "developer"

    def test_invalid_subagent_type_raises_validation_error(self, managed_agents_config):
        """Test that an invalid subagent_type raises ValidationError."""
        model = build_delegate_task_model(managed_agents_config)

        with pytest.raises(ValidationError):
            model(
                subagent_type="nonexistent_agent",
                prompt="Do work",
            )

    def test_schema_contains_only_valid_agent_names(
        self, managed_agents_config, managed_agent_names
    ):
        """Test that the JSON schema only contains valid agent names."""
        model = build_delegate_task_model(managed_agents_config)

        field_annotation = model.model_fields["subagent_type"].annotation
        enum_names = {member.value for member in field_annotation}
        assert enum_names == set(managed_agent_names)

    def test_single_agent(self):
        """Test building model with a single agent."""
        config = [ManagedAgentConfig(name="solo_agent", description="Does everything.")]
        model = build_delegate_task_model(config)
        task = model(subagent_type="solo_agent", prompt="Work alone")
        assert str(task.subagent_type) == "solo_agent"

    def test_model_inherits_from_delegate_task(self, managed_agents_config):
        """Test that the built model inherits from DelegateTask."""
        model = build_delegate_task_model(managed_agents_config)
        assert issubclass(model, DelegateTask)
        assert model.tool_title == "delegate_task"

    def test_prompt_field_required(self, managed_agents_config):
        """Test that prompt field is required."""
        model = build_delegate_task_model(managed_agents_config)
        with pytest.raises(ValidationError):
            model(subagent_type="developer")

    def test_subsession_id_optional(self, managed_agents_config):
        """Test that subsession_id defaults to None."""
        model = build_delegate_task_model(managed_agents_config)
        task = model(subagent_type="developer", prompt="Work")
        assert task.subsession_id is None

    def test_subsession_id_with_value(self, managed_agents_config):
        """Test that subsession_id can be set to an integer."""
        model = build_delegate_task_model(managed_agents_config)
        task = model(subagent_type="developer", subsession_id=3, prompt="Resume work")
        assert task.subsession_id == 3

    def test_field_description_contains_agent_descriptions(self, managed_agents_config):
        """Test that the subagent_type field description embeds agent descriptions."""
        model = build_delegate_task_model(managed_agents_config)
        field_description = model.model_fields["subagent_type"].description
        assert "developer" in field_description
        assert "tester" in field_description
        # Descriptions from the config should appear too
        for cfg in managed_agents_config:
            assert cfg["description"] in field_description
