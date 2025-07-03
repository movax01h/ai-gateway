import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    FlowStatusEnum,
    IOKey,
    create_nested_dict,
    get_vars_from_state,
    merge_nested_dict,
    merge_nested_dict_reducer,
)


class TestMergeNestedDict:
    """Test merge_nested_dict function."""

    def test_merge_nested_dict_simple(self):
        """Test merging simple dictionaries."""
        existing = {"key1": "value1", "key2": "value2"}
        new = {"key2": "new_value2", "key3": "value3"}

        result = merge_nested_dict(existing, new)

        assert result == {"key1": "value1", "key2": "new_value2", "key3": "value3"}

    def test_merge_nested_dict_nested(self):
        """Test merging nested dictionaries."""
        existing = {
            "level1": {"level2": {"key1": "value1", "key2": "value2"}, "other": "value"}
        }
        new = {"level1": {"level2": {"key2": "new_value2", "key3": "value3"}}}

        result = merge_nested_dict(existing, new)

        expected = {
            "level1": {
                "level2": {"key1": "value1", "key2": "new_value2", "key3": "value3"},
                "other": "value",
            }
        }
        assert result == expected

    def test_merge_nested_dict_non_dict_existing(self):
        """Test merging when existing is not a dict."""
        existing = "not_a_dict"
        new = {"key": "value"}

        result = merge_nested_dict(existing, new)
        assert result == {"key": "value"}

    def test_merge_nested_dict_non_dict_new(self):
        """Test merging when new is not a dict."""
        existing = {"key1": "value1"}
        new = "not_a_dict"

        result = merge_nested_dict(existing, new)
        assert result == "not_a_dict"

    def test_merge_nested_dict_empty_dicts(self):
        """Test merging empty dictionaries."""
        existing = {}
        new = {}

        result = merge_nested_dict(existing, new)
        assert len(result) == 0

    def test_merge_nested_dict_none_values(self):
        """Test merging with None values."""
        existing = None
        new = {"key": "value"}

        result = merge_nested_dict(existing, new)
        assert result == {"key": "value"}

    def test_merge_nested_dict_immutability(self):
        """Test that original dictionaries are not modified."""
        existing = {"key1": {"nested": "value1"}}
        new = {"key1": {"nested": "value2"}}

        result = merge_nested_dict(existing, new)

        # Original dictionaries should remain unchanged
        existing["other"] = "value"
        new["other"] = "value"
        assert result != existing
        assert result != new


class TestMergeNestedDictReducer:
    """Test merge_nested_dict_reducer function."""

    def test_merge_nested_dict_reducer_normal(self):
        """Test reducer with normal dictionaries."""
        left = {"key1": "value1"}
        right = {"key2": "value2"}

        result = merge_nested_dict_reducer(left, right)
        assert result == {"key1": "value1", "key2": "value2"}

    def test_merge_nested_dict_reducer_none_left(self):
        """Test reducer with None left value."""
        left = None
        right = {"key": "value"}

        result = merge_nested_dict_reducer(left, right)
        assert result == {"key": "value"}

    def test_merge_nested_dict_reducer_none_right(self):
        """Test reducer with None right value."""
        left = {"key": "value"}
        right = None

        result = merge_nested_dict_reducer(left, right)
        assert result == {"key": "value"}

    def test_merge_nested_dict_reducer_both_none(self):
        """Test reducer with both None values."""
        left = None
        right = None

        result = merge_nested_dict_reducer(left, right)
        assert len(result) == 0


class TestCreateNestedDict:
    """Test create_nested_dict function."""

    def test_create_nested_dict_single_key(self):
        """Test creating nested dict with single key."""
        keys = ["key1"]
        value = "test_value"

        result = create_nested_dict(keys, value)
        assert result == {"key1": "test_value"}

    def test_create_nested_dict_multiple_keys(self):
        """Test creating nested dict with multiple keys."""
        keys = ["level1", "level2", "level3"]
        value = "deep_value"

        result = create_nested_dict(keys, value)
        expected = {"level1": {"level2": {"level3": "deep_value"}}}
        assert result == expected

    def test_create_nested_dict_empty_keys(self):
        """Test creating nested dict with empty keys."""
        keys = []
        value = "test_value"

        result = create_nested_dict(keys, value)
        assert len(result) == 0

    def test_create_nested_dict_complex_value(self):
        """Test creating nested dict with complex value."""
        keys = ["outer", "inner"]
        value = {"nested": "object", "list": [1, 2, 3]}

        result = create_nested_dict(keys, value)
        expected = {"outer": {"inner": {"nested": "object", "list": [1, 2, 3]}}}
        assert result == expected

    def test_create_nested_dict_single_level(self):
        """Test creating nested dict with just one level."""
        keys = ["root"]
        value = [1, 2, 3, 4]

        result = create_nested_dict(keys, value)
        assert result == {"root": [1, 2, 3, 4]}


class TestIOKey:
    """Test IOKey class functionality."""

    def test_iokey_creation_simple_target(self):
        """Test creating IOKey with simple target."""
        io_key = IOKey(target="status")
        assert io_key.target == "status"
        assert io_key.subkeys is None

    def test_iokey_creation_with_subkeys(self):
        """Test creating IOKey with subkeys."""
        io_key = IOKey(target="context", subkeys=["project", "name"])
        assert io_key.target == "context"
        assert io_key.subkeys == ["project", "name"]

    def test_iokey_creation_empty_subkeys(self):
        """Test creating IOKey with empty subkeys list."""
        io_key = IOKey(target="context", subkeys=[])
        assert io_key.target == "context"
        assert len(io_key.subkeys) == 0

    def test_iokey_parse_key_simple(self):
        """Test parsing simple key without subkeys."""
        io_key = IOKey.parse_key("status")
        assert io_key.target == "status"
        assert io_key.subkeys is None

    def test_iokey_parse_key_with_subkeys(self):
        """Test parsing key with subkeys."""
        io_key = IOKey.parse_key("context:project.name")
        assert io_key.target == "context"
        assert io_key.subkeys == ["project", "name"]

    def test_iokey_parse_key_single_subkey(self):
        """Test parsing key with single subkey."""
        io_key = IOKey.parse_key("context:user")
        assert io_key.target == "context"
        assert io_key.subkeys == ["user"]

    def test_iokey_parse_key_deep_nesting(self):
        """Test parsing key with deep nesting."""
        io_key = IOKey.parse_key("context:project.config.database.host")
        assert io_key.target == "context"
        assert io_key.subkeys == ["project", "config", "database", "host"]

    def test_iokey_parse_keys_multiple(self):
        """Test parsing multiple keys at once."""
        keys = ["status", "context:project.name", "ui_chat_log"]
        io_keys = IOKey.parse_keys(keys)

        assert len(io_keys) == 3
        assert io_keys[0].target == "status"
        assert io_keys[0].subkeys is None
        assert io_keys[1].target == "context"
        assert io_keys[1].subkeys == ["project", "name"]
        assert io_keys[2].target == "ui_chat_log"
        assert io_keys[2].subkeys is None

    def test_iokey_read_from_state_simple_target(self):
        """Test reading simple target from state."""
        state: FlowState = {
            "status": FlowStatusEnum.PLANNING,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {},
        }

        io_key = IOKey(target="status")
        result = io_key.read_from_state(state)

        assert result == {"status": FlowStatusEnum.PLANNING}

    def test_iokey_read_from_state_with_subkeys(self):
        """Test reading nested value using subkeys."""
        state: FlowState = {
            "status": FlowStatusEnum.PLANNING,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {"project": {"name": "test-project", "version": "1.0.0"}},
        }

        io_key = IOKey(target="context", subkeys=["project", "name"])
        result = io_key.read_from_state(state)

        assert result == {"name": "test-project"}

    def test_iokey_read_from_state_deep_nesting(self):
        """Test reading deeply nested value."""
        state: FlowState = {
            "status": FlowStatusEnum.PLANNING,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {
                "project": {"config": {"database": {"host": "localhost", "port": 5432}}}
            },
        }

        io_key = IOKey(
            target="context", subkeys=["project", "config", "database", "host"]
        )
        result = io_key.read_from_state(state)

        assert result == {"host": "localhost"}

    def test_iokey_read_from_state_empty_subkeys(self):
        """Test reading with empty subkeys list."""
        state: FlowState = {
            "status": FlowStatusEnum.PLANNING,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {"key": "value"},
        }

        io_key = IOKey(target="context", subkeys=[])
        result = io_key.read_from_state(state)

        assert result == {"context": {"key": "value"}}

    def test_iokey_read_from_state_none_subkeys(self):
        """Test reading with None subkeys."""
        state: FlowState = {
            "status": FlowStatusEnum.PLANNING,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {"key": "value"},
        }

        io_key = IOKey(target="context", subkeys=None)
        result = io_key.read_from_state(state)

        assert result == {"context": {"key": "value"}}

    def test_iokey_read_from_state_list_value(self):
        """Test reading list value from state."""
        messages = [
            SystemMessage(content="System prompt"),
            HumanMessage(content="User input"),
        ]
        state: FlowState = {
            "status": FlowStatusEnum.PLANNING,
            "conversation_history": {"main": messages},
            "ui_chat_log": [],
            "context": {},
        }

        io_key = IOKey(target="conversation_history", subkeys=["main"])
        result = io_key.read_from_state(state)

        assert result == {"main": messages}
        assert len(result["main"]) == 2
        assert isinstance(result["main"][0], SystemMessage)

    def test_iokey_read_from_state_complex_structure(self):
        """Test reading complex nested structure."""
        state: FlowState = {
            "status": FlowStatusEnum.PLANNING,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {
                "flow": {
                    "steps": [
                        {"name": "step1", "status": "completed"},
                        {"name": "step2", "status": "in_progress"},
                    ],
                    "metadata": {
                        "created_at": "2023-01-01",
                        "tags": ["important", "urgent"],
                    },
                }
            },
        }

        # Test reading array
        io_key1 = IOKey(target="context", subkeys=["flow", "steps"])
        result1 = io_key1.read_from_state(state)
        assert len(result1["steps"]) == 2
        assert result1["steps"][0]["name"] == "step1"

        # Test reading nested object
        io_key2 = IOKey(target="context", subkeys=["flow", "metadata", "tags"])
        result2 = io_key2.read_from_state(state)
        assert result2["tags"] == ["important", "urgent"]

    def test_get_vars_from_state_multiple_keys(self):
        """Test extracting variables from state using multiple IOKeys."""
        state: FlowState = {
            "status": FlowStatusEnum.EXECUTION,
            "conversation_history": {"main": [HumanMessage(content="Hello")]},
            "ui_chat_log": [],
            "context": {"project": {"name": "test-project"}, "user": {"id": 123}},
        }

        io_keys = [
            IOKey(target="status"),
            IOKey(target="context", subkeys=["project", "name"]),
            IOKey(target="context", subkeys=["user", "id"]),
        ]

        variables = get_vars_from_state(io_keys, state)

        assert variables["status"] == FlowStatusEnum.EXECUTION
        assert variables["name"] == "test-project"
        assert variables["id"] == 123

    def test_get_vars_from_state_overlapping_keys(self):
        """Test extracting variables with overlapping key names."""
        state: FlowState = {
            "status": FlowStatusEnum.EXECUTION,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {
                "config": {"name": "config-name"},
                "project": {"name": "project-name"},
            },
        }

        io_keys = [
            IOKey(target="context", subkeys=["config", "name"]),
            IOKey(target="context", subkeys=["project", "name"]),
        ]

        variables = get_vars_from_state(io_keys, state)

        # The last key should win due to merge behavior
        assert variables["name"] == "project-name"

    def test_iokey_edge_cases(self):
        """Test edge cases for IOKey functionality."""
        # Test with special characters in keys
        state: FlowState = {
            "status": FlowStatusEnum.PLANNING,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {
                "special-key": {"nested_key": "value"},
                "key.with.dots": {"another": "test"},
            },
        }

        # These should work with the current implementation
        io_key1 = IOKey(target="context", subkeys=["special-key", "nested_key"])
        result1 = io_key1.read_from_state(state)
        assert result1["nested_key"] == "value"


class TestIOKeyModelValidations:
    """Test IOKey class model validations."""

    def test_iokey_valid_targets(self):
        """Test IOKey creation with valid targets."""
        # Test all valid targets from FlowState
        valid_targets = ["status", "conversation_history", "ui_chat_log", "context"]

        for target in valid_targets:
            io_key = IOKey(target=target)
            assert io_key.target == target
            assert io_key.subkeys is None

    def test_iokey_invalid_target_raises_validation_error(self):
        """Test IOKey creation with invalid target raises ValidationError."""
        invalid_targets = [
            "invalid_target",
            "nonexistent",
            "wrong_field",
            "status_typo",
            "context_wrong",
            "",
            "123",
            "special@chars",
        ]

        for invalid_target in invalid_targets:
            with pytest.raises(ValidationError) as exc_info:
                IOKey(target=invalid_target)

                # Check that the error message contains information about invalid target
                error_message = str(exc_info.value)
                assert "Invalid target" in error_message
                assert invalid_target in error_message
                assert "allowed targets are" in error_message

    def test_iokey_valid_targets_with_subkeys(self):
        """Test IOKey creation with valid targets that support subkeys."""
        # Only 'context' and 'conversation_history' should support subkeys based on FlowState
        valid_targets_with_subkeys = [
            ("context", ["project", "name"]),
            ("context", ["user"]),
            ("context", ["config", "debug"]),
            ("conversation_history", ["main"]),
            ("conversation_history", ["thread1", "messages"]),
        ]

        for target, subkeys in valid_targets_with_subkeys:
            io_key = IOKey(target=target, subkeys=subkeys)
            assert io_key.target == target
            assert io_key.subkeys == subkeys

    def test_iokey_targets_without_subkey_support(self):
        """Test IOKey creation with targets that don't support subkeys."""
        # 'status' and 'ui_chat_log' should not support subkeys based on FlowState annotations
        targets_without_subkey_support = ["status", "ui_chat_log"]

        for target in targets_without_subkey_support:
            # Should work without subkeys
            io_key = IOKey(target=target)
            assert io_key.target == target
            assert io_key.subkeys is None

            # Should work with None subkeys
            io_key = IOKey(target=target, subkeys=None)
            assert io_key.target == target
            assert io_key.subkeys is None

            # Should work with empty subkeys list
            io_key = IOKey(target=target, subkeys=[])
            assert io_key.target == target
            assert len(io_key.subkeys) == 0

    def test_iokey_invalid_subkeys_for_non_dict_targets(self):
        """Test IOKey creation fails when providing subkeys for non-dict targets."""
        # Note: Based on the current implementation, this test might need adjustment
        # depending on the actual type annotations in FlowState

        # Test with targets that shouldn't support subkeys
        invalid_combinations = [
            ("status", ["some", "subkey"]),
            ("ui_chat_log", ["index"]),
        ]

        for target, subkeys in invalid_combinations:
            with pytest.raises(ValidationError) as exc_info:
                IOKey(target=target, subkeys=subkeys)

                # Check that the error message contains information about non-supported subkeys
                error_message = str(exc_info.value)
                assert "does not support subkeys" in error_message

    def test_iokey_empty_string_target(self):
        """Test IOKey creation with empty string target."""
        with pytest.raises(ValidationError) as exc_info:
            IOKey(target="")

            error_message = str(exc_info.value)
            assert "Invalid target" in error_message

    def test_iokey_none_target_fails(self):
        """Test IOKey creation with None target fails."""
        with pytest.raises(ValidationError):
            IOKey(target=None)

    def test_iokey_model_validation_with_parse_key(self):
        """Test that parse_key method also triggers model validation."""
        # Valid key should work
        valid_key = "context:project.name"
        io_key = IOKey.parse_key(valid_key)
        assert io_key.target == "context"
        assert io_key.subkeys == ["project", "name"]

        # Invalid target in key should raise ValidationError
        invalid_key = "invalid_target:some.subkey"
        with pytest.raises(ValidationError) as exc_info:
            IOKey.parse_key(invalid_key)

        error_message = str(exc_info.value)
        assert "Invalid target" in error_message
        assert "invalid_target" in error_message

    def test_iokey_model_validation_with_parse_keys(self):
        """Test that parse_keys method also triggers model validation."""
        # Mix of valid and invalid keys
        keys = [
            "context:project.name",  # valid
            "status",  # valid
            "invalid_target:subkey",  # invalid
        ]

        with pytest.raises(ValidationError) as exc_info:
            IOKey.parse_keys(keys)

            error_message = str(exc_info.value)
            assert "Invalid target" in error_message
            assert "invalid_target" in error_message

    def test_iokey_case_sensitivity(self):
        """Test that target validation is case sensitive."""
        # Valid target
        IOKey(target="context")

        # Invalid case variations should fail
        invalid_case_targets = ["Context", "CONTEXT", "Status", "STATUS"]

        for invalid_target in invalid_case_targets:
            with pytest.raises(ValidationError) as exc_info:
                IOKey(target=invalid_target)

            error_message = str(exc_info.value)
            assert "Invalid target" in error_message

    def test_iokey_subkeys_type_validation(self):
        """Test subkeys type validation."""
        # Valid subkeys types
        valid_subkeys = [
            None,
            [],
            ["single"],
            ["multiple", "keys"],
            ["with", "many", "nested", "levels"],
        ]

        for subkeys in valid_subkeys:
            io_key = IOKey(target="context", subkeys=subkeys)
            assert io_key.subkeys == subkeys

    def test_iokey_immutability_after_creation(self):
        """Test that IOKey fields are immutable after creation."""
        io_key = IOKey(target="context", subkeys=["project", "name"])

        # Pydantic models are immutable by default, so this should fail
        with pytest.raises(ValidationError):
            io_key.target = "status"

        with pytest.raises(ValidationError):
            io_key.subkeys = ["different", "keys"]


class TestIntegration:
    """Integration tests for the state module."""

    def test_complete_workflow_state_manipulation(self):
        """Test a complete workflow of state manipulation."""
        # Create initial state
        initial_state: FlowState = {
            "status": FlowStatusEnum.NOT_STARTED,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {},
        }

        # Update context using merge_nested_dict
        new_context = {"flow": {"step": 1, "name": "initialization"}}
        updated_context = merge_nested_dict(initial_state["context"], new_context)

        # Create IO keys for variable extraction
        io_keys = IOKey.parse_keys(["context:flow.step", "context:flow.name"])

        # Update state
        updated_state = initial_state.copy()
        updated_state["context"] = updated_context
        updated_state["status"] = FlowStatusEnum.PLANNING

        # Extract variables
        variables = get_vars_from_state(io_keys, updated_state)

        # Verify results
        assert variables["step"] == 1
        assert variables["name"] == "initialization"
        assert updated_state["status"] == FlowStatusEnum.PLANNING
        assert updated_state["context"]["flow"]["step"] == 1

    def test_io_key_parsing_and_variable_extraction(self):
        """Test parsing IO keys and extracting variables."""
        # Complex state structure
        state = {
            "context": {
                "project": {
                    "name": "test-project",
                    "version": "1.0.0",
                    "config": {"debug": True, "features": ["feature1", "feature2"]},
                },
                "user": {"id": 123, "preferences": {"theme": "dark"}},
            },
            "conversation_history": {
                "main": [
                    SystemMessage(content="System prompt"),
                    HumanMessage(content="User input"),
                ]
            },
            "ui_chat_log": [
                {"type": "user", "message": "Hello"},
                {"type": "assistant", "message": "Hi there!"},
            ],
        }

        # Parse various IO keys
        keys = [
            "context:project.name",
            "context:project.config.debug",
            "context:user.preferences.theme",
            "conversation_history:main",
            "ui_chat_log",
        ]

        io_keys = IOKey.parse_keys(keys)
        variables = get_vars_from_state(io_keys, state)

        # Verify extracted variables
        assert variables["name"] == "test-project"
        assert variables["debug"] is True
        assert variables["theme"] == "dark"
        assert len(variables["main"]) == 2
        assert isinstance(variables["main"][0], SystemMessage)
        assert len(variables["ui_chat_log"]) == 2
