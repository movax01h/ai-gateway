# pylint: disable=too-many-lines
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import ValidationError

from duo_workflow_service.agent_platform.experimental.state import (
    FlowState,
    IOKey,
    IOKeyTemplate,
    create_nested_dict,
    get_vars_from_state,
    merge_nested_dict,
    merge_nested_dict_reducer,
)
from duo_workflow_service.entities.state import WorkflowStatusEnum


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


class TestIOKey:
    """Test IOKey class functionality."""

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

    def test_iokey_parse_key_advanced_syntax(self):
        """Test parsing key with deep nesting."""
        io_key = IOKey.parse_key(
            {"from": "context:project.config.database.host", "as": "db_host"}
        )
        assert io_key.target == "context"
        assert io_key.subkeys == ["project", "config", "database", "host"]
        assert io_key.alias == "db_host"

    def test_iokey_parse_key_advanced_syntax_empty_alias(self):
        """Test parsing key with deep nesting."""
        io_key = IOKey.parse_key(
            {"from": "context:project.config.database.host", "as": ""}
        )
        assert io_key.target == "context"
        assert io_key.subkeys == ["project", "config", "database", "host"]
        assert io_key.alias == ""

    def test_iokey_parse_key_advanced_syntax_none_alias(self):
        """Test parsing key with deep nesting."""
        io_key = IOKey.parse_key(
            {"from": "context:project.config.database.host", "as": None}
        )
        assert io_key.target == "context"
        assert io_key.subkeys == ["project", "config", "database", "host"]
        assert io_key.alias is None

    def test_iokey_parse_key_advanced_syntax_none_from(self):
        """Test parsing key with deep nesting."""
        with pytest.raises(ValidationError) as exc_info:
            IOKey.parse_key({"as": None})

        assert "_AliasedIOKeyConfig\nfrom\n  Field required [type=missing" in str(
            exc_info.value
        )

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

    @pytest.mark.parametrize(
        "target,subkeys,alias,state_data,expected_result",
        [
            # Simple target without subkeys
            (
                "status",
                None,
                None,
                {
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {},
                    "ui_chat_log": [],
                    "context": {},
                },
                {"status": WorkflowStatusEnum.PLANNING},
            ),
            # Nested value using subkeys
            (
                "context",
                ["project", "name"],
                None,
                {
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {},
                    "ui_chat_log": [],
                    "context": {
                        "project": {"name": "test-project", "version": "1.0.0"}
                    },
                },
                {"name": "test-project"},
            ),
            # Deeply nested value
            (
                "context",
                ["project", "config", "database", "host"],
                None,
                {
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {},
                    "ui_chat_log": [],
                    "context": {
                        "project": {
                            "config": {"database": {"host": "localhost", "port": 5432}}
                        }
                    },
                },
                {"host": "localhost"},
            ),
            # Empty subkeys list
            (
                "context",
                [],
                None,
                {
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {},
                    "ui_chat_log": [],
                    "context": {"key": "value"},
                },
                {"context": {"key": "value"}},
            ),
            # None subkeys
            (
                "context",
                None,
                None,
                {
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {},
                    "ui_chat_log": [],
                    "context": {"key": "value"},
                },
                {"context": {"key": "value"}},
            ),
            # Complex structure - nested object
            (
                "context",
                ["flow", "metadata", "tags"],
                None,
                {
                    "status": WorkflowStatusEnum.PLANNING,
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
                },
                {"tags": ["important", "urgent"]},
            ),
            # Target with alias
            (
                "status",
                None,
                "current_status",
                {
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {},
                    "ui_chat_log": [],
                    "context": {},
                },
                {"current_status": WorkflowStatusEnum.PLANNING},
            ),
            # Target with empty string alias
            (
                "status",
                None,
                "",
                {
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {},
                    "ui_chat_log": [],
                    "context": {},
                },
                {"status": WorkflowStatusEnum.PLANNING},
            ),
        ],
        ids=[
            "simple_target",
            "nested_value_with_subkeys",
            "deeply_nested_value",
            "empty_subkeys_list",
            "none_subkeys",
            "complex_nested_object",
            "advanced_syntax_with_alias",
            "advanced_syntax_with_empty_string_alias",
        ],
    )
    def test_iokey_template_variable_from_state(
        self, target, subkeys, alias, state_data, expected_result
    ):
        """Test reading values from state using template_variable_from_state method."""
        state: FlowState = state_data
        io_key = IOKey(target=target, subkeys=subkeys, alias=alias)
        result = io_key.template_variable_from_state(state)

        assert result == expected_result

    @pytest.mark.parametrize(
        "target,subkeys,state_data,expected_result",
        [
            # List value from state
            (
                "conversation_history",
                ["main"],
                {
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {
                        "main": [
                            SystemMessage(content="System prompt"),
                            HumanMessage(content="User input"),
                        ]
                    },
                    "ui_chat_log": [],
                    "context": {},
                },
                {
                    "main": [
                        SystemMessage(content="System prompt"),
                        HumanMessage(content="User input"),
                    ]
                },
            ),
        ],
        ids=["list_value_from_state"],
    )
    def test_iokey_template_variable_from_state_with_list_validation(
        self, target, subkeys, state_data, expected_result
    ):
        """Test reading list values from state with additional validation."""
        state: FlowState = state_data
        io_key = IOKey(target=target, subkeys=subkeys)
        result = io_key.template_variable_from_state(state)

        assert result == expected_result
        assert len(result["main"]) == 2
        assert isinstance(result["main"][0], SystemMessage)

    @pytest.mark.parametrize(
        "target,subkeys,state_data,expected_result",
        [
            # Simple target without subkeys
            (
                "status",
                None,
                {
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {},
                    "ui_chat_log": [],
                    "context": {},
                },
                WorkflowStatusEnum.PLANNING,
            ),
            # Nested value using subkeys
            (
                "context",
                ["project", "name"],
                {
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {},
                    "ui_chat_log": [],
                    "context": {
                        "project": {"name": "test-project", "version": "1.0.0"}
                    },
                },
                "test-project",
            ),
            # Deeply nested value
            (
                "context",
                ["project", "config", "database", "host"],
                {
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {},
                    "ui_chat_log": [],
                    "context": {
                        "project": {
                            "config": {"database": {"host": "localhost", "port": 5432}}
                        }
                    },
                },
                "localhost",
            ),
            # Empty subkeys list
            (
                "context",
                [],
                {
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {},
                    "ui_chat_log": [],
                    "context": {"key": "value"},
                },
                {"key": "value"},
            ),
            # None subkeys
            (
                "context",
                None,
                {
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {},
                    "ui_chat_log": [],
                    "context": {"key": "value"},
                },
                {"key": "value"},
            ),
            # Numeric value
            (
                "context",
                ["project", "config", "database", "port"],
                {
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {},
                    "ui_chat_log": [],
                    "context": {
                        "project": {
                            "config": {"database": {"host": "localhost", "port": 5432}}
                        }
                    },
                },
                5432,
            ),
            # Boolean value
            (
                "context",
                ["project", "config", "debug"],
                {
                    "status": WorkflowStatusEnum.PLANNING,
                    "conversation_history": {},
                    "ui_chat_log": [],
                    "context": {
                        "project": {
                            "config": {
                                "debug": True,
                                "features": ["feature1", "feature2"],
                            }
                        }
                    },
                },
                True,
            ),
            # Complex nested object
            (
                "context",
                ["flow", "metadata"],
                {
                    "status": WorkflowStatusEnum.PLANNING,
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
                },
                {
                    "created_at": "2023-01-01",
                    "tags": ["important", "urgent"],
                },
            ),
        ],
        ids=[
            "simple_target",
            "nested_value_with_subkeys",
            "deeply_nested_value",
            "empty_subkeys_list",
            "none_subkeys",
            "numeric_value",
            "boolean_value",
            "complex_nested_object",
        ],
    )
    def test_iokey_value_from_state(self, target, subkeys, state_data, expected_result):
        """Test reading values from state using value_from_state method."""
        state: FlowState = state_data
        io_key = IOKey(target=target, subkeys=subkeys)
        result = io_key.value_from_state(state)

        assert result == expected_result

    def test_get_vars_from_state_multiple_keys(self):
        """Test extracting variables from state using multiple IOKeys."""
        state: FlowState = {
            "status": WorkflowStatusEnum.EXECUTION,
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

        assert variables["status"] == WorkflowStatusEnum.EXECUTION
        assert variables["name"] == "test-project"
        assert variables["id"] == 123

    def test_get_vars_from_state_overlapping_keys(self):
        """Test extracting variables with overlapping key names."""
        state: FlowState = {
            "status": WorkflowStatusEnum.EXECUTION,
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
            "status": WorkflowStatusEnum.PLANNING,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {
                "special-key": {"nested_key": "value"},
                "key.with.dots": {"another": "test"},
            },
        }

        # These should work with the current implementation
        io_key1 = IOKey(target="context", subkeys=["special-key", "nested_key"])
        result1 = io_key1.template_variable_from_state(state)
        assert result1["nested_key"] == "value"

    @pytest.mark.parametrize(
        "target,subkeys,value,expected_result",
        [
            # Simple target without subkeys
            (
                "status",
                None,
                WorkflowStatusEnum.PLANNING,
                {"status": WorkflowStatusEnum.PLANNING},
            ),
            # Target with single subkey
            (
                "context",
                ["project"],
                {"name": "test-project", "version": "1.0.0"},
                {"context": {"project": {"name": "test-project", "version": "1.0.0"}}},
            ),
            # Target with multiple subkeys
            (
                "context",
                ["project", "config", "database"],
                {"host": "localhost", "port": 5432},
                {
                    "context": {
                        "project": {
                            "config": {"database": {"host": "localhost", "port": 5432}}
                        }
                    }
                },
            ),
            # Target with empty subkeys list
            (
                "context",
                [],
                {"key": "value"},
                {"context": {"key": "value"}},
            ),
            # Complex nested object value
            (
                "context",
                ["metadata"],
                {
                    "created_at": "2023-01-01",
                    "tags": ["important", "urgent"],
                    "config": {"retry": 3, "timeout": 30},
                },
                {
                    "context": {
                        "metadata": {
                            "created_at": "2023-01-01",
                            "tags": ["important", "urgent"],
                            "config": {"retry": 3, "timeout": 30},
                        }
                    }
                },
            ),
        ],
        ids=[
            "simple_target_no_subkeys",
            "single_subkey",
            "multiple_subkeys",
            "empty_subkeys_list",
            "complex_object_value",
        ],
    )
    def test_iokey_to_nested_dict(self, target, subkeys, value, expected_result):
        """Test generating nested dictionary from IOKey with various configurations."""
        io_key = IOKey(target=target, subkeys=subkeys)
        result = io_key.to_nested_dict(value)

        assert result == expected_result

    @pytest.mark.parametrize(
        "state_has_key,optional,should_raise_error,expected_value",
        [
            (True, False, False, "test_value"),  # key not missing
            (True, True, False, "test_value"),  # key not missing, optional
            (False, False, True, None),  # key missing with error
            (False, True, False, None),  # key missing optional is true
        ],
    )
    def test_iokey_optional_behavior(
        self,
        state_has_key,
        optional,
        should_raise_error,
        expected_value,
    ):
        state: FlowState = {
            "status": WorkflowStatusEnum.NOT_STARTED,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {},
        }

        if state_has_key:
            state["context"]["test_key"] = "test_value"

        io_key = IOKey(target="context", subkeys=["test_key"], optional=optional)

        if should_raise_error:
            with pytest.raises(KeyError):
                io_key.value_from_state(state)
        else:
            result = io_key.value_from_state(state)
            assert result == expected_value


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


class TestIOKeyTemplate:
    """Test IOKeyTemplate class functionality."""

    def test_iokey_template_creation_simple_target(self):
        """Test creating IOKeyTemplate with simple target."""
        template = IOKeyTemplate(target="status")
        assert template.target == "status"
        assert template.subkeys is None

    def test_iokey_template_creation_with_subkeys(self):
        """Test creating IOKeyTemplate with subkeys."""
        template = IOKeyTemplate(target="context", subkeys=["project", "name"])
        assert template.target == "context"
        assert template.subkeys == ["project", "name"]

    def test_iokey_template_creation_with_template_placeholder(self):
        """Test creating IOKeyTemplate with template placeholder in subkeys."""
        template = IOKeyTemplate(target="context", subkeys=["<name>", "config"])
        assert template.target == "context"
        assert template.subkeys == ["<name>", "config"]

    def test_iokey_template_creation_multiple_placeholders(self):
        """Test creating IOKeyTemplate with multiple template placeholders."""
        template = IOKeyTemplate(
            target="context", subkeys=["<name>", "<type>", "value"]
        )
        assert template.target == "context"
        assert template.subkeys == ["<name>", "<type>", "value"]

    def test_to_iokey_simple_replacement(self):
        """Test converting template to IOKey with simple replacement."""
        template = IOKeyTemplate(target="context", subkeys=["<name>", "config"])
        replacements = {"<name>": "component1"}

        io_key = template.to_iokey(replacements)

        assert isinstance(io_key, IOKey)
        assert io_key.target == "context"
        assert io_key.subkeys == ["component1", "config"]

    def test_to_iokey_multiple_replacements(self):
        """Test converting template to IOKey with multiple replacements."""
        template = IOKeyTemplate(
            target="context", subkeys=["<name>", "<type>", "value"]
        )
        replacements = {"<name>": "component1", "<type>": "database"}

        io_key = template.to_iokey(replacements)

        assert isinstance(io_key, IOKey)
        assert io_key.target == "context"
        assert io_key.subkeys == ["component1", "database", "value"]

    def test_to_iokey_no_replacements_needed(self):
        """Test converting template to IOKey when no replacements are needed."""
        template = IOKeyTemplate(target="context", subkeys=["project", "config"])
        replacements = {"<name>": "component1"}

        io_key = template.to_iokey(replacements)

        assert isinstance(io_key, IOKey)
        assert io_key.target == "context"
        assert io_key.subkeys == ["project", "config"]

    def test_to_iokey_partial_replacements(self):
        """Test converting template to IOKey with partial replacements."""
        template = IOKeyTemplate(
            target="context", subkeys=["<name>", "config", "<type>"]
        )
        replacements = {"<name>": "component1"}  # Missing replacement for <type>

        io_key = template.to_iokey(replacements)

        assert isinstance(io_key, IOKey)
        assert io_key.target == "context"
        assert io_key.subkeys == [
            "component1",
            "config",
            "<type>",
        ]  # <type> remains unreplaced

    def test_to_iokey_empty_replacements(self):
        """Test converting template to IOKey with empty replacements dict."""
        template = IOKeyTemplate(target="context", subkeys=["<name>", "config"])
        replacements = {}

        io_key = template.to_iokey(replacements)

        assert isinstance(io_key, IOKey)
        assert io_key.target == "context"
        assert io_key.subkeys == ["<name>", "config"]  # No replacements made

    def test_to_iokey_none_subkeys(self):
        """Test converting template to IOKey when subkeys is None."""
        template = IOKeyTemplate(target="status")
        replacements = {"<name>": "component1"}

        io_key = template.to_iokey(replacements)

        assert isinstance(io_key, IOKey)
        assert io_key.target == "status"
        assert io_key.subkeys is None

    def test_to_iokey_extra_replacements(self):
        """Test converting template to IOKey with extra replacements that aren't used."""
        template = IOKeyTemplate(target="context", subkeys=["<name>", "config"])
        replacements = {"<name>": "component1", "<unused>": "value", "<extra>": "data"}

        io_key = template.to_iokey(replacements)

        assert isinstance(io_key, IOKey)
        assert io_key.target == "context"
        assert io_key.subkeys == ["component1", "config"]


class TestIOKeyLiteralField:
    """Test IOKey literal field functionality."""

    def test_iokey_literal_basic_usage(self):
        """Test basic literal field usage."""
        io_key = IOKey(target="hello_world", literal=True, alias="greeting")

        assert io_key.target == "hello_world"
        assert io_key.literal is True
        assert io_key.alias == "greeting"
        assert io_key.subkeys is None

    def test_iokey_literal_requires_alias(self):
        """Test that literal=True requires alias to be set."""
        with pytest.raises(ValueError) as exc_info:
            IOKey(target="hello_world", literal=True)

        assert "Field 'as' is required when using 'literal: true'" in str(
            exc_info.value
        )

    def test_iokey_literal_requires_non_empty_alias(self):
        """Test that literal=True requires non-empty alias."""
        with pytest.raises(ValueError) as exc_info:
            IOKey(target="hello_world", literal=True, alias="")

        assert "Field 'as' is required when using 'literal: true'" in str(
            exc_info.value
        )

    def test_iokey_literal_bypasses_target_validation(self):
        """Test that literal=True bypasses normal target validation."""
        invalid_targets = [
            "custom_value",
            "any_string",
            "123",
            "special@chars",
            "context:goal.blah.something",
            "not_a_flow_state_key",
        ]

        for target in invalid_targets:
            io_key = IOKey(target=target, literal=True, alias="test_alias")
            assert io_key.target == target
            assert io_key.literal is True
            assert io_key.alias == "test_alias"

    def test_iokey_literal_template_variable_from_state(self):
        """Test template_variable_from_state with literal=True returns literal value."""
        state: FlowState = {
            "status": WorkflowStatusEnum.PLANNING,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {},
        }

        io_key = IOKey(target="custom_literal_value", literal=True, alias="my_var")
        result = io_key.template_variable_from_state(state)

        assert result == {"my_var": "custom_literal_value"}

    def test_iokey_literal_with_complex_string_values(self):
        """Test literal field with various complex string values."""
        complex_values = [
            "http://example.com/api/v1",
            '{"json": "object", "nested": {"key": "value"}}',
            "Multi-line\nstring\nwith\nbreaks",
            "String with spaces and special chars !@#$%^&*()",
            "",
            "unicode_test_🚀_🎉_مرحبا",
        ]

        for value in complex_values:
            io_key = IOKey(target=value, literal=True, alias="test_value")
            state: FlowState = {
                "status": WorkflowStatusEnum.PLANNING,
                "conversation_history": {},
                "ui_chat_log": [],
                "context": {},
            }
            result = io_key.template_variable_from_state(state)
            assert result == {"test_value": value}

    def test_iokey_parse_key_with_literal_dict_format(self):
        """Test parsing keys with literal field using dict format."""
        key_config = {"from": "context:project.name", "as": "my_alias", "literal": True}
        io_key = IOKey.parse_key(key_config)

        assert io_key.target == "context:project.name"
        assert io_key.literal is True
        assert io_key.alias == "my_alias"
        assert io_key.subkeys is None

    def test_iokey_parse_keys_mixed_literal_and_regular(self):
        """Test parsing mixed literal and regular keys."""
        keys = [
            "context:project.name",
            {"from": "literal_value", "as": "my_literal", "literal": True},
            "status",
        ]

        io_keys = IOKey.parse_keys(keys)

        assert len(io_keys) == 3

        assert io_keys[0].target == "context"
        assert io_keys[0].subkeys == ["project", "name"]
        assert io_keys[0].literal is False
        assert io_keys[0].alias is None

        assert io_keys[1].target == "literal_value"
        assert io_keys[1].literal is True
        assert io_keys[1].alias == "my_literal"
        assert io_keys[1].subkeys is None

        assert io_keys[2].target == "status"
        assert io_keys[2].subkeys is None
        assert io_keys[2].literal is False
        assert io_keys[2].alias is None

    @pytest.mark.parametrize(
        "target_value,alias_value,expected_result",
        [
            ("hello", "greeting", {"greeting": "hello"}),
            ("123", "number", {"number": "123"}),
            ("special@chars#$%", "special", {"special": "special@chars#$%"}),
            ("path/to/file.txt", "filepath", {"filepath": "path/to/file.txt"}),
            (
                "https://api.example.com/v1",
                "api_url",
                {"api_url": "https://api.example.com/v1"},
            ),
            ("SELECT * FROM table", "query", {"query": "SELECT * FROM table"}),
            ('{"key": "value"}', "json_data", {"json_data": '{"key": "value"}'}),
            ("Hello World Test", "phrase", {"phrase": "Hello World Test"}),
            (" ", "space", {"space": " "}),
            ("  whitespace  ", "ws", {"ws": "  whitespace  "}),
        ],
    )
    def test_iokey_literal_parameterized_values(
        self, target_value, alias_value, expected_result
    ):
        """Test literal field with various target and alias combinations."""
        io_key = IOKey(target=target_value, literal=True, alias=alias_value)

        state: FlowState = {
            "status": WorkflowStatusEnum.PLANNING,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {"some": "data"},  # This should be ignored for literal keys
        }

        result = io_key.template_variable_from_state(state)
        assert result == expected_result

    @pytest.mark.parametrize(
        "complex_literal_value,alias",
        [
            ("SELECT id, name FROM users WHERE active = true", "sql_query"),
            ("key1=value1;key2=value2;key3=value3", "config_string"),
            ("/path/to/config/file.json", "config_path"),
            ("--verbose --output /tmp/result.txt --format json", "cli_args"),
            ("dGVzdCBzdHJpbmcgZm9yIGVuY29kaW5n", "encoded_data"),
            ("550e8400-e29b-41d4-a716-446655440000", "uuid"),
            ("v1.2.3-alpha.1+build.456", "version"),
        ],
        ids=[
            "sql_query",
            "config_string",
            "file_path",
            "cli_args",
            "base64_like",
            "uuid_like",
            "version_string",
        ],
    )
    def test_iokey_literal_complex_real_world_values(
        self, complex_literal_value, alias
    ):
        """Test literal field with complex real-world string values."""
        io_key = IOKey(target=complex_literal_value, literal=True, alias=alias)

        state: FlowState = {
            "status": WorkflowStatusEnum.PLANNING,
            "conversation_history": {},
            "ui_chat_log": [],
            "context": {},
        }

        result = io_key.template_variable_from_state(state)
        assert result == {alias: complex_literal_value}


class TestIntegration:
    """Integration tests for the state module."""

    def test_complete_workflow_state_manipulation(self):
        """Test a complete workflow of state manipulation."""
        # Create initial state
        initial_state: FlowState = {
            "status": WorkflowStatusEnum.NOT_STARTED,
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
        updated_state["status"] = WorkflowStatusEnum.PLANNING

        # Extract variables
        variables = get_vars_from_state(io_keys, updated_state)

        # Verify results
        assert variables["step"] == 1
        assert variables["name"] == "initialization"
        assert updated_state["status"] == WorkflowStatusEnum.PLANNING
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
