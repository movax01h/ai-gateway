"""Tests for JSON to pydantic conversion for ResponseSchemaRegistry."""

from unittest.mock import Mock

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from ai_gateway.response_schemas.converter import (
    RECURSION_LIMIT,
    json_schema_to_pydantic,
)


@pytest.fixture(name="basic_schema")
def basic_schema_fixture():
    """Basic schema with summary and score fields - used across multiple tests."""
    return {
        "title": "test_response_tool",
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Brief summary"},
            "score": {"type": "integer", "description": "Score from 1-10"},
        },
        "required": ["summary", "score"],
    }


class TestConverter:
    """Test suits for conversion logic for Response Schema Registry."""

    def test_schema_has_tool_attributes(self, basic_schema):
        """Test that generated schema has required tool attributes."""
        schema_class = json_schema_to_pydantic(basic_schema)

        # Check tool_title attribute
        assert hasattr(schema_class, "tool_title")
        assert schema_class.tool_title == "test_response_tool"

        # Check from_ai_message classmethod
        assert hasattr(schema_class, "from_ai_message")
        assert callable(schema_class.from_ai_message)

    def test_schema_validation_enforces_required_fields(self, basic_schema):
        """Test that schema enforces required field validation."""
        schema_class = json_schema_to_pydantic(basic_schema)

        # Valid data should work
        valid_instance = schema_class(summary="Looks good", score=8)
        assert valid_instance.summary == "Looks good"
        assert valid_instance.score == 8

        # Missing required field should fail
        with pytest.raises(ValidationError):
            schema_class(summary="Missing score")

        # Missing required field should fail
        with pytest.raises(Exception):
            schema_class(score=5)

    def test_schema_handles_optional_fields(self):
        """Test that optional fields work correctly."""
        schema = {
            "title": "test_optional_tool",
            "type": "object",
            "properties": {
                "required_field": {"type": "string", "description": "This is required"},
                "optional_field": {"type": "string", "description": "This is optional"},
            },
            "required": ["required_field"],
        }

        schema_class = json_schema_to_pydantic(schema)

        # With only required field
        instance1 = schema_class(required_field="test")
        assert instance1.required_field == "test"
        assert instance1.optional_field is None

        # With both fields
        instance2 = schema_class(required_field="test", optional_field="optional")
        assert instance2.optional_field == "optional"

    def test_model_has_final_response_instructions(self, basic_schema):
        """Test that generated model has instructions in docstring."""
        schema_class = json_schema_to_pydantic(basic_schema)

        assert schema_class.__doc__ is not None
        assert "MANDATORY COMPLETION TOOL" in schema_class.__doc__
        assert "CRITICAL INSTRUCTIONS" in schema_class.__doc__

    def test_schema_handles_complex_types(self):
        """Test that complex types (array, object, number, boolean) are handled."""
        schema = {
            "title": "complex_response_tool",
            "type": "object",
            "properties": {
                "items": {"type": "array", "description": "List of items"},
                "metadata": {"type": "object", "description": "Metadata object"},
                "count": {"type": "number", "description": "Count as float"},
                "enabled": {"type": "boolean", "description": "Boolean flag"},
            },
            "required": ["items"],
        }

        schema_class = json_schema_to_pydantic(schema)

        instance = schema_class(
            items=["a", "b", "c"],
            metadata={"key": "value"},
            count=3.14,
            enabled=True,
        )

        assert instance.items == ["a", "b", "c"]
        assert instance.metadata == {"key": "value"}
        assert instance.count == 3.14
        assert instance.enabled is True

    def test_model_includes_schema_description_in_docstring(self):
        """Test that schema description is included in model docstring."""
        schema = {
            "title": "test_response_tool",
            "description": "Updated schema with details field",  # Custom description
            "type": "object",
            "properties": {"summary": {"type": "string"}, "score": {"type": "integer"}},
            "required": ["summary", "score"],
        }

        schema_class = json_schema_to_pydantic(schema)

        assert schema_class.__doc__ is not None
        assert "Updated schema with details field" in schema_class.__doc__

    def test_model_config_is_frozen(self, basic_schema):
        """Test that generated models are frozen (immutable)."""
        schema_class = json_schema_to_pydantic(basic_schema)

        instance = schema_class(summary="test", score=5)

        # Should not be able to modify after creation
        with pytest.raises(Exception):  # Pydantic ValidationError
            instance.summary = "modified"

    def test_from_ai_message_method_works(self, basic_schema):
        """Test that from_ai_message() classmethod works correctly."""
        schema_class = json_schema_to_pydantic(basic_schema)

        # Create mock AIMessage with tool call
        mock_message = Mock(spec=AIMessage)
        mock_message.tool_calls = [
            {
                "id": "call_123",
                "name": "code_review_response_tool",
                "args": {"summary": "Test summary", "score": 7},
            }
        ]

        # Use from_ai_message to create instance
        instance = schema_class.from_ai_message(mock_message)

        assert instance.summary == "Test summary"
        assert instance.score == 7

    def test_invalid_json_schema_raises_error(self):
        """Test that invalid JSON schema raises appropriate error."""
        invalid_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "invalid_tool",
            # Missing 'type' field
        }

        with pytest.raises(ValueError, match="Schema type must be 'object'"):
            json_schema_to_pydantic(invalid_schema)

    def test_schema_without_title_raises_error(self):
        """Test that schema without title raises error."""
        invalid_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            # Missing title
            "type": "object",
            "properties": {"field": {"type": "string"}},
        }

        with pytest.raises(ValueError, match="Schema must have 'title'"):
            json_schema_to_pydantic(invalid_schema)

    def test_nested_objects_with_typed_arrays(self):
        """Test nested object schemas with typed arrays."""
        schema = {
            "title": "test_nested_tool",
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Summary text",
                },
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "priority": {
                                "type": "string",
                                "enum": ["low", "high"],
                            },
                            "count": {"type": "integer", "minimum": 0},
                        },
                        "required": ["name", "priority"],
                    },
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["summary", "items"],
        }

        model = json_schema_to_pydantic(schema)

        # Valid nested structure
        instance = model(
            summary="Test summary",
            items=[
                {"name": "Item1", "priority": "high", "count": 5},
                {"name": "Item2", "priority": "low"},
            ],
            tags=["tag1", "tag2"],
        )

        data = instance.model_dump()

        assert data["summary"] == "Test summary"
        assert len(data["items"]) == 2
        assert data["items"][0]["name"] == "Item1"
        assert data["items"][0]["priority"] == "high"
        assert data["items"][0]["count"] == 5
        assert data["items"][1]["count"] is None  # Optional field
        assert data["tags"] == ["tag1", "tag2"]

        # Invalid - wrong enum value in nested object
        with pytest.raises(ValidationError) as exc_info:
            model(
                summary="Test",
                items=[{"name": "Item", "priority": "medium"}],  # Invalid enum
            )
        assert "Input should be 'low' or 'high'" in str(exc_info.value)

        # Invalid - violates minimum constraint in nested object
        with pytest.raises(ValidationError) as exc_info:
            model(
                summary="Test",
                items=[{"name": "Item", "priority": "low", "count": -1}],
            )
        assert "greater than or equal to 0" in str(exc_info.value)

    def test_to_output_method_returns_dict(self, basic_schema):
        """Test that converter adds to_output() method that returns dict."""
        schema_class = json_schema_to_pydantic(basic_schema)

        # Create an instance
        instance = schema_class(summary="Test summary", score=8)

        # Verify to_output() method exists
        assert hasattr(instance, "to_output")
        assert callable(instance.to_output)

        # Call to_output()
        output = instance.to_output()

        # Should return a dict (for writing to flow state)
        assert isinstance(output, dict)
        assert output == {"summary": "Test summary", "score": 8}

        # Should be equivalent to model_dump()
        assert output == instance.model_dump()

    @pytest.mark.parametrize(
        ("depth", "should_succeed", "test_id"),
        [
            (RECURSION_LIMIT - 1, True, "within_limit"),
            (RECURSION_LIMIT, True, "at_exact_limit"),
            (RECURSION_LIMIT + 1, False, "exceeds_limit"),
            (RECURSION_LIMIT + 10, False, "far_exceeds_limit"),
        ],
    )
    def test_schema_depth_limits(  # pylint: disable=unused-argument
        self, depth, should_succeed, test_id
    ):
        """Test that schema recursion limits prevent DoS while allowing legitimate nesting."""

        # Build nested object schema to specified depth
        schema = {
            "title": "depth_test",
            "type": "object",
            "properties": {"field0": {"type": "string"}},
        }

        current = schema["properties"]
        for i in range(1, depth + 1):
            current[f"field{i}"] = {
                "type": "object",
                "properties": {f"nested_{i}": {"type": "string"}},
            }
            current = current[f"field{i}"]["properties"]

        if should_succeed:
            model = json_schema_to_pydantic(schema)
            assert model.tool_title == "depth_test"
        else:
            with pytest.raises(ValueError) as exc_info:
                json_schema_to_pydantic(schema)
            assert "exceeds maximum depth" in str(exc_info.value)
