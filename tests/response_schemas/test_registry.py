"""Tests for ResponseSchemaRegistry."""

# mypy: disable-error-code="call-arg,attr-defined"
# To deal with dynamic pydantic models creating mypy errors:

from pathlib import Path
from unittest.mock import patch

import pytest
from pyfakefs.fake_filesystem import FakeFilesystem

from ai_gateway.response_schemas import ResponseSchemaRegistry
from ai_gateway.response_schemas.base import BaseAgentOutput


# Clear the cache before and after each test
@pytest.fixture(autouse=True)
def clear_schema_cache():
    """Clear cache before and after each test to ensure test isolation."""
    cache_clear = getattr(ResponseSchemaRegistry.get, "cache_clear", None)

    if cache_clear is not None:
        cache_clear()

    yield

    if cache_clear is not None:
        cache_clear()


# editorconfig-checker-disable
@pytest.fixture(name="mock_fs")
def mock_fs_fixture(fs: FakeFilesystem):
    ai_gateway_dir = Path(__file__).parent.parent.parent / "ai_gateway"
    schemas_dir = ai_gateway_dir / "response_schemas" / "definitions"

    # Schema 1: Basic schema with version 1.0.0
    fs.create_file(
        schemas_dir / "general/structured_response/base/1.0.0.json",
        contents="""{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "code_review_response_tool",
  "description": "Test schema for code reviews",
  "type": "object",
  "properties": {
    "summary": {
      "type": "string",
      "description": "Brief summary of the review"
    },
    "score": {
      "type": "integer",
      "description": "Overall score from 1-10"
    }
  },
  "required": ["summary", "score"]
}""",
    )

    # Schema 2: Version 1.0.1 (for version resolution tests)
    fs.create_file(
        schemas_dir / "general/structured_response/base/1.0.1.json",
        contents="""{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "code_review_response_tool",
  "description": "Updated schema with details field",
  "type": "object",
  "properties": {
    "summary": {
      "type": "string",
      "description": "Brief summary"
    },
    "score": {
      "type": "integer"
    },
    "details": {
      "type": "string",
      "description": "Detailed findings"
    }
  },
  "required": ["summary", "score"]
}""",
    )

    # Schema 3: Dev version (for stability testing)
    fs.create_file(
        schemas_dir / "general/structured_response/base/1.0.2-dev.json",
        contents="""{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "code_review_response_tool",
  "description": "Development version",
  "type": "object",
  "properties": {
    "summary": {
      "type": "string"
    }
  },
  "required": ["summary"]
}""",
    )

    # Schema with optional fields
    fs.create_file(
        schemas_dir / "test/optional_fields/base/1.0.0.json",
        contents="""{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "test_optional_tool",
  "type": "object",
  "properties": {
    "required_field": {
      "type": "string",
      "description": "This is required"
    },
    "optional_field": {
      "type": "string",
      "description": "This is optional"
    }
  },
  "required": ["required_field"]
}""",
    )

    # Schema with array and nested types
    fs.create_file(
        schemas_dir / "test/complex_types/base/1.0.0.json",
        contents="""{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "complex_response_tool",
  "type": "object",
  "properties": {
    "items": {
      "type": "array",
      "description": "List of items"
    },
    "metadata": {
      "type": "object",
      "description": "Metadata object"
    },
    "count": {
      "type": "number",
      "description": "Count as float"
    },
    "enabled": {
      "type": "boolean",
      "description": "Boolean flag"
    }
  },
  "required": ["items"]
}""",
    )

    yield


# editorconfig-checker-enable


@pytest.fixture(name="registry")
def registry_fixture(mock_fs):  # pylint: disable=unused-argument
    """Create ResponseSchemaRegistry instance."""
    return ResponseSchemaRegistry()


@pytest.fixture(name="mock_converter")
def mock_converter_fixture():
    """Mock json_schema_to_pydantic to isolate registry logic."""
    with patch("ai_gateway.response_schemas.registry.json_schema_to_pydantic") as mock:
        # Track call count to generate unique models
        call_count = [0]

        def create_mock_model(json_schema):
            """Create a unique mock model for each JSON schema."""
            call_count[0] += 1
            # Create a unique class for each call
            mock_model = type(
                f"MockSchema_{call_count[0]}",
                (BaseAgentOutput,),
                {
                    "tool_title": json_schema.get("title", "mock_tool"),
                    "__doc__": f"Mock schema {call_count[0]}",
                },
            )
            return mock_model

        # Use side_effect to call our function for each invocation
        mock.side_effect = create_mock_model
        yield mock


class TestResponseSchemaRegistry:
    """Test suite for ResponseSchemaRegistry class."""

    def test_get_schema_returns_pydantic_model(
        self, registry: ResponseSchemaRegistry, mock_converter
    ):
        """Test that get() returns a Pydantic model class."""
        schema_class = registry.get("general/structured_response", "^1.0.0")

        # Assert converter was called
        assert mock_converter.called

        # Verify it's a class, not an instance
        assert isinstance(schema_class, type)

        # Verify converter received a dict (JSON schema)
        json_schema = mock_converter.call_args[0][0]
        assert isinstance(json_schema, dict)
        assert json_schema["title"] == "code_review_response_tool"

    @pytest.mark.parametrize(
        ("version_constraint", "expected_version"),
        [
            ("^1.0.0", "1.0.1"),  # Should resolve to 1.0.1 (has details)
            ("=1.0.0", "1.0.0"),  # Exact 1.0.0
            ("~1.0.0", "1.0.1"),  # Should resolve to 1.0.1
            ("1.0.0", "1.0.0"),  # Exact 1.0.0
            ("1.0.1", "1.0.1"),  # Exact 1.0.1
        ],
    )
    def test_version_resolution(
        self,
        registry: ResponseSchemaRegistry,
        version_constraint: str,
        expected_version: str,
        mock_converter,
    ):
        """Test that version constraints resolve correctly."""
        registry.get("general/structured_response", version_constraint)

        # Get the JSON schema that was passed to converter
        json_schema = mock_converter.call_args[0][0]

        # Verify the correct version was selected by checking the description
        # (each version has a different description in the JSON fixture)
        expected_descriptions = {
            "1.0.0": "Test schema for code reviews",
            "1.0.1": "Updated schema with details field",
        }

        assert json_schema["description"] == expected_descriptions[expected_version]

    def test_missing_schema_raises_error(self, registry: ResponseSchemaRegistry):
        """Test error when schema doesn't exist."""
        with pytest.raises(ValueError, match="Failed to load schema"):
            registry.get("nonexistent/schema", "1.0.0")

    def test_no_compatible_version_raises_error(self, registry: ResponseSchemaRegistry):
        """Test error when no version matches constraint."""
        with pytest.raises(ValueError, match="No compatible versions found"):
            registry.get("general/structured_response", "2.0.0")

    def test_schema_caching(self, registry: ResponseSchemaRegistry):
        """Test that schemas are cached after first load."""
        schema1 = registry.get("general/structured_response", "^1.0.0")
        schema2 = registry.get("general/structured_response", "^1.0.0")

        # Should be the exact same class object (not just equal)
        assert schema1 is schema2

    def test_different_schemas_are_different_objects(
        self, registry: ResponseSchemaRegistry
    ):
        """Test that different schemas return different objects."""
        schema1 = registry.get("general/structured_response", "1.0.0")
        schema2 = registry.get("general/structured_response", "1.0.1")

        # Different versions should be different classes
        assert schema1 is not schema2

    def test_empty_schema_directory_raises_error(
        self, fs: FakeFilesystem, registry: ResponseSchemaRegistry
    ):
        """Test error when schema dir exists but has no JSON files."""
        ai_gateway_dir = Path(__file__).parent.parent.parent / "ai_gateway"
        empty_dir = (
            ai_gateway_dir / "response_schemas" / "definitions" / "empty/schema/base"
        )
        fs.create_dir(empty_dir)

        with pytest.raises(ValueError, match="No JSON files found"):
            registry.get("empty/schema", "1.0.0")

    def test_dev_version_excluded_from_caret_constraint(
        self, registry: ResponseSchemaRegistry, mock_converter
    ):
        """Test that dev versions are excluded from non-exact constraints."""
        # ^1.0.0 should NOT match 1.0.2-dev, should use 1.0.1 instead
        registry.get("general/structured_response", "^1.0.0")

        # Verify registry selected 1.0.1, not 1.0.2-dev
        json_schema = mock_converter.call_args[0][0]
        assert (
            json_schema["description"] == "Updated schema with details field"
        )  # 1.0.1

    def test_dev_version_accessible_with_exact_constraint(
        self, registry: ResponseSchemaRegistry, mock_converter
    ):
        """Test that dev versions can be accessed with exact version."""
        registry.get("general/structured_response", "1.0.2-dev")

        # Verify registry selected 1.0.2-dev
        json_schema = mock_converter.call_args[0][0]
        assert json_schema["description"] == "Development version"  # 1.0.2-dev

    @pytest.mark.parametrize(
        "malicious_path",
        [
            "../../../etc/passwd",
            "../../secret",
            "general/../../../../etc/passwd",
            "../../../../../../tmp",
        ],
    )
    def test_path_traversal_attack(
        self,
        registry: ResponseSchemaRegistry,
        malicious_path: str,
    ):
        """Test that path traversal attempts are blocked."""
        with pytest.raises(ValueError, match="path traversal detected"):
            registry.get(malicious_path, "1.0.0")
