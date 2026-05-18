"""Tests for InlineResponseSchemaRegistry."""

# mypy: disable-error-code="call-arg,attr-defined"
# To deal with dynamic pydantic models creating mypy errors.

from unittest.mock import MagicMock

import pytest

from ai_gateway.response_schemas import InlineResponseSchemaRegistry
from ai_gateway.response_schemas.base import BaseAgentOutput, BaseResponseSchemaRegistry


@pytest.fixture(name="shared_registry")
def shared_registry_fixture():
    """A mock shared registry to act as the file-based fallback."""
    return MagicMock(spec=BaseResponseSchemaRegistry)


@pytest.fixture(name="inline_registry")
def inline_registry_fixture(shared_registry):
    return InlineResponseSchemaRegistry(shared_registry)


@pytest.fixture(name="simple_schema")
def simple_schema_fixture():
    return {
        "type": "object",
        "properties": {
            "summary": {"type": "string", "description": "Brief summary"},
            "score": {"type": "integer", "minimum": 1, "maximum": 10},
        },
        "required": ["summary", "score"],
    }


class TestRegisterSchema:
    """Tests for register_schema()."""

    def test_register_schema_without_title_uses_schema_id_as_fallback(
        self, inline_registry, simple_schema
    ):
        """Schema without title uses schema_id as the tool title."""
        inline_registry.register_schema("my_component", simple_schema)
        model = inline_registry.get("my_component", "")
        assert model.tool_title == "my_component"

    def test_register_schema_with_title_keeps_schema_title(self, inline_registry):
        """Schema with explicit title keeps its own title."""
        schema = {
            "title": "explicit_tool_name",
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }
        inline_registry.register_schema("component_name", schema)
        model = inline_registry.get("component_name", "")
        assert model.tool_title == "explicit_tool_name"

    def test_register_schema_converts_to_valid_pydantic_model(
        self, inline_registry, simple_schema
    ):
        """Registered schema can be instantiated as a valid Pydantic model."""
        inline_registry.register_schema("reviewer", simple_schema)
        model_cls = inline_registry.get("reviewer", "")

        instance = model_cls(summary="Looks good", score=8)
        assert instance.summary == "Looks good"
        assert instance.score == 8

    def test_register_schema_produces_base_agent_output_subclass(
        self, inline_registry, simple_schema
    ):
        """Converted model is a subclass of BaseAgentOutput."""
        inline_registry.register_schema("reviewer", simple_schema)
        model_cls = inline_registry.get("reviewer", "")
        assert issubclass(model_cls, BaseAgentOutput)

    def test_register_invalid_schema_raises(self, inline_registry):
        """Invalid schema raises ValueError during registration."""
        invalid_schema = {"type": "string"}  # not an object
        with pytest.raises(ValueError):
            inline_registry.register_schema("bad_component", invalid_schema)

    def test_register_multiple_schemas(self, inline_registry):
        """Multiple schemas can be registered and retrieved independently."""
        schema_a = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
        }
        schema_b = {
            "type": "object",
            "properties": {"b": {"type": "integer"}},
            "required": ["b"],
        }
        inline_registry.register_schema("component_a", schema_a)
        inline_registry.register_schema("component_b", schema_b)

        model_a = inline_registry.get("component_a", "")
        model_b = inline_registry.get("component_b", "")

        assert model_a.tool_title == "component_a"
        assert model_b.tool_title == "component_b"

    def test_register_duplicate_schema_id_raises(self, inline_registry, simple_schema):
        """Registering the same schema_id twice raises ValueError."""
        inline_registry.register_schema("my_comp", simple_schema)
        with pytest.raises(ValueError, match="already registered"):
            inline_registry.register_schema("my_comp", simple_schema)


class TestGet:
    """Tests for get()."""

    def test_get_inline_schema_ignores_version(self, inline_registry, simple_schema):
        """schema_version is ignored for inline schemas."""
        inline_registry.register_schema("comp", simple_schema)
        # Different version strings all return the same pre-converted model
        model_v1 = inline_registry.get("comp", "^1.0.0")
        model_v2 = inline_registry.get("comp", "2.0.0")
        model_empty = inline_registry.get("comp", "")

        assert model_v1 is model_v2
        assert model_v1 is model_empty

    def test_get_unknown_id_delegates_to_shared_registry(
        self, inline_registry, shared_registry
    ):
        """Unknown schema_id falls through to the shared file-based registry."""
        inline_registry.get("fix_pipeline_decide_approach", "^1.0.0")
        shared_registry.get.assert_called_once_with(
            "fix_pipeline_decide_approach", "^1.0.0"
        )

    def test_get_inline_id_does_not_delegate_to_shared_registry(
        self, inline_registry, shared_registry, simple_schema
    ):
        """Inline schema IDs are served from cache, not delegated."""
        inline_registry.register_schema("my_comp", simple_schema)
        inline_registry.get("my_comp", "^1.0.0")
        shared_registry.get.assert_not_called()

    def test_get_shared_registry_error_propagates(
        self, inline_registry, shared_registry
    ):
        """Errors from the shared registry propagate unchanged."""
        shared_registry.get.side_effect = ValueError("Schema not found")
        with pytest.raises(ValueError, match="Schema not found"):
            inline_registry.get("nonexistent", "1.0.0")

    def test_get_unknown_id_with_empty_version_raises(self, inline_registry):
        """Unknown schema_id with empty version raises a descriptive ValueError."""
        with pytest.raises(
            ValueError, match="not defined in the flow's 'response_schemas' block"
        ):
            inline_registry.get("missing_schema", "")

    def test_get_unknown_id_with_empty_version_does_not_delegate_to_shared_registry(
        self, inline_registry, shared_registry
    ):
        """Unknown schema_id with empty version raises before hitting the shared registry."""
        with pytest.raises(ValueError, match="response_schema_version"):
            inline_registry.get("missing_schema", "")
        shared_registry.get.assert_not_called()
