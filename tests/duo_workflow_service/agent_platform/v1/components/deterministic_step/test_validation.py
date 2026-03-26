"""Tests for DeterministicStepComponent validation utilities."""

from unittest.mock import Mock

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from duo_workflow_service.agent_platform.v1.components.deterministic_step.validation import (
    extract_configured_params,
    select_validated_tool,
    validate_against_schema,
)
from duo_workflow_service.agent_platform.v1.state import IOKey


class MockSchema(BaseModel):
    """Mock schema for testing."""

    param1: str = Field(..., description="First parameter")
    param2: int = Field(..., description="Second parameter")
    optional_param: str = Field(default="default", description="Optional parameter")


class MockOldSchema(BaseModel):
    """Mock old schema for supersession testing."""

    old_param1: str = Field(..., description="Old first parameter")
    old_param2: str = Field(..., description="Old second parameter")


class MockTool(BaseTool):
    """Mock tool for testing."""

    name: str = "mock_tool"
    description: str = "A mock tool"
    args_schema: type[BaseModel] = MockSchema

    def _run(self):
        return "mock"


class MockOldTool(BaseTool):
    """Mock old tool for supersession testing."""

    name: str = "mock_tool"
    description: str = "A mock old tool"
    args_schema: type[BaseModel] = MockOldSchema

    def _run(self):
        return "mock_old"


class MockSupersedesToolTool(BaseTool):
    """Mock tool that supersedes another."""

    name: str = "mock_tool"
    description: str = "A mock tool that supersedes another"
    args_schema: type[BaseModel] = MockSchema
    supersedes: type[BaseTool] = MockOldTool

    def _run(self):
        return "mock_new"


class TestExtractConfiguredParams:
    """Tests for extract_configured_params function."""

    def test_extract_from_alias(self):
        """Test parameter extraction when alias is provided."""
        inputs = [
            IOKey(target="value1", alias="param1", literal=True),
            IOKey(target="value2", alias="param2", literal=True),
        ]
        params = extract_configured_params(inputs)
        assert params == {"param1", "param2"}

    def test_extract_from_subkeys(self):
        """Test parameter extraction when subkeys are provided."""
        inputs = [
            IOKey(target="context", subkeys=["nested", "param1"]),
            IOKey(target="context", subkeys=["nested", "param2"]),
        ]
        params = extract_configured_params(inputs)
        assert params == {"param1", "param2"}

    def test_mixed_extraction(self):
        """Test parameter extraction with mixed input types."""
        inputs = [
            IOKey(target="value1", alias="param1", literal=True),
            IOKey(target="context", subkeys=["nested", "param2"]),
            IOKey(target="value3", alias="param3", literal=True),
        ]
        params = extract_configured_params(inputs)
        assert params == {"param1", "param2", "param3"}


class TestValidateAgainstSchema:
    """Tests for validate_against_schema function."""

    def test_valid_all_required_params(self):
        """Test validation succeeds when all required params provided."""
        params = {"param1", "param2"}
        # Should not raise any exception
        validate_against_schema(MockSchema, params)

    def test_missing_required_param(self):
        """Test validation fails when required param is missing."""
        params = {"param1"}  # Missing param2
        with pytest.raises(ValueError) as exc_info:
            validate_against_schema(MockSchema, params)
        assert "Missing required parameters" in str(exc_info.value)
        assert "param2" in str(exc_info.value)

    def test_unknown_param(self):
        """Test validation fails when unknown param is provided."""
        params = {"param1", "param2", "unknown_param"}
        with pytest.raises(ValueError) as exc_info:
            validate_against_schema(MockSchema, params)
        assert "Unknown parameters" in str(exc_info.value)
        assert "unknown_param" in str(exc_info.value)


class TestSelectValidatedTool:
    """Tests for select_validated_tool function."""

    def test_matches_current_schema(self):
        """Test returns current tool when config matches its schema."""
        tool = MockTool()
        tool.metadata = {"test_key": "test_value"}
        params = {"param1", "param2"}

        result = select_validated_tool(tool, "mock_tool", params)

        assert result is tool
        assert isinstance(result, MockTool)

    def test_matches_superseded_schema(self):
        """Test instantiates old tool when config matches superseded schema."""
        tool = MockSupersedesToolTool()
        tool.metadata = {"test_key": "test_value"}
        params = {"old_param1", "old_param2"}

        result = select_validated_tool(tool, "mock_tool", params)

        assert result is not tool
        assert isinstance(result, MockOldTool)
        assert result.metadata == tool.metadata

    def test_matches_neither_schema(self):
        """Test raises error when config matches neither schema."""
        tool = MockTool()
        tool.metadata = {"test_key": "test_value"}
        params = {"wrong_param"}

        with pytest.raises(ValueError) as exc_info:
            select_validated_tool(tool, "mock_tool", params)

        assert "Missing required parameters" in str(exc_info.value)

    def test_tool_with_no_schema(self):
        """Test accepts tool with no schema."""
        tool = Mock(spec=BaseTool)
        tool.args_schema = None
        tool.metadata = {"test_key": "test_value"}
        params = {"any_param"}

        result = select_validated_tool(tool, "mock_tool", params)

        assert result is tool

    def test_instantiation_failure(self):
        """Test handles superseded tool instantiation failure gracefully."""

        class BrokenOldTool(BaseTool):
            name: str = "broken"
            description: str = "broken tool"
            args_schema: type[BaseModel] = MockOldSchema

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                raise RuntimeError("Cannot instantiate")

            def _run(self):
                return "broken"

        class ToolWithBrokenSupersedes(BaseTool):
            name: str = "test"
            description: str = "test tool"
            args_schema: type[BaseModel] = MockSchema
            supersedes: type[BaseTool] = BrokenOldTool

            def _run(self):
                return "test"

        tool = ToolWithBrokenSupersedes()
        tool.metadata = Mock()
        params = {"old_param1", "old_param2"}

        with pytest.raises(ValueError) as exc_info:
            select_validated_tool(tool, "test", params)

        assert "failed to instantiate superseded tool" in str(exc_info.value)

    def test_metadata_copied_to_superseded_tool(self):
        """Test that metadata is properly copied to superseded tool."""
        tool = MockSupersedesToolTool()
        mock_metadata = {"gitlab_client": "test_client", "project": "test_project"}
        tool.metadata = mock_metadata
        params = {"old_param1", "old_param2"}

        result = select_validated_tool(tool, "mock_tool", params)

        assert isinstance(result, MockOldTool)
        assert result.metadata == mock_metadata
        assert result.metadata["gitlab_client"] == mock_metadata["gitlab_client"]
        assert result.metadata["project"] == mock_metadata["project"]
