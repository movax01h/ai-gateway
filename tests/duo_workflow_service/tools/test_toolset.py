import pytest
from langchain.tools import BaseTool
from langchain_core.messages import ToolCall
from pydantic import BaseModel, Field

from duo_workflow_service import tools
from duo_workflow_service.tools import (
    MalformedToolCallError,
    Toolset,
    ToolType,
    UnknownToolError,
)


class MockBaseTool(BaseTool):
    name: str = "mock_tool"
    description: str = "A mock tool for testing"

    def _run(self, *args, **kwargs):
        return "Mock tool ran"


class MockPydanticTool(BaseModel):
    tool_title: str = "mock_pydantic_tool"


class MockToolWithParams(BaseTool):
    name: str = "mock_tool_with_params"
    description: str = "A mock tool with parameters for testing"

    class InputSchema(BaseModel):
        param1: str = Field(..., description="First parameter")
        param2: int = Field(..., description="Second parameter")

    def _run(self, param1: str, param2: int, **kwargs):
        return f"Mock tool ran with {param1} and {param2}"


class MockPydanticToolWithParams(BaseModel):
    name: str = Field(..., description="Name parameter")
    age: int = Field(..., description="Age parameter")


@pytest.fixture
def mock_all_tools() -> dict[str, ToolType]:
    """Create a dictionary of mock tools for testing."""
    return {
        "mock_tool": MockBaseTool(),
        "mock_tool_pre_approved": MockBaseTool(name="mock_tool_pre_approved"),
        "mock_pydantic_tool": MockPydanticTool,
        "mock_tool_with_params": MockToolWithParams(),
        "mock_pydantic_tool_with_params": MockPydanticToolWithParams,
    }


@pytest.fixture
def mock_pre_approved() -> list[str]:
    """Create a list of pre-approved tool names for testing."""
    return ["mock_tool_pre_approved"]


@pytest.fixture
def toolset(mock_all_tools, mock_pre_approved) -> Toolset:
    """Create a ToolSet instance for testing."""
    return Toolset(pre_approved=mock_pre_approved, all_tools=mock_all_tools)


class TestToolset:
    def test_initialization(self, mock_all_tools, mock_pre_approved):
        """Test that ToolSet initializes correctly."""
        toolset = Toolset(pre_approved=mock_pre_approved, all_tools=mock_all_tools)

        assert toolset._all_tools == mock_all_tools
        assert toolset._pre_approved == mock_pre_approved
        assert len(toolset.bindable) == 5
        assert len(toolset._executable_tools) == 3  # Only BaseTool instances
        assert set(toolset._executable_tools.keys()) == {
            "mock_tool",
            "mock_tool_pre_approved",
            "mock_tool_with_params",
        }

    def test_mapping_interface(self, toolset):
        """Test that ToolSet implements the Mapping interface correctly."""
        # Test __getitem__
        assert isinstance(toolset["mock_tool"], MockBaseTool)
        assert toolset["mock_tool"].name == "mock_tool"

        # Test __iter__
        assert set(toolset) == {
            "mock_tool",
            "mock_tool_pre_approved",
            "mock_tool_with_params",
        }

        # Test __len__
        assert len(toolset) == 3

        # Test with dict methods
        assert "mock_tool" in toolset
        assert "nonexistent_tool" not in toolset

        # Test KeyError when accessing nonexistent tool
        with pytest.raises(KeyError):
            toolset["nonexistent_tool"]

    def test_approved_method(self, toolset):
        """Test that the approved method works correctly."""
        # Test approved tools
        assert toolset.approved("mock_tool_pre_approved") is True

        # Test non-approved tools
        assert toolset.approved("mock_tool") is False

        # Test UnknownToolError for nonexistent tool
        with pytest.raises(UnknownToolError):
            toolset.approved("nonexistent_tool")

    def test_validate_tool_call_base_tool(self, toolset):
        """Test that validate_tool_call works correctly with BaseTool instances."""
        # Valid tool call
        tool_call = ToolCall(id="123", name="mock_tool", args={})
        validated_call = toolset.validate_tool_call(tool_call)
        assert validated_call == tool_call

        # Valid tool call with parameters
        tool_call_with_params = ToolCall(
            id="456",
            name="mock_tool_with_params",
            args={"param1": "test", "param2": 42},
        )
        validated_call = toolset.validate_tool_call(tool_call_with_params)
        assert validated_call == tool_call_with_params

    def test_validate_tool_call_pydantic_model(self, toolset):
        """Test that validate_tool_call works correctly with Pydantic models."""
        # Valid tool call with Pydantic model
        tool_call = ToolCall(
            id="789",
            name="mock_pydantic_tool_with_params",
            args={"name": "John", "age": 30},
        )
        validated_call = toolset.validate_tool_call(tool_call)
        assert validated_call == tool_call

    def test_validate_tool_call_unknown_tool(self, toolset):
        """Test that validate_tool_call raises MalformedToolCallError for unknown tools."""
        tool_call = ToolCall(id="131415", name="unknown_tool", args={})

        with pytest.raises(MalformedToolCallError) as exc_info:
            toolset.validate_tool_call(tool_call)

        assert "not found" in str(exc_info.value)
        assert exc_info.value.tool_call == tool_call

    def test_validate_tool_call_invalid_args_base_tool(self, toolset):
        """Test that validate_tool_call raises MalformedToolCallError for invalid args with BaseTool."""
        # Missing required parameter
        tool_call = ToolCall(
            id="161718",
            name="mock_tool_with_params",
            args={"param1": "test"},  # Missing param2
        )

        with pytest.raises(MalformedToolCallError) as exc_info:
            toolset.validate_tool_call(tool_call)

        assert "Invalid arguments" in str(exc_info.value)
        assert exc_info.value.tool_call == tool_call

        # Wrong type for parameter
        tool_call = ToolCall(
            id="192021",
            name="mock_tool_with_params",
            args={"param1": "test", "param2": "not_an_integer"},
        )

        with pytest.raises(MalformedToolCallError) as exc_info:
            toolset.validate_tool_call(tool_call)

        assert "Invalid arguments" in str(exc_info.value)
        assert exc_info.value.tool_call == tool_call

    def test_validate_tool_call_invalid_args_pydantic_model(self, toolset):
        """Test that validate_tool_call raises MalformedToolCallError for invalid args with Pydantic model."""
        # Missing required parameter
        tool_call = ToolCall(
            id="222324",
            name="mock_pydantic_tool_with_params",
            args={"name": "John"},  # Missing age
        )

        with pytest.raises(MalformedToolCallError) as exc_info:
            toolset.validate_tool_call(tool_call)

        assert "Invalid arguments" in str(exc_info.value)
        assert exc_info.value.tool_call == tool_call

        # Wrong type for parameter
        tool_call = ToolCall(
            id="252627",
            name="mock_pydantic_tool_with_params",
            args={"name": "John", "age": "thirty"},  # Age should be an integer
        )

        with pytest.raises(MalformedToolCallError) as exc_info:
            toolset.validate_tool_call(tool_call)

        assert "Invalid arguments" in str(exc_info.value)
        assert exc_info.value.tool_call == tool_call
