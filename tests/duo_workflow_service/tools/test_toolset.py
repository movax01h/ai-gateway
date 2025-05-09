import pytest
from langchain.tools import BaseTool
from pydantic import BaseModel

from duo_workflow_service import tools
from duo_workflow_service.tools import Toolset, ToolType, UnknownToolError


class MockBaseTool(BaseTool):
    name: str = "mock_tool"
    description: str = "A mock tool for testing"

    def _run(self, *args, **kwargs):
        return "Mock tool ran"


class MockPydanticTool(BaseModel):
    tool_title: str = "mock_pydantic_tool"


@pytest.fixture
def mock_all_tools() -> dict[str, ToolType]:
    """Create a dictionary of mock tools for testing."""
    return {
        "mock_tool": MockBaseTool(),
        "mock_tool_pre_approved": MockBaseTool(name="mock_tool_pre_approved"),
        "mock_pydantic_tool": MockPydanticTool,
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
        assert len(toolset.bindable) == 3
        assert len(toolset._executable_tools) == 2  # Only BaseTool instances
        assert ["mock_tool", "mock_tool_pre_approved"] == list(
            toolset._executable_tools.keys()
        )

    def test_mapping_interface(self, toolset):
        """Test that ToolSet implements the Mapping interface correctly."""
        # Test __getitem__
        assert isinstance(toolset["mock_tool"], MockBaseTool)
        assert toolset["mock_tool"].name == "mock_tool"

        # Test __iter__
        assert set(toolset) == {"mock_tool", "mock_tool_pre_approved"}

        # Test __len__
        assert len(toolset) == 2

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
