"""Tests for tool output security trust level classification."""

from duo_workflow_service.security.tool_output_security import (
    ToolTrustLevel,
    get_tool_trust_level,
)
from duo_workflow_service.tools.duo_base_tool import DuoBaseTool


class MockTrustedTool(DuoBaseTool):
    """Mock tool with TRUSTED_INTERNAL trust level."""

    name: str = "mock_trusted_tool"
    description: str = "A mock trusted tool for testing"
    trust_level: ToolTrustLevel = ToolTrustLevel.TRUSTED_INTERNAL

    async def _execute(self, *args, **kwargs):
        return "trusted output"


class MockUntrustedTool(DuoBaseTool):
    """Mock tool with UNTRUSTED_USER_CONTENT trust level."""

    name: str = "mock_untrusted_tool"
    description: str = "A mock untrusted tool for testing"
    trust_level: ToolTrustLevel = ToolTrustLevel.UNTRUSTED_USER_CONTENT

    async def _execute(self, *args, **kwargs):
        return "untrusted output"


class MockToolWithoutTrustLevel(DuoBaseTool):
    """Mock tool without explicit trust_level (should default to UNTRUSTED)."""

    name: str = "mock_tool_no_trust_level"
    description: str = "A mock tool without trust level for testing"

    async def _execute(self, *args, **kwargs):
        return "output"


class TestGetToolTrustLevel:
    """Tests for get_tool_trust_level()."""

    def test_trusted_tool_returns_trusted_internal(self):
        """Trusted tool should return TRUSTED_INTERNAL."""
        tool = MockTrustedTool()
        assert get_tool_trust_level(tool) == ToolTrustLevel.TRUSTED_INTERNAL

    def test_untrusted_tool_returns_untrusted_user_content(self):
        """Untrusted tool should return UNTRUSTED_USER_CONTENT."""
        tool = MockUntrustedTool()
        assert get_tool_trust_level(tool) == ToolTrustLevel.UNTRUSTED_USER_CONTENT

    def test_tool_without_trust_level_defaults_to_untrusted(self):
        """Tool without trust_level should default to UNTRUSTED (fail-secure)."""
        tool = MockToolWithoutTrustLevel()
        assert get_tool_trust_level(tool) == ToolTrustLevel.UNTRUSTED_USER_CONTENT
