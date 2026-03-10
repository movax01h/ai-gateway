import pytest

from duo_workflow_service.tools.think import Think, ThinkInput


@pytest.fixture(name="tool")
def tool_fixture() -> Think:
    return Think()


@pytest.mark.asyncio
async def test_think_returns_ok(tool: Think):
    result = await tool._arun(thought="The bug is in the __rmul__ method")
    assert result == "ok"


@pytest.mark.asyncio
async def test_think_empty_thought(tool: Think):
    result = await tool._arun(thought="")
    assert result == "ok"


def test_think_format_display_message_short():
    tool = Think()
    args = ThinkInput(thought="Simple plan")
    assert tool.format_display_message(args) == "Simple plan"


def test_think_format_display_message_truncates():
    tool = Think()
    long_thought = "x" * 100
    args = ThinkInput(thought=long_thought)
    result = tool.format_display_message(args)
    assert result == "x" * 80 + "..."
    assert len(result) == 83
