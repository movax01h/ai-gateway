import json

import pytest
from langchain_core.tools import ToolException

from duo_workflow_service.tools import RenderUiTool


def _spec(node_type="tool_approval", props=None):
    return {
        "root": "n0",
        "elements": {
            "n0": {
                "type": node_type,
                "props": (
                    props
                    if props is not None
                    else {"tool_name": "run_command", "args": {"command": "ls -la"}}
                ),
            }
        },
    }


@pytest.mark.asyncio
async def test_render_ui_echoes_a_valid_ui_spec():
    spec = _spec()
    result = await RenderUiTool()._execute(spec=spec)
    assert json.loads(result) == {"ui_spec": spec}


@pytest.mark.asyncio
async def test_render_ui_rejects_an_unknown_component():
    with pytest.raises(ToolException, match="Invalid ui_spec"):
        await RenderUiTool()._execute(spec=_spec(node_type="mystery_widget", props={}))


@pytest.mark.asyncio
async def test_render_ui_rejects_invalid_props():
    # tool_approval requires tool_name.
    with pytest.raises(ToolException, match="Invalid ui_spec"):
        await RenderUiTool()._execute(spec=_spec(props={"args": {}}))


@pytest.mark.asyncio
async def test_render_ui_rejects_a_root_not_in_elements():
    with pytest.raises(ToolException, match="root .* is not in elements"):
        await RenderUiTool()._execute(spec={"root": "missing", "elements": {}})


def test_render_ui_display_message_counts_components():
    tool = RenderUiTool()
    msg = tool.format_display_message(tool.args_schema(spec=_spec()))
    assert msg == "Rendering UI (1 component)"
