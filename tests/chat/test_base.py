import pytest
from gitlab_cloud_connector import GitLabUnitPrimitive

from ai_gateway.chat.tools.base import BaseTool


@pytest.mark.parametrize(
    ("min_required_gl_version", "gl_version", "is_compatible"),
    [
        (
            None,
            "",
            True,
        ),
        ("17.2.0", "", False),
        ("17.2.0", "invalid-version", False),
        ("17.2.0", "17.1.0", False),
        ("17.2.0", "17.3.0", True),
    ],
)
def test_is_compatible(
    min_required_gl_version: str,
    gl_version: str,
    is_compatible: bool,
):
    tool = BaseTool(
        name="test",
        description="test",
        unit_primitive=GitLabUnitPrimitive.DUO_CHAT,
        min_required_gl_version=min_required_gl_version,
    )

    assert tool.is_compatible(gl_version) == is_compatible
