from abc import abstractmethod

import pytest

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from duo_workflow_service.tools.utils import get_all_ops_tools, get_all_subclasses


@pytest.fixture
def test_classes():
    class TestBase(DuoBaseTool):
        name: str = "base_tool"
        description: str = "A base tool"

        @abstractmethod
        async def _execute(self, *args, **kwargs):
            pass

    class Mixin:
        pass

    class SearchBase(TestBase):
        name: str = "base_search_tool"
        description: str = "A base search tool"

        @abstractmethod
        async def _execute(self, *args, **kwargs):
            pass

    class SearchToolA(SearchBase):
        name: str = "search_tool_a"
        description: str = "Search tool A"

    class SearchToolB(SearchBase):
        name: str = "search_tool_b"
        description: str = "Search tool B"

        async def _execute(self, *args, **kwargs):
            return "ok"

    class EditTool(TestBase, Mixin):
        name: str = "edit_tool"
        description: str = "Edit Tool"

        async def _execute(self, *args, **kwargs):
            return "ok"

    return TestBase, Mixin, SearchBase, SearchToolA, SearchToolB, EditTool


@pytest.mark.parametrize(
    "base_class_name, expected_names",
    [
        ("SearchToolB", set()),
        ("TestBase", {"SearchBase", "SearchToolA", "SearchToolB", "EditTool"}),
        ("Mixin", {"EditTool"}),
    ],
)
def test_get_all_subclasses(test_classes, base_class_name, expected_names):
    TestBase, Mixin, SearchBase, SearchToolA, SearchToolB, EditTool = test_classes
    class_map = {
        "TestBase": TestBase,
        "Mixin": Mixin,
        "SearchBase": SearchBase,
        "SearchToolA": SearchToolA,
        "SearchToolB": SearchToolB,
        "EditTool": EditTool,
    }
    base_class = class_map[base_class_name]
    expected = {class_map[name] for name in expected_names}

    result = get_all_subclasses(base_class)
    assert result == expected


def test_get_all_ops_tools(test_classes):
    TestBase = test_classes[0]
    result = get_all_ops_tools(tool_class=TestBase)
    assert len(result) == 2
    assert set(tool.model_fields["name"].default for tool in result) == set(
        ["edit_tool", "search_tool_b"]
    )
