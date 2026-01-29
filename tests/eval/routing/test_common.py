from abc import abstractmethod
from unittest.mock import patch

import pytest
import yaml

from duo_workflow_service.tools.duo_base_tool import DuoBaseTool
from eval.routing.common import (
    get_all_ops_tools,
    get_all_subclasses,
    load_routing_configs,
    load_yamls,
)


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
    assert result[0]["function"]["name"] == "edit_tool"
    assert result[1]["function"]["name"] == "search_tool_b"


@pytest.mark.parametrize(
    ("contents", "file_paths"),
    [
        ([{"key": "value"}], ["test.yaml"]),
        ([{"file": "one"}, {"file": "two"}], ["file1.yml", "file2.yaml"]),
        ([{"nested": "file"}], ["nested/deep/nested.yaml"]),
    ],
)
def test_load_yamls(tmp_path, contents, file_paths):
    expected = []
    for content, file_path in zip(contents, file_paths):
        if "/" in file_path:
            base_dir = "/".join(file_path.split("/")[:-1])
            file_name = file_path.split("/")[-1]
            nested_dir = tmp_path / base_dir
            nested_dir.mkdir(parents=True)
            yaml_file = nested_dir / file_name
        else:
            yaml_file = tmp_path / file_path
        yaml_file.write_text(yaml.dump(content))

        expected.append((file_path, content))

    result = load_yamls(str(tmp_path))

    assert sorted(result) == sorted(expected)


@pytest.mark.parametrize(
    ("yaml_values", "is_valid"),
    [
        # Valid: two properly formatted routing config
        (
            [
                (
                    "tool_a.yaml",
                    {
                        "tool_name": "tool_a",
                        "cases": [
                            {
                                "messages": [
                                    {"type": "human", "content": "test message"}
                                ],
                                "expected_tool_input": {"param": "value"},
                            }
                        ],
                    },
                ),
                (
                    "tool_z.yaml",
                    {
                        "tool_name": "tool_z",
                        "cases": [
                            {
                                "messages": [
                                    {"type": "human", "content": "test message"}
                                ],
                                "expected_tool_input": {"param": "value"},
                            }
                        ],
                    },
                ),
            ],
            True,
        ),
        # Invalid: missing messages and extra field
        (
            [
                (
                    "tool_a.yaml",
                    {
                        "tool_name": "tool_a",
                        "cases": [{"expected_tool_input": {"param": "value"}}],
                    },
                ),
                (
                    "tool_z.yaml",
                    {
                        "tool_name": "tool_z",
                        "cases": [
                            {
                                "messages": [
                                    {"type": "human", "content": "test message"}
                                ],
                                "expected_tool_input": {"param": "value"},
                                "extra_field": "not allowed",
                            }
                        ],
                    },
                ),
            ],
            False,
        ),
    ],
)
@patch("eval.routing.common.load_yamls")
def test_load_routing_configs(mock_load_yamls, yaml_values, is_valid):
    mock_load_yamls.return_value = yaml_values

    if is_valid:
        configs = load_routing_configs("dummy_path")
        assert len(configs) == len(yaml_values)
        assert [c.tool_name for c in configs] == sorted(
            y["tool_name"] for _, y in yaml_values
        )
    else:
        with pytest.raises(Exception):
            load_routing_configs("dummy_path")

    mock_load_yamls.assert_called_once_with("dummy_path")
