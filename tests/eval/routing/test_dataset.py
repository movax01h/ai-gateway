import datetime as dt
import uuid
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import HumanMessage

from eval.routing.dataset import (
    ExampleCreate,
    create_ls_dataset,
    create_ls_examples,
    generate_dataset,
    tag_new_version,
)
from eval.routing.schema import (
    DEFAULT_LS_DATASET_INPUTS_SCHEMA,
    DEFAULT_LS_DATASET_OUTPUTS_SCHEMA,
    RoutingCase,
    ToolRoutingEvaluation,
)


@pytest.fixture
def ls_client():
    client = MagicMock()
    # common methods used across tests
    client.create_dataset = MagicMock()
    client.has_dataset = MagicMock()
    client.list_examples = MagicMock(
        return_value=[
            ExampleCreate(id="9636c903-9de5-452c-a05f-7b8fd315aa10"),
            ExampleCreate(id="d0b3e35a-7b91-4ff6-acc4-66661c04a773"),
        ]
    )
    client.delete_examples = MagicMock()
    client.create_examples = MagicMock(
        return_value={"count": 1, "example_ids": ["uuid3"]}
    )
    client.read_example = MagicMock(
        return_value=SimpleNamespace(
            created_at=dt.datetime(2024, 1, 2, 3, 4, 5, tzinfo=dt.timezone.utc)
        )
    )
    client.update_dataset_tag = MagicMock()
    return client


@pytest.fixture
def tool_specs():
    return [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "search description",
                "parameters": {},
            },
        },
        {
            "type": "function",
            "function": {
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory_path": {
                            "description": "The directory path to create. Must be within the current working directory tree.",
                            "type": "string",
                        }
                    },
                    "required": ["directory_path"],
                },
                "description": "Create a new directory using the mkdir command.\n    The directory creation is restricted to the current working directory tree.",
                "name": "mkdir",
            },
        },
    ]


@pytest.fixture
def eval_samples():
    return [
        ToolRoutingEvaluation(
            tool_name="mkdir",
            cases=[
                RoutingCase(
                    messages=[
                        HumanMessage(
                            content="make a new directory named ./tmp/test_dir"
                        )
                    ],
                    expected_tool_input={
                        "directory_path": "./tmp/test_dir",
                    },
                )
            ],
        )
    ]


def test_create_ls_dataset(ls_client):
    create_ls_dataset(ls_client, dataset_name="my-ds")

    ls_client.create_dataset.assert_called_once_with(
        dataset_name="my-ds",
        inputs_schema=DEFAULT_LS_DATASET_INPUTS_SCHEMA,
        outputs_schema=DEFAULT_LS_DATASET_OUTPUTS_SCHEMA,
    )


def test_create_ls_examples(tool_specs, eval_samples):
    examples = create_ls_examples(tool_specs=tool_specs, eval_samples=eval_samples)

    assert len(examples) == 1
    ex = examples[0]

    assert ex.metadata == {"tool_name": "mkdir"}

    assert ex.inputs["messages"] == [
        {
            "role": "system",
            "content": "You are a helpful assistant!",
        },
        {
            "role": "user",
            "content": "make a new directory named ./tmp/test_dir",
        },
    ]
    assert ex.inputs["tools"] == tool_specs

    assert ex.outputs == {
        "tool": {
            "name": "mkdir",
            "args": {
                "directory_path": "./tmp/test_dir",
            },
        }
    }


def test_tag_new_version(ls_client):
    tag = tag_new_version(
        ls_client=ls_client,
        tag_prefix="ds-tag",
        example_id="ex_123",
        dataset_name="my-ds",
    )

    assert tag == "ds-tag-2024-01-02T03:04:05+0000"
    ls_client.read_example.assert_called_once_with(example_id="ex_123")
    ls_client.update_dataset_tag.assert_called_once_with(
        dataset_name="my-ds",
        as_of=ls_client.read_example.return_value.created_at,
        tag=tag,
    )


@pytest.mark.parametrize("has_dataset", [True, False])
@patch("eval.routing.dataset.create_ls_examples")
@patch("eval.routing.dataset.tag_new_version")
def test_generate_dataset(
    mock_tag_new_version,
    mock_create_ls_examples,
    has_dataset,
    ls_client,
    tool_specs,
    eval_samples,
):
    dataset_name = f"my-ds-{str(uuid.uuid4())}"
    ls_client.has_dataset.return_value = has_dataset
    ls_examples = [
        ExampleCreate(
            inputs={"messages": [], "tools": []},
            outputs={},
            metadata={},
        )
    ]
    mock_create_ls_examples.return_value = ls_examples

    mock_tag_new_version.return_value = "username-tag-123"

    result_tag = generate_dataset(
        ls_client=ls_client,
        tool_specs=tool_specs,
        eval_samples=eval_samples,
        dataset_name=dataset_name,
        tag_prefix="username",
    )

    ls_client.has_dataset.assert_called_once_with(dataset_name=dataset_name)

    if has_dataset:
        ls_client.list_examples.assert_called_once_with(dataset_name=dataset_name)
        ls_client.delete_examples.assert_called_once_with(
            example_ids=[
                uuid.UUID("9636c903-9de5-452c-a05f-7b8fd315aa10"),
                uuid.UUID("d0b3e35a-7b91-4ff6-acc4-66661c04a773"),
            ]
        )
    else:
        ls_client.create_dataset.assert_called_once_with(
            dataset_name=dataset_name,
            inputs_schema=DEFAULT_LS_DATASET_INPUTS_SCHEMA,
            outputs_schema=DEFAULT_LS_DATASET_OUTPUTS_SCHEMA,
        )

    mock_create_ls_examples.assert_called_once_with(
        tool_specs=tool_specs, eval_samples=eval_samples
    )

    ls_client.create_examples.assert_called_once_with(
        dataset_name=dataset_name, examples=ls_examples
    )
    mock_tag_new_version.assert_called_once_with(
        ls_client=ls_client,
        tag_prefix="username",
        example_id="uuid3",
        dataset_name=dataset_name,
    )
    assert result_tag == "username-tag-123"
