from typing import Any, Dict, List

from langchain_core.messages import SystemMessage, convert_to_openai_messages
from langsmith import Client as LangSmithClient
from langsmith.schemas import ExampleCreate
from structlog import get_logger

from eval.routing.common import memory
from eval.routing.schema import (
    DEFAULT_LS_DATASET_INPUTS_SCHEMA,
    DEFAULT_LS_DATASET_OUTPUTS_SCHEMA,
    ToolRoutingEvaluation,
)

logger = get_logger("eval.routing.dataset")


def create_ls_dataset(ls_client: LangSmithClient, dataset_name: str):
    ls_client.create_dataset(
        dataset_name=dataset_name,
        inputs_schema=DEFAULT_LS_DATASET_INPUTS_SCHEMA,
        outputs_schema=DEFAULT_LS_DATASET_OUTPUTS_SCHEMA,
    )


def create_ls_examples(
    tool_specs: List[Dict[str, Any]],
    eval_samples: List[ToolRoutingEvaluation],
) -> List[ExampleCreate]:
    logger.info("Generate examples from tool specs...")
    examples = []
    default_system_message = SystemMessage(content="You are a helpful assistant!")
    for sample in eval_samples:
        for case in sample.cases:
            messages = (
                case.messages
                if isinstance(case.messages[0], SystemMessage)
                else [default_system_message] + case.messages
            )
            example = ExampleCreate(
                inputs={
                    "messages": convert_to_openai_messages(messages=messages),
                    "tools": tool_specs,
                },
                outputs={
                    "tool": {
                        "name": sample.tool_name,
                        "args": case.expected_tool_input,
                    },
                },
                metadata={
                    "tool_name": sample.tool_name,
                },
            )
            examples.append(example)
    return examples


def tag_new_version(
    ls_client: LangSmithClient, tag_prefix: str, example_id: str, dataset_name: str
) -> str:
    example = ls_client.read_example(example_id=example_id)
    dataset_tag = f"{tag_prefix}-{example.created_at.strftime('%Y-%m-%dT%H:%M:%S%z')}"
    logger.info(f"Tag the newly created dataset with tag: {dataset_tag}")
    ls_client.update_dataset_tag(
        dataset_name=dataset_name,
        as_of=example.created_at,
        tag=dataset_tag,
    )
    return dataset_tag


@memory.cache(ignore=["ls_client"])
def generate_dataset(
    ls_client: LangSmithClient,
    tool_specs: List[Dict[str, Any]],
    eval_samples: List[ToolRoutingEvaluation],
    dataset_name: str,
    tag_prefix: str,
) -> str:
    if ls_client.has_dataset(dataset_name=dataset_name):
        logger.info(
            f"Dataset: {dataset_name} already exists, removing current examples."
        )
        old_examples = ls_client.list_examples(dataset_name=dataset_name)
        example_ids = [example.id for example in old_examples]
        if example_ids:
            ls_client.delete_examples(example_ids=example_ids)
    else:
        logger.info(f"Dataset: {dataset_name} not found, creating new dataset...")
        create_ls_dataset(ls_client=ls_client, dataset_name=dataset_name)

    logger.info("Adding dataset examples...")
    new_examples = create_ls_examples(tool_specs=tool_specs, eval_samples=eval_samples)
    response = ls_client.create_examples(
        dataset_name=dataset_name, examples=new_examples
    )
    logger.info(f"Uploaded dataset with {response.get('count')} examples.")

    example_ids = response.get("example_ids", [])
    if not example_ids:
        raise ValueError("No examples were created")

    dataset_tag = tag_new_version(
        ls_client=ls_client,
        tag_prefix=tag_prefix,
        example_id=example_ids[-1],
        dataset_name=dataset_name,
    )
    return dataset_tag
