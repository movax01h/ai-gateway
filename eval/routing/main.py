from typing import Annotated, List, Optional

import typer
from structlog import get_logger

from eval.routing.common import get_all_ops_tools, load_routing_configs, ls_client
from eval.routing.dataset import generate_dataset
from eval.routing.schema import EvalDataset
from eval.routing.validator import execute_routing, is_correct

logger = get_logger("eval.routing.run")

app = typer.Typer()


@app.command(
    help="Create a dataset from the latest tool specs and run the routing evaluation."
)
def run(
    version_prefix: Annotated[
        str,
        typer.Option(
            "--version-prefix",
            "-p",
            help=(
                "Unique prefix to tag your dataset version in LangSmith (e.g., your "
                "username or feature name)."
            ),
        ),
    ],
    num_repetitions: Annotated[
        int,
        typer.Option(
            "--num-repetitions",
            "-n",
            min=1,
            help=(
                "How many times to run routing per dataset row (higher = more confidence, more cost)."
            ),
        ),
    ] = 3,
    selected_tools: Annotated[
        Optional[List[str]],
        typer.Option(
            "--tool",
            "-t",
            help=(
                "Filter to only include the specified tool name(s). Repeat --tool for multiple, "
                "e.g., -t read_file -t write_file"
            ),
        ),
    ] = None,
    eval_dataset: Annotated[
        EvalDataset,
        typer.Option(
            "--dataset-type",
            "-d",
            help="Type of evaluation dataset to use: local, mr, or main.",
        ),
    ] = EvalDataset.LOCAL,
):
    logger.info(
        f"Starting tool routing evaluation with version prefix: {version_prefix}, num_repetitions: {num_repetitions}, "
        f"selected_tools: {selected_tools}, eval_dataset: {eval_dataset.full_path}"
    )

    logger.info("Generating dataset...")
    tool_specs = get_all_ops_tools()
    eval_configs = load_routing_configs()

    dataset_tag = generate_dataset(
        ls_client=ls_client,
        tool_specs=tool_specs,
        eval_samples=eval_configs,
        dataset_name=eval_dataset.full_path,
        tag_prefix=version_prefix,
    )
    logger.info(
        f"Performing evaluation on dataset: {eval_dataset.full_path} with version tag: {dataset_tag}"
    )

    data_full = ls_client.list_examples(
        dataset_name=eval_dataset.full_path, tag=dataset_tag
    )

    logger.info(f"Filtering examples with tools setting: {selected_tools}...")
    data_filtered = [
        example
        for example in data_full
        if not selected_tools
        or (example.metadata or {}).get("tool_name") in selected_tools
    ]

    ls_client.evaluate(
        execute_routing,
        data=data_filtered,
        evaluators=[is_correct],
        num_repetitions=num_repetitions,
        max_concurrency=None,
    )


if __name__ == "__main__":
    app()
