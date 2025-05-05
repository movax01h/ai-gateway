import asyncio
from typing import Annotated

import typer
from dependency_injector.wiring import Provide, inject
from eli5.core.evaluators.prebuilt import CorrectnessEvaluator
from eli5.evaluator import evaluate
from langchain_anthropic import ChatAnthropic

from ai_gateway.config import Config
from ai_gateway.container import ContainerApplication
from ai_gateway.prompts.base import BasePromptRegistry

EVALUATORS = {
    # In this iteration, we assume that all evaluators are LLM-based and require an LLM to be configured.
    # We rely on Claude 3.5 Sonnet (20240620) with the temperature set to 0.
    "correctness": CorrectnessEvaluator,
}


@inject
def eval(
    prompt_id: str,
    prompt_version: str,
    dataset: str,
    evaluators: list[str] | None,
    prompt_registry: BasePromptRegistry = Provide[
        ContainerApplication.pkg_prompts.prompt_registry
    ],
):
    prompt = prompt_registry.get(prompt_id, prompt_version)

    if evaluators:
        model = ChatAnthropic(model="claude-3-5-sonnet-20240620", temperature=0.0)  # type: ignore[call-arg]
        evaluators_instances = [EVALUATORS[name](model=model) for name in evaluators]
    else:
        evaluators_instances = None

    evaluate(
        lambda inputs: asyncio.run(prompt.ainvoke(inputs)).content,
        dataset=dataset,
        evaluators=evaluators_instances,
        limit=100,
    )


def _main(
    prompt_id: Annotated[str, typer.Option()],
    prompt_version: Annotated[str, typer.Option()],
    dataset: Annotated[str, typer.Option()],
    evaluators: Annotated[list[str] | None, typer.Argument()] = None,
):
    container_application = ContainerApplication()
    container_application.config.from_dict(Config().model_dump())
    container_application.wire(modules=[__name__])

    eval(prompt_id, prompt_version, dataset, evaluators)


def main():
    typer.run(_main)


if __name__ == "__main__":
    main()
