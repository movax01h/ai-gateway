import asyncio

import typer
from dependency_injector.wiring import Provide, inject
from eli5.evaluator import evaluate

from ai_gateway.config import Config
from ai_gateway.container import ContainerApplication
from ai_gateway.prompts.base import BasePromptRegistry


@inject
def eval(
    prompt_id: str,
    prompt_version: str,
    dataset: str,
    prompt_registry: BasePromptRegistry = Provide[
        ContainerApplication.pkg_prompts.prompt_registry
    ],
):
    prompt = prompt_registry.get(prompt_id, prompt_version)

    evaluate(
        lambda inputs: asyncio.run(prompt.ainvoke(inputs)).content,
        dataset=dataset,
        limit=100,
    )


def run(prompt_id: str, prompt_version: str, dataset: str):
    container_application = ContainerApplication()
    container_application.config.from_dict(Config().model_dump())
    container_application.wire(modules=[__name__])

    eval(prompt_id, prompt_version, dataset)


def main():
    typer.run(run)


if __name__ == "__main__":
    main()
