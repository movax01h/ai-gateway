import os
from typing import Annotated, Optional, cast

import typer
from dependency_injector.wiring import Provide, inject
from eli5.datasets.generator import generate_dataset as eli5_generate_dataset
from jinja2 import PackageLoader
from jinja2.loaders import BaseLoader
from jinja2.sandbox import SandboxedEnvironment
from langsmith import Client

from ai_gateway.config import Config
from ai_gateway.container import ContainerApplication
from ai_gateway.prompts.base import BasePromptRegistry


def get_message_source(prompt_template: dict[str, str]) -> dict[str, str]:
    """Gets the raw Jinja templates content from include statements.

    Args:
        prompt_template: A dictionary of template strings keyed by their role (e.g., "system", "user")

    Returns:
        A dictionary with the same keys as prompt_template, but with raw template content
    """
    jinja_env = SandboxedEnvironment(
        loader=PackageLoader("ai_gateway.prompts", "definitions")
    )

    raw_templates = {}

    for role, template_str in prompt_template.items():
        # Extract the template path from include statement
        # Example: "{% include 'chat/explain_code/system/1.0.0.jinja' %}\n"
        # For direct content without includes, use the original template string
        try:
            ast = jinja_env.parse(template_str)
            if not ast.body or not hasattr(ast.body[0], "template"):
                raw_templates[role] = template_str
                continue

            template_path = ast.body[0].template.value
            loader = cast(BaseLoader, jinja_env.loader)
            raw_content = loader.get_source(jinja_env, template_path)[0]
            raw_templates[role] = raw_content
        except Exception as e:
            raise ValueError(f"Error loading template {template_str}: {e}")

    return raw_templates


@inject
def get_prompt_source(
    prompt_id: str,
    prompt_version: str,
    prompt_registry: BasePromptRegistry = Provide[
        ContainerApplication.pkg_prompts.prompt_registry
    ],
):
    prompt = prompt_registry.get(prompt_id, prompt_version)

    # Extract prompt message templates from LangChain objects encapsulated by the Prompt returned from the registry
    chat_prompt_template = prompt.prompt_tpl
    prompt_template = {}
    messages = getattr(chat_prompt_template, "messages", [])
    for message in messages:
        if message.__class__.__name__ == "SystemMessagePromptTemplate":
            role = "system"
        elif message.__class__.__name__ == "HumanMessagePromptTemplate":
            role = "user"
        elif message.__class__.__name__ == "AIMessagePromptTemplate":
            role = "assistant"
        else:
            role = message.__class__.__name__

        template = message.prompt.template
        prompt_template[role] = template

    source_messages = get_message_source(prompt_template)

    user_message = source_messages.get("user", None)
    if not user_message:
        raise ValueError("Prompt must include a user message")

    # The LLM prompt in ELI5 that's used to generate the dataset examples only expects system or user messages.
    # Append any other messages to the user message.
    other_messages = []
    for role, content in source_messages.items():
        if role not in ["system", "user"] and content:
            other_messages.append(f"[{role}]: {content}")

    return {
        "name": prompt.name,
        "prompt_template": {
            "system": source_messages.get("system", None),
            "user": f"{user_message}\n\n{"\n\n".join(other_messages)}",
        },
    }


def create_langsmith_client(upload: bool = False) -> Optional[Client]:
    if not upload:
        return None

    # pylint: disable=direct-environment-variable-reference
    langsmith_api_key = os.environ.get("LANGCHAIN_API_KEY")
    # pylint: enable=direct-environment-variable-reference
    if not langsmith_api_key:
        raise typer.BadParameter(
            "LangSmith API key is required for upload. Set the LANGCHAIN_API_KEY environment variable."
        )
    try:
        return Client(api_key=langsmith_api_key)
    except Exception as e:
        typer.echo(f"Error connecting to LangSmith: {e}", err=True)
        raise typer.Exit(code=1)


def run(
    prompt_id: Annotated[
        str, typer.Argument(help="Prompt ID (e.g., 'chat/explain_code')")
    ],
    prompt_version: Annotated[str, typer.Argument(help="Prompt version constraint")],
    dataset_name: Annotated[str, typer.Argument(help="Name for the dataset")],
    output_dir: Annotated[
        Optional[str],
        typer.Option(
            os.path.join(os.path.dirname(__file__), ".."),
            help="Directory to save the dataset (default: current directory)",
        ),
    ] = None,
    num_examples: Annotated[
        int, typer.Option(help="Number of examples to generate")
    ] = 10,
    temperature: Annotated[
        float, typer.Option(help="Temperature setting for generation")
    ] = 0.7,
    upload: Annotated[
        bool,
        typer.Option(
            "--upload",
            "-u",
            help="Upload the dataset to LangSmith after generation",
        ),
    ] = False,
    description: Annotated[
        Optional[str],
        typer.Option(
            "--description",
            help="Optional description for the LangSmith dataset (only used with --upload)",
        ),
    ] = None,
):
    """Generate a synthetic dataset for evaluating prompts using templates from the prompt registry.

    Args:
        prompt_id: The ID of the prompt template in the registry (e.g., 'chat/explain_code')
        prompt_version: Version constraint for the prompt template (e.g., '1.0.0')
        dataset_name: Name for the generated dataset (will be used in the output filename)
        output_dir: Directory to save the dataset file (defaults to the project root directory)
        num_examples: Number of examples to generate (default: 10)
        temperature: Temperature setting for generation (higher values = more diverse examples)
        upload: Whether to upload the dataset to LangSmith
        description: Optional description for the LangSmith dataset

    Returns:
        Path to the generated dataset file
    """
    container_application = ContainerApplication()
    container_application.config.from_dict(Config().model_dump())
    container_application.wire(modules=[__name__])

    langsmith_client = create_langsmith_client(upload)

    typer.echo(
        f"Generating dataset with {num_examples} examples from prompt: {prompt_id}"
    )

    prompt_source = get_prompt_source(prompt_id, prompt_version)

    output_file = eli5_generate_dataset(
        prompt_source=prompt_source,
        dataset_name=dataset_name,
        output_dir=output_dir,
        num_examples=num_examples,
        temperature=temperature,
        upload=upload,
        langsmith_client=langsmith_client,
        dataset_description=description,
    )

    typer.echo(f"Dataset generated successfully: {output_file.resolve()}")
    if upload:
        typer.echo(f"Dataset '{dataset_name}' uploaded to LangSmith")

    return output_file


def main() -> None:
    typer.run(run)


if __name__ == "__main__":
    main()
