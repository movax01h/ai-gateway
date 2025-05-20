# Dataset Generation for AIGW Prompts

This document explains how to generate synthetic evaluation datasets directly from AI Gateway prompt definitions.

## Overview

The dataset generation tool allows you to:

1. Select any prompt from the AIGW registry.
1. Generate synthetic examples based on the prompt structure.
1. Export a LangSmith-compatible dataset in JSONL format.

## Using the Dataset Generator

Add the following to your `.env`:

- `ANTHROPIC_API_KEY`
- `LANGCHAIN_API_KEY` (see
[the eli5 prerequisites doc](https://gitlab.com/gitlab-org/modelops/ai-model-validation-and-research/ai-evaluation/prompt-library/-/tree/main/doc/eli5#prerequisites)
on instructions on how to gain access to LangSmith)

### Command-line Interface

You can generate a dataset using the Poetry script:

```shell
poetry run generate-dataset [OPTIONS] PROMPT_ID PROMPT_VERSION DATASET_NAME
```

#### Arguments

- `PROMPT_ID`: The ID of the AIGW prompt (e.g., `chat/explain_code/base`)
- `PROMPT_VERSION`: Version constraint for the prompt template
- `DATASET_NAME`: Name for the output dataset (required)

#### Options

- `--output-dir`: Directory to save the dataset (default: project root directory)
- `--num-examples`: Number of examples to generate (default: 10)
- `--temperature`: Temperature setting for generation (default: 0.7)
- `--upload`, `-u`: Upload the dataset to LangSmith after generation (default: False)
- `--description`: Optional description for the LangSmith dataset (only used with --upload)

### Examples

Generate 10 examples for the explain code prompt:

```shell
poetry run generate-dataset \
  chat/explain_code \
  1.0.0 \
  duo_chat.explain_code.1
```

Generate a larger dataset with different temperature:

```shell
poetry run generate-dataset \
  generate_commit_message \
  1.0.0 \
  generate_commit_message.1 \
  --num-examples 50 \
  --temperature 0.3
```

Generate a dataset and upload it to LangSmith:

```shell
poetry run generate-dataset \
  resolve_vulnerability \
  1.0.0 \
  resolve_vulnerability.1 \
  --upload
```

## How It Works

1. **Prompt Loading**: The tool loads the specified prompt from the AI Gateway registry.
1. **Template Resolution**: All Jinja templates referenced in the prompt are resolved using `get_message_source()` function.
1. **Dataset Generation**: The tool uses ELI5 libraries to generate examples:
   - Extracts system and user templates from the prompt structure.
   - Creates diverse input examples with realistic values.
   - Generates expected outputs.
   - Formats everything in a LangSmith-compatible structure.
1. **Export**: The dataset is exported as a JSONL file and can be uploaded to LangSmith using the `--upload` option.

## Output Format

The generated dataset will be in JSONL format, with each line representing a test case:

```jsonl
{"inputs": {"variable1": "value1", ...}, "outputs": {"output": "expected response"}}
{"inputs": {"variable1": "value2", ...}, "outputs": {"output": "expected response"}}
```

## Integration with LangSmith

The generated datasets are compatible with LangSmith. To use a generated dataset in LangSmith:

1. **Upload the dataset** using the `--upload` option
1. **Run evaluations** against this dataset using the [eval command](../tests.md#running-prompt-evaluations-locally)
