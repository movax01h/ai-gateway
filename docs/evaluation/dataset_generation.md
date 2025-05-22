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

## Sample Datasets

The following sample datasets were generated using this tool:

| Prompt Definition | Sample Dataset |
| - | - |
| [categorize_question](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/categorize_question) | [categorize_question.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/1f956241-b910-441d-9187-faf35c3a6a88?tab=2) |
| [chat.build_reader](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/chat/build_reader) | [chat.build_reader.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/34e07a8e-ab2a-48d7-8804-9fa015c90083?tab=2) |
| [chat.commit_reader](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/chat/commit_reader) | [chat.commit_reader.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/9db4403b-bd12-42ca-af87-a7b281536f47?tab=2) |
| [chat.documentation_search](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/chat/documentation_search) | [chat.documentation_search.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/7e1b70df-e6d8-4b39-9d25-511a589385f6?tab=2) |
| [chat.epic_reader](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/chat/epic_reader) | [chat.epic_reader.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/79ecdd8b-9032-4d84-bfa5-07d8cabb8402?tab=2) |
| [chat.explain_code](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/chat/explain_code) | [chat.explain_code.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/961a12a1-6017-4d37-bb5d-f7cdb2c5e615?tab=2) |
| [chat.explain_vulnerability](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/chat/explain_vulnerability) | [chat.explain_vulnerability.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/f8c02b9d-1e19-47c5-b712-0473d617aed4?tab=2) |
| [chat.fix_code](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/chat/fix_code) | [chat.fix_code.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/1cffe977-c1a4-4427-989d-989600120d7f?tab=2) |
| [chat.issue_reader](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/chat/issue_reader) | [chat.issue_reader.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/b591285d-ddae-41f7-aa84-3a1d49eb5707?tab=2) |
| [chat.merge_request_reader](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/chat/merge_request_reader) | [chat.merge_request_reader.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/f044e0cb-dcb4-40ae-9f73-ccfa8a9d3c5e?tab=2) |
| [chat.refactor_code](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/chat/refactor_code) | [chat.refactor_code.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/af26cdd0-f2fc-4ba3-ad45-1fed3466939f?tab=2) |
| [chat.summarize_comments](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/chat/summarize_comments) | [chat.summarize_comments.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/2f6513a8-fc64-408d-968b-a1392835fdab?tab=2) |
| [chat.troubleshoot_job](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/chat/troubleshoot_job) | [chat.troubleshoot_job.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/6c9dee7b-b2db-40d3-9d76-39c0b0a7cf9f?tab=2) |
| [code_suggestions.generations](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/code_suggestions/generations) | [code_suggestions.generations.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/cf58f527-ef1c-4cf6-9c40-641ae69ba1b5?tab=2) |
| [generate_commit_message](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/generate_commit_message) | [generate_commit_message.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/5bf4e310-fc9c-4aee-adb3-d1b164be3a26?tab=2) |
| [generate_description](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/generate_description) | [generate_description.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/c4695810-810d-489f-950b-9b59305e4313?tab=2) |
| [glab_ask_git_command](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/glab_ask_git_command) | [glab_ask_git_command.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/053f6f45-b74a-47e9-b02a-7e0c5c6c147d?tab=2) |
| [measure_comment_temperature](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/measure_comment_temperature) | [measure_comment_temperature.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/febdeb7f-223f-4e4e-b406-713b28188b6b?tab=2) |
| [resolve_vulnerability](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/resolve_vulnerability) | [resolve_vulnerability.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/76f16f2c-1ee5-4da0-890a-76856e4f7ffc?tab=2) |
| [review_merge_request](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4d43bdb7748cca4a009b7a3b328f8f0c76b68e69/ai_gateway/prompts/definitions/review_merge_request) | [review_merge_request.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/5b1bc95d-5e23-4923-949b-7e29cb26f7ec?tab=2) |
| [summarize_new_merge_request](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/summarize_new_merge_request) | [summarize_new_merge_request.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/f534f3f4-0b45-43fb-a79c-26deab188831?tab=2) |
| [summarize_review](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/4bb032d493065cf61bdb9f35ee4ef8004c145e78/ai_gateway/prompts/definitions/summarize_review) | [summarize_review.ai_gen_sample.1](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/a7b3705e-e377-43db-b080-cc2ebc9c47ef?tab=2) |
