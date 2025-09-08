# Model selection

Model selection enables AI Gateway clients to seamlessly switch between GitLab-provided and self-hosted models. This
capability allows GitLab to rapidly adopt new models while providing customers with flexibility and compliance within
their AI toolchain.

## Overview

Model selection provides a unified approach for determining the provider, model, and model parameters to use for each
LLM invocation. It aims to reduce the amount of boilerplate needed to integrate and maintain models, and have a clear
hierarchy of values. Its main components are the following:

- Model definitions listed in `ai_gateway/model_selection/models.yml`. They determine the prompt definition file to use
(see [Defining a model](#defining-a-model)), along with default model parameters.
- GitLab-provided models and defaults listed in `ai_gateway/model_selection/unit_primitives.yml` (see
[Configuring model selection](#configuring-model-selection))
- Prompt definition YAML files in `ai_gateway/prompts/definitions`, containing prompt templates and prompt-specific
parameters (see [AI Gateway Prompt and Prompt Registry Documentation](aigw_prompt_registry.md))

## How models and prompts are selected

The primary input for model selection is the [`ModelMetadata`](../ai_gateway/model_metadata.py). Fetching the
appropriate definitions is abstracted via the `create_model_metadata` method, according to the following rules:

- If `name` is provided, the model definition with the matching `gitlab_identifier` is fetched. This logic is used
primarily by model metadata requesting a custom model.
- If `feature_setting` is provided, the `default_model` from the matching entry from `configurable_unit_primitives`
is fetched. This is the case applicable to users selecting "GitLab default model" in the UI
- If `identifier` is provided, the model definition with the matching `gitlab_identifier` is fetched. This is triggered
when the user has selected a GitLab-provided model other than "GitLab default model" in the UI

If none of these values is given to `create_model_metadata`, a `ValueError` is raised. Once the model definition has
been established, the Prompt Registry determines the prompt definition to use based on the following logic:

- If the model definition has a `family` list, it is iterated in order, and the first existing prompt folder is returned
- If no `family` is specified, or if none of the items in the list match a prompt folder, `base` is used

### Examples

Given the following configuration:

```yaml
# ai_gateway/model_selection/models.yml
models:
  - name: Codestral
    gitlab_identifier: codestral
    family:
      - codestral
      - mistral
    params:
      model: codestral:22b
      max_tokens: 4_096
      temperature: 0.0
```

```yaml
# ai_gateway/model_selection/unit_primitives.yml
configurable_unit_primitives:
  - feature_setting: "code_suggestions"
    unit_primitives:
      - "code_suggestions"
    default_model: "codestral"
    selectable_models:
      - "codestral"
```

```yaml
# ai_gateway/prompts/definitions/code_suggestions/completions/mistral/1.0.0.yml
name: Mistral Code Suggestions
model:
  params:
    model_class_provider: litellm
    temperature: 0.1
unit_primitives:
  - complete_code
prompt_template:
  system: Complete the following code
  user: Here's my code: {{code}}
```

Assume `code_suggestions/completions` is given as the `prompt_id`, and `1.0.0` as the `prompt_version` for each example.

#### "Default GitLab model" example

If the model metadata from the request is `{"provider": "gitlab", "feature_setting": "code_suggestions"}`, the following
steps will occur:

- `unit_primitives.yml` will be searched for  `feature_setting: "code_suggestions"`, and its `default_model` retrieved
- `models.yml` will be searched for `gitlab_identifier: codestral`, and its `family` and parameters retrieved
- Each `value` from `family` will be checked in order for a corresponding prompt definition folder, so first the path
`ai_gateway/prompts/definitions/code_suggestions/completions/codestral` is checked. Assuming it doesn't exist, next
`ai_gateway/prompts/definitions/code_suggestions/completions/mistral` is checked. Since it exists, the version file is
searched in that folder
- The prompt definition `ai_gateway/prompts/definitions/code_suggestions/completions/mistral/1.0.0.yml` is loaded

Thus, the resulting values passed to the LLM invocation will be:

```yaml
model: codestral:22b # From model definition
max_tokens: 4_096 # From model definition
temperature: 0.1 # Overwritten in prompt definition
prompt_template: # From prompt definition
  system: Complete the following code
  user: Here's my code: {{code}}
```

#### Custom models example

If the model metadata from the request is
`{"name": "codellama", "provider": "litellm", "endpoint": "http://localhost", "identifier": "codestral:22b-v0.1-q2_K" }`
the following steps occur:

- `models.yml` will be searched for `gitlab_identifier: codestral`, and its `family` and parameters retrieved
- The same process in the previous example to determine the prompt to load based on the `family` will apply
- The prompt definition is loaded

In this case, the resulting values passed to the LLM invocation will be:

```yaml
model: codestral:22b-v0.1-q2_K # Overwritten from user-supplied "identifier"
max_tokens: 4_096 # From model definition
temperature: 0.1 # Overwritten in prompt definition
endpoint: http://localhost # From user-supplied "endpoint"
prompt_template: # From prompt definition
  system: Complete the following code
  user: Here's my code: {{code}}
```

## Defining a model

Models are defined in `ai_gateway/model_selection/models.yml`, and each model has the following properties:

- `name`: A human-readable name for the model, to be displayed in UI
- `gitlab_identifier`: an identifier used to reference the model within GitLab configuration
- `family` (optional): an ordered list of preferred prompt definitions to use with this model (see
[How models and prompts are selected](#how-models-and-prompts-are-selected) for more details)
- `params`: Dictionary with custom parameters to be passed to the model client

Example:

```yaml
models:
  - name: "Claude Sonnet 3.5 - Anthropic"
    gitlab_identifier: "claude_3_5_sonnet_20240620"
    params:
      model_class_provider: "anthropic"
      model: "claude-3-5-sonnet-20240620"
      temperature: 0.0
      max_tokens: 4_096
      max_retries: 1
```

### Authorization

Authentication mechanisms differ across provider APIs. For cloud services like Bedrock and VertexAI, authentication is
handled externally through environment credentials, while FireworksAI requires dynamic endpoint configuration based on
regional deployment. Custom authentication logic can be implemented by extending the `parameters_for_gitlab_provider`
method in the `ai_gateway/model_metadata.py` module.

## Configuring model selection

Each Unit Primitive can be configured with:

- A default model
- A set of selectable models
- Beta models available for testing

Unit primitive groups are defined in `ai_gateway/model_selection/unit_primitives.yml` and the following properties are
available:

- feature_setting: An identifier used to refer to the feature name
- unit_primitives: the list of unit primitives that belong to this group, as defined in
  the [cloud_connector](https://gitlab.com/gitlab-org/cloud-connector/gitlab-cloud-connector/-/blob/main/src/python/gitlab_cloud_connector/gitlab_features.py#L19)
- default_model: the `gitlab_identifier` of the model that is used if the user has not selected a different model
- selectable_models: a list of `gitlab_identifier` for the models that the user can select from
- beta_models: a list of models that are not fully supported but users can select from

Example:

```yaml
configurable_unit_primitives:
  - feature_setting: "duo_chat"
    unit_primitives:
      - "ask_build"
      - "ask_commit"
    default_model: "claude_sonnet_3_7_20250219"
    selectable_models:
      - "claude-3-7-sonnet-20250219"
      - "claude_3_5_sonnet_20240620"
```
