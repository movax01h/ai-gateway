# Model selection

Model selection enables GitLab.com customers to choose from a curated set of GitLab-provided models. This capability
allows GitLab to rapidly adopt new models while providing customers with flexibility and compliance within their AI
toolchain.

## Defining a model

Models are defined in `ai_gateway/model_selection/models.yml`, and each model has the following properties:

- name: A human-readable name for the model, to be displayed in UI
- `gitlab_identifier`: an identifier used to reference the model within GitLab configuration
- provider: actual provider that will serve the model (litellm, anthropic, vertexai, etc)
- provider_identifier: the provider-specific model identifier.
- `params`: Dictionary with custom parameters to be passed to the model client

Example:

```yaml
models:
  - name: "Claude Sonnet 3.5 - Anthropic"
    gitlab_identifier: "claude_3_5_sonnet_20240620"
    provider: "anthropic"
    provider_identifier: "claude-3-5-sonnet-20240620"
    params:
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
