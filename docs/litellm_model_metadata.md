# External LiteLLM Model Metadata

AI Gateway uses [LiteLLM](https://docs.litellm.ai/) to talk to many model
providers behind a single interface. LiteLLM keeps an internal registry of
known models and their capabilities (token limits, pricing, whether they
support `tool_choice`, function calling, vision, etc.). It consults that
registry to decide which parameters can be forwarded to the upstream
provider.

When a customer uses a self-hosted or third-party model that is not in
LiteLLM's built-in registry (for example, a custom Fireworks AI deployment),
LiteLLM may strip or reject parameters such as `tool_choice`, causing
requests to fail even though the underlying provider supports them.

To work around this without requiring a LiteLLM proxy, AI Gateway lets
operators supply a JSON file declaring custom model metadata. The entries
are registered with LiteLLM at startup via
[`litellm.register_model`](https://docs.litellm.ai/docs/completion/input#registering-a-custom-model--provider).

## Enabling the feature

Set the `AIGW_LITELLM__MODEL_METADATA_FILE` environment variable to the
path of your JSON file:

```shell
export AIGW_LITELLM__MODEL_METADATA_FILE=/etc/aigw/litellm_models.json
```

If the variable is unset, no external metadata is registered. If the file
is missing or invalid, a warning is logged and AI Gateway continues to
start - existing model integrations are unaffected.

The file is registered last, after LiteLLM's bundled registry and AI
Gateway's own registrations in `ai_gateway/models/v2/chat_litellm.py`. An
entry fills in a model that is missing. It overrides a model that is already
known only when the key has the same shape as the lookup; see
[Key rules](#key-rules).

## File format

The file must be a valid JSON document with a top-level object containing
a `models` key. Each key under `models` is a model identifier (a bare model
name, or a name prefixed with the LiteLLM provider) and the value is an object
of model capability flags.

```json
{
  "models": {
    "fireworks_ai/accounts/gitlab/deployments/my-model": {
      "litellm_provider": "fireworks_ai",
      "mode": "chat",
      "input_cost_per_token": 1.3e-07,
      "output_cost_per_token": 3.8e-07,
      "max_input_tokens": 262144,
      "max_output_tokens": 262144,
      "supports_function_calling": true,
      "supports_tool_choice": true,
      "supports_response_schema": true
    },
    "my-internal-model": {
      "mode": "chat",
      "max_input_tokens": 131072,
      "max_output_tokens": 16384,
      "supports_function_calling": true,
      "supports_tool_choice": true,
      "supports_vision": false
    }
  }
}
```

### Key rules

How a key is matched depends on its shape:

- A bare model name with no `litellm_provider` (`my-internal-model`) applies
  under every provider, for every flag. A self-hosted model configured as
  `custom_openai/my-internal-model` is looked up as `my-internal-model` under
  `custom_openai`, so the bare key is the one that matches. Prefer this shape.
  With `litellm_provider` set, LiteLLM's own lookups, such as
  `supports_tool_choice`, find the entry only under that provider. The AI
  Gateway's image check still finds it under every provider.
- A prefixed key (`custom_openai/my-internal-model`) applies only under that
  provider, and only when `litellm_provider` is not set or has the same value
  as the prefix. A prefixed key with a different `litellm_provider` is never
  found.
- For a model LiteLLM already knows, an entry overrides only in the matching
  shape: a bare key overrides bare lookups, `provider/name` overrides lookups
  under that provider. A bare key does not override a `provider/name` entry
  in the bundled registry.

### Common fields

| Field | Description |
|-------|-------------|
| `litellm_provider` | The LiteLLM provider name (`fireworks_ai`, `openai`, `bedrock_converse`, etc.). Leave it out on a bare key, so that the entry applies under every provider. |
| `mode` | Usually `chat` for chat completion models. |
| `max_input_tokens` | Maximum context window in tokens. |
| `max_output_tokens` | Maximum tokens the model may emit. |
| `input_cost_per_token` | Optional. Accepted, but the AI Gateway does not read it. |
| `output_cost_per_token` | Optional. Accepted, but the AI Gateway does not read it. |
| `supports_function_calling` | Set to `true` if the model supports tool/function calling. |
| `supports_tool_choice` | Set to `true` to ensure LiteLLM forwards the `tool_choice` parameter. |
| `supports_response_schema` | Set to `true` if the model supports structured output / response schemas. |
| `supports_vision` | Set to `false` for a model that cannot read images; the gateway then replaces image blocks with a text notice before the request. Without a declaration the image is sent. Customer steps: [Declare image support for a self-hosted model](https://docs.gitlab.com/administration/gitlab_duo_self_hosted/configure_duo_features/#declare-image-support-for-a-self-hosted-model). |
| `supports_prompt_caching` | Set to `true` if the provider supports prompt caching. |

The full set of recognized fields is defined by LiteLLM. Refer to the
[LiteLLM model JSON](https://github.com/BerriAI/litellm/blob/main/litellm/model_prices_and_context_window_backup.json)
for the complete list.

## When to use this

Use this feature when:

- You have configured a self-hosted or third-party model in AI Gateway
  (for example via Duo Self-Hosted) and requests fail because LiteLLM
  strips `tool_choice` (or another parameter) from the payload.
- You do not want to run a LiteLLM proxy in front of AI Gateway.
- You need to override or fill in missing capability flags for a model
  that is not yet in LiteLLM's bundled registry.

If your model **is** in LiteLLM's bundled registry, you usually do not
need this file: LiteLLM already knows the model's capabilities.

The self-hosted AI Gateway images set `LITELLM_LOCAL_MODEL_COST_MAP=True`,
so LiteLLM uses this bundled registry and does not download the latest
one from `raw.githubusercontent.com` at startup. Where the deployment
can reach that host, setting `LITELLM_LOCAL_MODEL_COST_MAP=""` restores
the download.

## Troubleshooting

- **My models are not being registered.** Check the AI Gateway logs at
  startup for messages from the `litellm_model_registry` logger. Common
  causes are an unreadable file path, invalid JSON, or a `models` value
  that is not an object.
- **`tool_choice` is still being stripped.** Make sure
  `"supports_tool_choice": true` is present in the model entry, and that
  the key matches how the gateway looks the model up; see
  [Key rules](#key-rules). For a self-hosted model, the key is the served
  model name without the `custom_openai/` prefix.
