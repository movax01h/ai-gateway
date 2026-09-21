# ADR-004: A routing decision carries inference parameters, temperature in v1

- Status: proposed
- Date: 2026-09-03

## Context

Choosing the model is one lever; how it samples is another. Today temperature comes from
`models.yml`, where both tier models declare `temperature: 0.0`. The prompt builder merges
parameters with a fixed precedence, highest first: provider block, the prompt definition's
`model.params`, identifier-derived parameters, `models.yml` defaults. The Duo Developer prompt
definition sets no `model.params`. The only override is deployment-wide and applied at startup;
nothing exists per request.

Provider rules differ. The Claude 4.x line accepts `temperature`; Sonnet 5 rejects non-default
values; Opus 4.7 and later and Fable 5 return a 400 for any sampling parameter. Gemini 3 omits
temperature unless set and accepts 0 to 2. Temperature does not invalidate a provider prompt
cache; `thinking`, `effort`, and `thinking_level` do, and they differ per provider.

vLLM Semantic Router's `modelRefs` entries carry a per-ref inference knob (`use_reasoning`),
precedent that a per-tag parameter is a reasonable shape, even though
[ADR-003](adr-003-keyword-signal-vllm-sr-schema.md) keeps the policy in `unit_primitives.yml`
rather than adopting that schema.

## Decision

A tag entry in `models_for_tags` may carry `params`, alongside `models` and `keywords`. In v1 the
only supported key is `temperature`. The router attaches the matched tag's `params` to the
resolved model metadata as a request-scoped override, and the prompt builder merges it as one new
layer between the prompt definition's `model.params` and the identifier-derived parameters. A
routed temperature beats the `models.yml` default and loses to a prompt author's explicit choice.

Parameters are validated when the policy loads: against the target model's parameter schema, and
against a provider rule that rejects sampling parameters for models that do not accept them. A
policy that fails validation is not loaded, and the gateway starts with routing disabled.

## Consequences

- Temperature becomes a per-task-class policy knob without touching prompt definitions or
  `models.yml`.
- The evaluation grid gains an axis: model by temperature. The 20% gate applies to the pair that
  ships.
- The prompt builder gains one merge layer and the model metadata one optional field. Both are
  small, but they are code changes in the gateway, not only in Duo Workflow Service.
- `thinking`, `effort`, and `thinking_level` stay out until a per-decision change that
  invalidates the messages cache has evidence behind it.

## Alternatives considered

- **Pre-baked model variants in `models.yml`** (the same model with a different temperature,
  bound to an extra tag): no code change, but one entry per model and temperature pair, and every
  variant leaks into `selectable_models` validation and the admin model list.
- **Prompt-definition `model.params`**: per prompt, not per decision, and it would override the
  policy for every request.
- **Letting routed parameters outrank the prompt definition**: rejected so a prompt author's
  explicit choice stays authoritative; revisit if a prompt ever needs to be routed away from its
  own setting.
