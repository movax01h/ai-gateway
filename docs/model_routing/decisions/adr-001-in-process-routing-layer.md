# ADR-001: Small/large routing is in-house code at the `models_for_tags` seam

- Status: proposed
- Date: 2026-09-03

## Context

Duo Developer needs to choose between the `small` and `large` models that `unit_primitives.yml`
already binds for the feature. Four candidates were scored: in-house code at the existing tag
seam, LiteLLM Router, vLLM Semantic Router as a sidecar, and the AI21 hosted proxy. One hard gate
carries over from [`gitlab-org/gitlab#605921`](https://gitlab.com/gitlab-org/gitlab/-/work_items/605921):
it has to work on self-managed installs.

Two facts decide it. vLLM Semantic Router is a network service: Envoy calls it over the External
Processing protocol, its docs describe the Router and the inference backends as separate services,
and the `vllm-sr` PyPI package is a CLI that orchestrates Docker containers with no importable
evaluation API. And the piece of it we need, keyword decision evaluation, already exists in this
repository as roughly 110 lines of Python with tests in the offline harness from
[!6289](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/6289).

## Decision

The routing decision is in-house code in Duo Workflow Service. It runs once per request and sets
the request's default model to the model bound to the chosen tag. No new runtime dependency enters
the request path, and no component or flow config changes.

The policy lives in `unit_primitives.yml` itself, the file already reviewed for model changes; see
[ADR-003](adr-003-keyword-signal-vllm-sr-schema.md) for its shape. If learned signals ever earn
their cost and adopting the router makes sense later, the keyword lists and model bindings port
over even though the file format doesn't match vLLM Semantic Router's schema.

## Consequences

- We own the matcher, its tests, and its failure modes. There is no vendor to escalate to.
- Every self-managed install gets routing with no extra service to run.
- The policy format is `unit_primitives.yml`'s own shape, extended with `keywords` and, per
  [ADR-004](adr-004-decision-carries-inference-params.md), `params`.
- Learned signals, if wanted later, must arrive either as in-process code or as a deliberate
  decision to adopt the router, not as an accident of the first implementation.

## Alternatives considered

- **vLLM Semantic Router sidecar**: a second critical-path service for every self-managed
  customer, and its learned signals need an embedding runtime calibrated against labeled traffic.
  Rejected for v1 as a service; its decision schema is kept only as reference (see
  [ADR-003](adr-003-keyword-signal-vllm-sr-schema.md)).
- **LiteLLM Router**: solves endpoint balancing, retries, and cooldowns, none of which is in
  scope, and it is unused in this codebase.
- **AI21 hosted proxy**: hosted, so it fails the self-managed gate outright.
