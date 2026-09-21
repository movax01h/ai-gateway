# ADR-002: The routing decision is made once per request; explicit selection bypasses it

- Status: proposed
- Date: 2026-09-03

## Context

Two designs were on the table: switch models per turn inside a flow (the
[!6305](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/6305)
prototype ran turns that digested read-only tool results on the small model), or decide once per
request from the goal.

Provider prompt caches are scoped to one model. Anthropic's guidance: a model switch has no escape
hatch and invalidates the tools, system, and messages caches together; writes cost 1.25 times the
input price and entries expire after five minutes without a read. Haiku 4.5 needs a 4,096-token
prefix before anything caches, Sonnet 4.6 needs 1,024. Gemini's implicit cache is also model-bound.
Duo Developer turns are input-heavy and share a long prefix.

In this codebase the model is bound once per flow: the agent component builds its prompt with the
resolved model at graph construction. Per-turn switching needs a prompt object per tier and a
per-turn selector, which !6305 had to add.

Separately, GitLab Rails marks an explicit choice by sending an `identifier` in the model
metadata header; self-hosted models always carry one and arrive with `provider: openai`.

## Decision

The routing decision is made exactly once per request, from the goal text, before the flow is
built. Every turn of the flow runs on that model. The decision is skipped when the header carries
an `identifier` or a provider other than `gitlab`, so pinned models and every self-hosted model
are never overridden.

## Consequences

- The prompt cache stays warm on one model for the whole flow.
- No component, flow config, or prompt definition changes; the decision flows through the
  request's default model that components already read.
- A wrong decision at the start is wrong for the whole request. The CEF evaluation, pinned to an
  exact gateway build, is how that cost is measured.
- Self-hosted and air-gapped installs are unaffected by construction. The gateway never needs to
  know customer-defined models.

## Alternatives considered

- **Per-turn tiering (!6305)**: bounded upside (only all-read-only tool-result turns moved to the
  small model) against an unbounded cache cost, plus prompt-building changes. Rejected for v1.
- **One-way escalation from `small` to `large` at a step boundary, at most once per flow**: one
  cache rebuild, never a downgrade. Deferred until the per-request version has evidence.
