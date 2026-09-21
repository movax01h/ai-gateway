# ADR-003: The v1 signal is deterministic keyword rules on `models_for_tags`

- Status: proposed
- Date: 2026-09-03

## Context

Something has to decide whether a goal is `small` or `large` work. vLLM Semantic Router's
[signal catalogue](https://vllm-sr.ai/docs/tutorials/signal/overview) splits signals into
heuristic ones that need no model (keyword, context token band, conversation shape, structure,
caller metadata) and learned ones that need an embedding runtime or classifier (complexity,
domain, embedding). Its docs call keyword signals "the fastest path to a useful routing graph" and
"a low-latency first pass before learned signals", and warn that they are "sensitive to wording
and can be triggered intentionally". Its complexity signal requires candidate phrases and
thresholds "calibrated together against labeled traffic", re-done whenever the embedding model
changes.

The offline harness in
[!6289](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/6289)
already evaluates keyword decisions against CEF data, with a starter policy for Duo Developer.

`unit_primitives.yml` already has a `models_for_tags` block per feature setting, mapping `small`
and `large` to one model each. Introducing a second, parallel file to hold the routing signal
means the same question, what model for this request, has two answers to keep in sync.

### Why this is an ADR and not a continuation of the RFC

[`gitlab-org/gitlab#604084`](https://gitlab.com/gitlab-org/gitlab/-/work_items/604084) framed the
problem and set out three options with a phased plan: start with cheap scoring, defer an LLM
classifier until cheap scoring proves insufficient. The decision below is that plan's first phase,
not an alternative to it. What the RFC leaves open is what a decision record has to pin down,
where the signal lives and what shape the policy takes, and that is what this ADR settles. The
RFC's own open action item, the
[cost benefit ratio](https://gitlab.com/gitlab-org/gitlab/-/work_items/604084#how-to-build-it--three-options),
is the number the 20% gate produces, so the evidence flows back the other way.

## Decision

The v1 signal is keyword matching on the goal text. The policy lives in `models_for_tags`, extended
with two fields per tag: `models`, a list, so a tag can still resolve to more than one provider
variant of the same model, the same way `default_models` already does, and `keywords`, the
deterministic signal for that tag. Tags are checked in a fixed order, `large` before `small`; the
first tag whose keywords match the goal wins, so a goal that matches both escalates to `large`. No
tag matching keeps today's behavior, `default_models`.

```yaml
models_for_tags:
  large:
    models: [claude_sonnet_4_6, claude_sonnet_4_6_vertex]
    keywords: [refactor, migration, race condition, concurrency, architecture, security]
  small:
    models: [claude_haiku_4_5_20251001, claude_haiku_4_5_20251001_vertex]
    keywords: [typo, readme, documentation, docstring, comment, bump, rename, unused, deprecate]
```

## Consequences

- No separate routing-policy file. The signal sits next to the models it selects, in the file
  already reviewed for every model change, and `models_for_tags` picks up load balancing across
  provider variants of the same model as a side effect of `models` being a list.
- A new signal type, a context-length band, a structure count, a metadata hint from Rails, adds a
  new key next to `keywords` on the tag entry it applies to. Entries that don't use it, and the
  file's shape, don't change. This is the constraint the design has to hold as it grows: adding
  complexity should cost a key, without requiring a migration of the file.
- Keyword lists are reviewable data in a merge request, and they are routing hints, never a
  security boundary.
- The decision costs a regex pass: no model call, no network call, identical on every install.
- Learned signals are deferred until the 20% gate shows keyword rules leaving savings or quality
  on the table.

## Alternatives considered

- The vLLM Semantic Router decision schema: `priority`, a `rules` tree of `AND`/`OR`/`NOT`
  over `type: keyword` conditions, `modelRefs`, and a `default_model`
  ([reference](https://vllm-sr.ai/docs/tutorials/decision/overview)). This was the original v1
  proposal. Kept as prior art for the signal catalogue and the `rules`-tree shape, but rejected as
  the file format: it introduces a routing-specific file that answers the same question
  `models_for_tags` already owns, with none of `unit_primitives.yml`'s existing admin tooling or
  `selectable_models` validation, for a scheme, numeric `priority` across arbitrarily many rules,
  more general than two tags need.
- Learned complexity or embedding signals: could catch complexity that surface wording never
  signals, where a keyword list misses the goal's real shape. Rejected for v1 because they need an
  embedding runtime and calibration against labeled traffic, adding a dependency to every install
  for a gain the 20% gate hasn't shown yet.
- LLM pre-flight classifier (Option B in
  [`gitlab-org/gitlab#604084`](https://gitlab.com/gitlab-org/gitlab/-/work_items/604084)): a quick
  classify step inside the flow, reacting per request instead of matching a fixed list, at the
  cost the RFC itself names, "adds cost + latency to every task", including ones a free regex pass
  already handles. That RFC's own phased plan defers Option B until cheap scoring, what this ADR
  ships, proves insufficient, the same order proposed here, and it leaves two questions
  unanswered either way: the acceptable latency budget for the step, and whether self-managed
  gets auto-routing at all versus admin-pin only. Self-hosted impact is bounded regardless:
  [ADR-002](adr-002-composition-precedence.md)'s bypass check already gates routing on
  `provider: gitlab` and a blank `identifier`, so self-hosted and pinned requests never reach a
  classifier call, the same way they never reach keyword matching today.
- A metadata hint from Rails: viable later as an extra signal type, but Rails has no task
  classification to send today.
