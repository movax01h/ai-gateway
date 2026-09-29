---
stage: AI-powered
group: AI Framework
info: Guidance on upgrading the litellm dependency in AI Gateway.
---

# Upgrading litellm

AI Gateway reaches deep into litellm. It imports private modules, replaces
private functions, and depends on the exact HTTP request litellm builds for each
provider. None of that is covered by a deprecation policy, and none of it is
exercised by the normal test suite, which patches litellm at the object level.

Upgrading litellm has caused repeated staging deploy failures and production
incidents. This page describes the gate that runs on every merge request that
could move litellm, and what to do when it fires.

## Why a litellm bump can fail a deployment

The Runway startup probe hits `/monitoring/ready`. That endpoint makes real
model calls on every instance start: `validate_fireworks_available` calls
Fireworks, and `validate_default_models` calls every default model in
`unit_primitives.yml` that is not in the hardcoded skip set in
`ai_gateway/prompts/base.py`.

If a litellm change breaks any one of those provider paths, instances never
become ready and the rollout fails. Production runs `min_instances: 8`, so this
is not a partial degradation. That is the mechanism behind several of the
incidents in the history below.

## The gate

Two CI jobs run on merge requests that touch litellm, the models or proxy code,
or the conformance files themselves. Neither needs provider credentials.

### `tests:litellm-conformance`

**Stage A - blocklist and resolution.** `scripts/litellm_blocklist.py` refuses a
litellm release we already know is broken. The list lives in
`scripts/litellm_blocklist.yml`, with a reason and an upstream link per entry.
The job also exports the resolved dependency set as an artifact so a reviewer can
see which transitive pins moved.

**Stage B - private API surface.** `scripts/litellm_surface.py` holds a manifest
of every litellm module, symbol, monkeypatch target and signature the repo
depends on. `--check` verifies the installed litellm against it.
`tests/scripts/test_litellm_surface.py` runs the same checks as tests, so they
also fire in `tests:unit` and locally.

### `tests:litellm-image-smoke`

**Stage D - production image.** The image installs with
`poetry install --only main`, while every test environment also installs the
`test` and `dev` groups. An import-time break can therefore exist only in the
image. This job imports the entry points inside the built image and then boots it
and polls `/monitoring/healthz`.

`/monitoring/healthz` is `health([])` and makes no model calls.
`/monitoring/ready` does make real provider calls and is deliberately not used
here.

## Running the gate locally

```shell
make test-litellm-conformance
```

That runs the blocklist check, the surface check and the contract tests. It does
not run the image smoke test, which needs Docker and a built image.

## When the gate fires

### A blocklisted version

Pick a version outside the blocked ranges. If the upstream issue is fixed, remove
the entry from `scripts/litellm_blocklist.yml` in the same merge request and say
in the description why you believe it is fixed.

### A missing module or symbol

litellm moved or removed something the repo imports. Find the new location,
update the import at the call site, and update `MODULES` or `SYMBOLS` in
`scripts/litellm_surface.py`. This is the failure mode behind `!5183`.

### A missing or non-callable patch target

This is the most dangerous finding, because without the check it is silent.

`ai_gateway/proxy/clients/__init__.py` assigns over
`OpenAIChatCompletionResponseIterator.chunk_parser`, and
`ai_gateway/models/v2/litellm_empty_text_patch.py` replaces
`_sanitize_empty_text_content`. If litellm renames either, the assignment creates
a dead attribute. The override never runs and nothing raises.

Find out what replaced it, move the patch, and update `PATCH_TARGETS`. If the
upstream behaviour the patch works around has been fixed, delete the patch
instead and note it in the merge request.

### A changed signature

`PINNED_SIGNATURES` covers narrow, stable functions and anything the repo wraps.
Read the diff in the job output. If our call sites are unaffected, re-record:

```shell
poetry run python scripts/litellm_surface.py --snapshot
```

Commit the updated `scripts/litellm_surface.json` and say in the merge request
what changed and why it is safe.

Expect some noise here. litellm periodically modernises its type hints, so
`Optional[List[str]]` becomes `list[str] | None` with no behaviour change. That
still shows as a changed signature. Dismissing it is a judgement call a reviewer
makes once per bump, which is the trade for catching the real shape changes.

### A missing required parameter

`REQUIRED_PARAMS` lists the parameters our call sites pass by name to wide,
growing functions such as `acompletion`. Pinning the whole signature there would
fire on every upstream release, so only the parameters we actually use are
asserted. A finding here means a call site has to change.

### A manifest completeness failure

Someone added a litellm import without adding it to the manifest, so it has no
protection. Add the module or symbol to `scripts/litellm_surface.py`.

This check exists because the manifest is hand-maintained. Adding a dependency on
a litellm internal should be a deliberate, reviewed act, and the one-line
manifest edit is what makes it one.

## Preparing a bump

Before opening the merge request, compare the two releases:

```shell
poetry run python scripts/litellm_surface.py --diff 1.101.0 1.103.0
```

This installs both versions in throwaway virtualenvs and prints what moved. Paste
the output into the merge request description. A reviewer should not have to
rediscover it.

Then:

1. `poetry add "litellm[proxy]@>=X,<Y"`. Never hand-edit `poetry.lock`.
2. Run `make test-litellm-conformance`.
3. Check the resolved-dependency artifact for transitive surprises. litellm pulls
   a large tree, and a bump has previously forced a `fastapi` downgrade (`!5531`)
   and changed `pyroscope-io` and `uvicorn` (`!5453`).
4. Run the full test suite.

## Never merge a litellm bump expedited

`.gitlab/ci/test.gitlab-ci.yml` is included with
`rules: *skip-when-expedited-include`. A merge request labelled
`pipeline::expedited`, or a commit message containing `pipeline_expedited: true`,
excludes the whole file. That skips these jobs along with `tests:unit`.

Expedited is an emergency path (see `docs/delivery/emergency_deployment.md`). A
dependency bump is never an emergency.

## What the gate does not catch

The gate covers failures that are deterministic and observable without load or a
live provider. It does not cover failures that depend on load, concurrency or
duration, because those need a deployed service under sustained traffic.

Three past incidents are in that category:

| Incident | Symptom |
|---|---|
| `!4315` | proxy endpoints returned 5xx with `Cannot send a request, as the client has been closed` |
| `!4421` | elevated timeouts across Duo Agent Platform |
| `!4351` | A1004 error-rate spike |

Runway has no review-app or per-branch environment, so catching these means a
scratch Cloud Run service built from the per-commit image plus a synthetic
traffic run. That work is tracked in `#2276`, "Traffic Simulator for
deployments", which serves every deploy rather than only litellm bumps.

The gate also does not yet assert on the HTTP request body litellm builds. That
is planned as Stage C and is described in `#2956`. It is what would catch the
`#2335` class of bug, where litellm silently rewrites a request. Until it lands,
a reviewer should read the litellm release notes for changes to request
construction on the providers the startup probe uses.

## Incident history

Kept here so the reasons behind the blocklist entries do not evaporate when the
merge requests are closed.

| Reference | Symptom | Found in |
|---|---|---|
| `#2335` / `!5755` | 1.85.0 rewrote empty assistant content into a user-visible placeholder | production |
| `!6829` / `production#22949` | 1.99.0 broke Bedrock with `'NoneType' object has no attribute 'access_key'`; staging deploys failed and blocked production | staging |
| `!4351` | 1.81.1 correlated with an A1004 error spike | production |
| `!4421` | 1.80.0 caused elevated timeouts across Duo Agent Platform | production |
| `!4315` | proxy 5xx from a closed httpx client | production |
| `!5453` | versions before 1.82.6 crashed on Vertex `finish_reason='refusal'` | production |
| `!5531` | 1.83.14 forced a `fastapi` downgrade and a `cryptography` bump | n/a |
| `!5183` | `ModuleNotFoundError` on a `litellm.proxy` submodule | downstream CI |
| `!5010` | needed to exclude 1.82.7 and 1.82.8 | n/a |
