---
stage: AI-powered
group: AI Framework
info: Guidance on upgrading the litellm dependency in AI Gateway.
---

# Upgrading litellm

AI Gateway imports private litellm modules, replaces private litellm functions, and depends on the HTTP requests litellm builds for each provider. The normal test suite patches litellm objects, so litellm's own code never runs in it.

The `tests:litellm-bump-check` job runs litellm's real code against fake provider transports. It sends no network traffic and needs no credentials.

## Review a Renovate litellm bump

1. Open the latest merge request from the `renovate/litellm-1.x` branch.
1. Check the `tests:litellm-bump-check` job.
   - Passed: the new version is suitable to merge.
   - Failed: do not merge. Renovate updates the same merge request when the next version is released. Check the job on that version.
1. Read the job log. The dependency report at the top lists packages that changed in `poetry.lock`. Check any line that starts with `MAJOR`.
1. Check that `tests:litellm-image-smoke` passed as well.

The job does not block the merge. A failed job shows as a warning on the pipeline.

To run the job on another merge request, start it manually from the pipeline.

## What the job checks

- **Provider requests.** For Fireworks, Vertex AI (Gemini and Claude) and Bedrock, litellm builds a request. The tests check the URL, authorization header and body, and that litellm reads a canned response. These are the providers `/monitoring/ready` calls through litellm on every instance start.
- **Patches.** Each patch of litellm internals still takes effect. Assigning to a renamed attribute does not raise an error in Python, so a renamed target leaves the patch silently inactive.
- **Image.** `tests:litellm-image-smoke` imports the entry points in the production image, which installs with `--only main`, and polls `/monitoring/healthz`.

## What the job does not check

- Response formats from real providers.
- Failures that appear only under sustained load.
- Anything that needs provider credentials.

## Run the checks locally

```shell
make test-litellm-bump
```

To try a version before Renovate proposes it:

```shell
poetry add "litellm[proxy]@>=X,<Y"
make test-litellm-bump
```

Check the installed version with `poetry run pip show litellm`. The resolved version can be above the lower bound of your pin. Commit `pyproject.toml` and `poetry.lock` together.

## When a test fails

| Test | What failed | What to do |
|---|---|---|
| `test_provider_requests.py` | litellm changed a request or response for a provider | Read the assertion. If the change is intended upstream, update our call or the expected value. |
| `test_patches.py`, route or `chunk_parser` tests | litellm renamed or stopped calling a function we patch | Find the new name and move the patch in `ai_gateway/proxy/clients/__init__.py`. Delete the patch if upstream fixed the behavior. |
| `test_patches.py`, callback test | The success callback payload changed | Update `litellm_async_success_callback` in `ai_gateway/proxy/clients/base.py`. |
| `test_patches.py`, Vertex and mTLS tests | litellm no longer reads `litellm.vertex_project`, `litellm.vertex_location` or `litellm.aclient_session` | Find the new hook. Without it, the Vertex location and client certificate are silently dropped. |

The `fake_vertex_auth` fixture replaces two private litellm methods to skip the Google token exchange. If it raises `AttributeError`, litellm renamed them. Update the fixture.

## Do not merge a litellm bump as expedited

`.gitlab/ci/test.gitlab-ci.yml` is included with `rules: *skip-when-expedited-include`. A merge request with the `pipeline::expedited` label skips these jobs.
