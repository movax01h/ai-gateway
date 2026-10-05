---
stage: AI-powered
group: AI Framework
info: Guidance on upgrading the litellm dependency in AI Gateway.
---

# Upgrading litellm

The `tests:litellm-bump-check` job runs on Renovate litellm bump merge requests. It does not block the merge.

- Passed: the new version builds the same request schemas as the version we run now. The merge request can be merged.
- Failed: wait for Renovate to update the merge request with the next version.

The job needs no provider credentials and sends no requests. It replaces the HTTP transport and runs litellm's real code.

## What the job checks

- **Request schemas.** For Fireworks, Vertex AI and Bedrock, the request litellm builds must match the schema in `tests/litellm_bump/schemas/`. The schema holds the URL path, the header names and the structure of the body, not the values.
- **Provider schema.** For Bedrock, the type and required flag of every field we send must match the Converse definition in `botocore`. Vertex AI and Fireworks have no provider check yet.
- **Patches.** Each litellm patch in this repo still takes effect. A renamed patch target fails the job.

A passing job proves the request schema is unchanged. It does not prove that the providers accept the request, or that credentials work.

## Update the schemas

An engineer must update the schemas in this job when a provider changes its schema or when an intended litellm change alters a request. Review the failure diff, then record the new schemas:

```shell
UPDATE_LITELLM_SCHEMAS=1 poetry run pytest tests/litellm_bump
```

Commit the changed files in `tests/litellm_bump/schemas/`. Do this in a separate merge request, because Renovate may overwrite commits on its branch.

## Run the job locally

```shell
LITELLM_LOCAL_MODEL_COST_MAP=True poetry run pytest tests/litellm_bump
```

## Do not merge a litellm bump as expedited

A merge request with the `pipeline::expedited` label skips this job.
