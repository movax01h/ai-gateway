# LiteLLM parameter tests

The `tests:litellm-params` job checks that LiteLLM accepts the parameters our code sends. It runs on Renovate LiteLLM
bump merge requests. It does not block the merge.

- Passed: LiteLLM still accepts the parameters for every LiteLLM entry in `models.yml`. The merge request can be merged.
- Failed: do not merge. Renovate updates the same merge request when the next version is released. Check the job again.

## How it works

The suite is in `litellm_param_tests/`. It is outside `testpaths`, so `tests:unit` and `make test` do not run it.

1. It reads every `litellm`, `litellm_completion` and `litellm_embedding` entry from `models.yml`.
1. It builds each model through the same prompt registry the service uses, with the mock flag off.
1. It sends each parameter set through the model. LiteLLM's real code validates the parameters and builds the request.
1. A fake HTTP transport answers the request. No request leaves the process, and the job needs no credentials.
1. A case fails if LiteLLM raises, if the request goes to an unexpected URL, or if anything tries to open a real
   connection.

A new entry in `models.yml` becomes a case with no code change.

## Read a failure

The test ID is `<entry>-<parameter set>`. The message names the model, the parameter set, the LiteLLM error, any
unexpected request and any blocked connection.

A pass proves that LiteLLM accepts the parameters. It does not prove that a provider accepts the request, or that
credentials work. Failures that need load, such as timeouts, are out of scope.

## Parameter sets

Each set is backed by an incident or by a parameter every route must accept.

| Kind | Sets |
|---|---|
| Chat | `plain`, `tools_tool_choice` (#2587), `tool_calls_without_tools` (#2121), `max_tokens`, `temperature` |
| Completion | `plain`, `fim` |
| Embedding | `plain` |

## Add a parameter set

After a new parameter validation incident, add a `ParamSet` to `litellm_param_tests/cases.py` with the incident in its
`why`. Run the suite. If a provider legitimately rejects the set, add a route skip.

## Skips

Every skip has a reason, and a test fails if a skip names something that no longer exists.

- `SKIPS` holds entries that are not run, for example deprecated models.
- `SET_SKIPS` holds `(route, parameter set)` pairs that a provider legitimately rejects and that production never
  sends. The route is the entry's `custom_llm_provider`. A new entry on an existing route needs no edit.

## Run it locally

```shell
make test-litellm-params
```

To try a version before Renovate proposes it, run `poetry add "litellm[proxy]@>=X,<Y"` and then the command above.
Check the installed version with `poetry run pip show litellm`.

The Vertex AI stub replaces two private LiteLLM methods to skip the Google token exchange. If LiteLLM renames them, the
suite fails with an `AttributeError` in the harness. Update `_stub_vertex_auth` in `litellm_param_tests/conftest.py`.

## Do not merge a bump as expedited

A merge request with the `pipeline::expedited` label skips this job.
