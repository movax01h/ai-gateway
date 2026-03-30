# Load testing

This document gives an overview of the test environment and tools involved in the Duo Workflow Service (DWS) load testing efforts:

- [Load test agentic chat and SW development flows](https://gitlab.com/groups/gitlab-org/quality/-/work_items/201)
- [Load test Security Analyst Agent for performance validation](https://gitlab.com/gitlab-org/gitlab/-/work_items/573854)

The test environment and tools can be reused for subsequent load testing.

## Test environment

### GitLab instance

For load testing, we [provisioned a GitLab test environment](https://gitlab.com/gitlab-org/quality/quality-engineering/team-tasks/-/issues/3791)
in the `gitlab-qa-ai-latency-baseline` GCP project based on the [3k Reference Architecture](https://docs.gitlab.com/administration/reference_architectures/3k_users/).

- [GitLab test environment configuration](https://gitlab.com/gitlab-org/quality/gitlab-environment-toolkit-configs/performance-test-rfh)
  and [deployment instructions](https://gitlab.com/gitlab-org/quality/gitlab-environment-toolkit-configs/performance-test-rfh/-/blob/main/configs/load-test-agentic-chat/README.md).

### DWS test instance

Two DWS test instances are available:

**Cloud Run instance** (`dev-ai-research-0e2f8974` GCP project): deploys DWS alongside an
[LLM caching proxy](profiling_with_llm_caching_proxy.md) and a Pyroscope profiling server.
The LLM caching proxy reduces API costs by caching LLM responses, producing more realistic results
than agentic mocking. This is the preferred approach.
See [Profiling DWS with Pyroscope and LLM caching proxy](profiling_with_llm_caching_proxy.md) for deployment instructions.

**Runway-managed instance** ([`dws-loadtest.staging.runway.gitlab.net`](https://console.cloud.google.com/run/detail/us-east1/dws-loadtest/observability/metrics?project=gitlab-runway-staging),
[provisioned here](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/1313)):
uses [Agentic Mocking](../workflows/agentic_mock.md) to avoid LLM costs by bypassing LLM calls entirely,
but with the consequence that rest results are not necessarily consistent with production.

## Configuring GitLab

The test instance does not have GitLab Duo Self-Hosted enabled, so the DWS URL is configured via Rails console.

For the Runway-managed instance:

```ruby
::Ai::Setting.instance.update!(duo_agent_platform_service_url: "dws-loadtest.staging.runway.gitlab.net")
```

For the Cloud Run instance, use the service URL output by `deploy_with_pyroscope_to_cloudrun.sh`:

```ruby
::Ai::Setting.instance.update!(duo_agent_platform_service_url: "<cloud-run-service-url>")
```

## Performance testing tools

k6 is the primary tool we used for load testing. Tests are located in [`performance_tests/stress_tests/`](../../performance_tests/stress_tests/).
See [`performance_tests/stress_tests/README.md`](../../performance_tests/stress_tests/README.md) for instructions on
running the tests.

For component-level performance testing of the AI Gateway in CI/CD pipelines, see [Performance Tests in CI](performance_tests_in_CI.md).
