# Load testing

This document gives an overview of the test environment and tools involved in the Duo Workflow Service (DWS) load testing effort in [Epic #201 - Load test agentic chat and SW development flows](https://gitlab.com/groups/gitlab-org/quality/-/work_items/201). The test environment and tools can be reused for subsequent load testing.

## Test environment

As part of [load testing Agentic Chat and the Software Development flow](https://gitlab.com/groups/gitlab-org/quality/-/epics/201),
we [provisioned a GitLab test environment](https://gitlab.com/gitlab-org/quality/quality-engineering/team-tasks/-/issues/3791)
in the `gitlab-qa-ai-latency-baseline` GCP project based on the [3k Reference Architecture](https://docs.gitlab.com/administration/reference_architectures/3k_users/).
We also [provisioned a DWS test instance via Runway for Cloud Run services](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/1313),
[`dws-loadtest.staging.runway.gitlab.net`](https://console.cloud.google.com/run/detail/us-east1/dws-loadtest/observability/metrics?project=gitlab-runway-staging).
It has [Agentic Mocking](../workflows/agentic_mock.md) enabled to allow load testing without incurring LLM costs.

- [GitLab test environment configuration](https://gitlab.com/gitlab-org/quality/gitlab-environment-toolkit-configs/performance-test-rfh)
  and [deployment instructions](https://gitlab.com/gitlab-org/quality/gitlab-environment-toolkit-configs/performance-test-rfh/-/blob/main/configs/load-test-agentic-chat/README.md).
- [Runway documentation for Cloud Run services](https://docs.runway.gitlab.com/runtimes/cloud-run/onboarding/)

## Configuring GitLab

The test instance does not have GitLab Duo Self-Hosted enabled, so the DWS URL is configured via Rails console:

```ruby
::Ai::Setting.instance.update!(duo_agent_platform_service_url: "dws-loadtest.staging.runway.gitlab.net")
```

## Performance testing tools

k6 is the primary tool we used for load testing. Tests are located in [`performance_tests/stress_tests/`](../../performance_tests/stress_tests/).
See [`performance_tests/stress_tests/README.md`](../../performance_tests/stress_tests/README.md) for instructions on
running the tests.

For component-level performance testing of the AI Gateway in CI/CD pipelines, see [Performance Tests in CI](performance_tests_in_CI.md).
