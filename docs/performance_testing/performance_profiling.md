# Performance profiling

## Google Cloud Profiler

Cloud Profiler is [automatically enabled for Duo Workflow Service (DWS) deployments on Cloud Run and GKE via Runway](https://docs.runway.gitlab.com/guides/continuous-profiling/).
For detailed information on using Cloud Profiler, see the [documentation](https://docs.cloud.google.com/profiler/docs).

### Accessing profiles

1. Navigate to [GCP Profiler](https://console.cloud.google.com/profiler)
1. Select project (e.g., `gitlab-runway-staging` or `gitlab-runway-production`)
1. Filter by service (e.g., `duo-loadtest`)
1. (Optionally) select version to filter by specific deployment (e.g., `dws-loadtest-lhzl`)
1. (Optionally) select weight to filter the top 1-50% of profiles by duration

### Profile types

- **CPU Time**: Shows where CPU cycles are spent. Useful for identifying computational bottlenecks
- **Wall Time**: Shows actual elapsed time including I/O waits. Better for understanding overall performance

Note that the profiler shows aggregate data from profiles sampled within the specified timespan.
If the service was mostly idle during the specified timespan, the profile is likely to appear quite different
in comparison to when the service is busy.

### Key findings from production profiling

Based on profiling data from production DWS (see [issue #3794 note](https://gitlab.com/gitlab-org/quality/quality-engineering/team-tasks/-/issues/3794#note_2838923664)),
we were able to identify [many opportunities for improvement](https://gitlab.com/groups/gitlab-org/-/work_items/19747).
Some of the changes lead to a [substantial performance gain](https://gitlab.com/gitlab-org/quality/quality-engineering/team-tasks/-/issues/3794#note_2848614309), from a maximum of 45 sustained concurrent requests to at least 200.

## Memray memory profiling

[Memray](https://bloomberg.github.io/memray/) provides detailed memory allocation tracking for Python applications.

### Running with Memray

```shell
# This make command will start DWS via poetry and memray
make duo-workflow-service-memray
```

### Generating flamegraphs

After you shutdown DWS, memray will output a command you can run to generate a flamegraph. For example:

```shell
[memray] Successfully generated profile results.

You can now generate reports from the stored allocation records.
Some example commands to generate reports:

/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/.venv/bin/python -m memray flamegraph memray-string.80634.bin
```

### Key findings

[Testing high memory usage](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/1648)
with 40 concurrent Agentic Chat requests, each streaming 50KB of text, pushed DWS memory usage to over 3GB.
Memory profiling showed that the checkpoint notifier was responsible for approximately 1.5GB of the memory consumption.

[A fix](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/3925)
reduced checkpoint notifier memory usage to 25MB under the same test conditions.

To reproduce that test for high memory usage in the checkpoint notifier, with memray profiling enabled:

1. Make sure your GDK is running and [configured to use your local DWS](https://gitlab.com/gitlab-org/gitlab-development-kit/-/blob/main/doc/howto/ai/_index.md#configure-ai-services).
1. Stop GDK's DWS service because we will restarting it via memray:

   ```shell
   gdk stop duo-workflow-service
   ```

1. From the AIGW/DWS directory, start DWS via memray with Agentic Mocking enabled:

   ```shell
   export AIGW_MOCK_MODEL_RESPONSES=true
   export AIGW_USE_AGENTIC_MOCK=true

   make duo-workflow-service-memray
   ```

1. Execute the Agentic Chat stress test with a goal that makes the agent stream a long response (in this example,
a 50kb file):

   ```shell
   export ENVIRONMENT_URL=http://gdk.test:3000
   export ACCESS_TOKEN=ypCa3Dzb23o5nvsixwPA
   export AI_DUO_WORKFLOW_ROOT_NAMESPACE_ID=1000000 # The ID of a root group with Duo enabled
   export AI_DUO_WORKFLOW_PROJECT_ID=1000001 # The ID of a project in the group with Duo enabled

   MOCKED_GOAL_FILE=goals/stream_long_response.txt \
     k6 run performance_tests/stress_tests/api_v4_duo_workflow_chat_graphql_api.js
   ```

   Note that this test will take several minutes to complete.

1. When the test is complete, stop the DWS and execute the command you're shown to generate a flamegraph.
