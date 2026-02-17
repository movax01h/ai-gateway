# Tests

This project uses [Pytest](https://docs.pytest.org/en/stable/) for testing.
Additionally, we use several Pytest plugins to enhance our testing capabilities:

- `pytest-asyncio`: For testing asynchronous code.
- `pytest-cov`: For code coverage reporting.
- `pytest-randomly`: For randomizing test order to detect inter-dependencies.
- `pytest-watcher`: For running tests in watch mode.
- `pytest-xdist`: For running tests in parallel.

## Unit test

To run the entire unit test suite, you can use the following command:

```shell
make test
```

To run the tests in watch mode, use the following command:

```shell
make test-watch
```

To see test coverage, you can run the following command:

```shell
make test-coverage
```

This will run all the tests, output coverage in the terminal and generate an HTML report.
You can view the HTML report by running:

```shell
open htmlcov/index.html
```

-**Important**: To prevent issues with `pytest-xdist` failing on `capture_logs` [requests](https://www.structlog.org/en/stable/testing.html) and leaking state in test environment, we've disabled `cache_logger_on_first_use=False` for testing purposes to preserve event state.

### Running a single test file

Tests run in parallel by default for speed. If you need to debug a specific test, it's recommended to run it
individually. To run a single test file, use the following command:

```shell
poetry run pytest -vv -W default {name of test file}
```

If you run into an error `command not found: pytest` try to run `make install-test-deps` first. This command will install `pytest` and will make your shell ready to run the tests.

## Integration test

Integration tests verify that the API endpoints correctly handle requests and return appropriate responses.
Unlike unit testing, integration tests send requests to the server as a black-box component without mocking application code.

- Integration tests are located at `integration_tests` directory.
- Integration tests are slower than unit tests. Adding too many test cases could result in slower CI pipelines, which could affect productivity negatively.
- According to [the GitLab testing guide](https://docs.gitlab.com/ee/development/testing_guide/testing_levels.html#white-box-tests-at-the-system-level-formerly-known-as-system--feature-tests),
  it's ideal to just cover happy path, unhappy path and regression cases. The rest of the low-level testing should be covered in unit testing.

### Run integration test locally

Update your [application settings](application_settings.md) file:

```shell
# .env

# Enable authentication and authorization by default.
# It can be skipped only when "Bypass-Auth" header is specified. We use it for getting the self-signed JWT at the initial request.
AIGW_AUTH__BYPASS_EXTERNAL=false
AIGW_AUTH__BYPASS_EXTERNAL_WITH_HEADER=true
```

Restart you local AI GW: `poetry run ai_gateway`. You need local AI GW up and running during the test:

```shell
poetry run ai_gateway
```

Open a new terminal and run the following command:

```shell
# Adjust `AI_GATEWAY_URL` for your local AI GW url

export AI_GATEWAY_URL=http://localhost:5052
make test-integration
```

## Agent tests

Agent tests validate Duo Workflow Service agent behavior by executing against a real Anthropic LLM.
Unlike integration tests, agent tests exercise the agent classes directly (for example, `ChatAgent` and `PromptAdapter`)
rather than sending requests through the DWS server API.
Tests can use deterministic assertions (tool calls, response structure) and, optionally,
an LLM-as-judge for semantic validation of response quality.

Here is an example of an agent test:

```python
@pytest.mark.asyncio
async def test_how_many_open_issues(analytics_agent, initial_state, mock_gitlab_client):
    mock_glql_response(mock_gitlab_client, glql_response(SAMPLE_ISSUES, count=42))

    result = await ask_agent(
        analytics_agent, initial_state,
        "How many open issues are there in the gitlab-org group?",
    )

    # Deterministic assertions on tool usage
    result.assert_has_tool_calls().assert_called_tool("run_glql_query")

    # Optional: LLM-as-judge semantic validation
    await result.assert_llm_validates(["Response says 42 open issues"])
```

- Agent tests are located in the `agent_tests/` directory.
- Shared fixtures and helpers (LLM setup, `ask_agent`, LLM-as-judge validator) live at the `agent_tests/` root.
- Agent-specific test suites live in subdirectories (for example, `agent_tests/analytics_agent/`).
- Agent tests require the `ANTHROPIC_API_KEY` environment variable.
- The execution and validation models are configurable via `--execution-model` and `--validation-model` CLI options
  (Anthropic models only).

### Run agent tests locally

Run all agent tests:

```shell
export ANTHROPIC_API_KEY=<your-key>
make test-agents
```

Run tests for a specific agent by setting `AGENT_TEST_DIR`:

```shell
AGENT_TEST_DIR=analytics_agent/ make test-agents
```

To override the models used for execution and validation:

```shell
EXECUTION_MODEL=claude-haiku-4-5-20251001 VALIDATION_MODEL=claude-haiku-4-5-20251001 make test-agents
```

### CI

Agent tests run as manual jobs in CI (for example, `tests:agents:analytics`).
Each agent has its own CI job extending the `.tests:agents` base template.
The jobs require the `ANTHROPIC_API_KEY` CI variable to be configured.

## Code guidelines

- Avoid using [provider overriding](https://python-dependency-injector.ets-labs.org/providers/overriding.html),
  since it can lead to divergences between test and runtime behavior. See
  [this issue](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/511)
  for more details.
- If you need to use a fixture in a test but the value of the fixture is not used within the test itself, use
  `@pytest.mark.usefixtures` instead of adding an unused argument. See
  [the pytest docs](https://docs.pytest.org/en/7.1.x/how-to/fixtures.html#use-fixtures-in-classes-and-modules-with-usefixtures)
  for more details.
