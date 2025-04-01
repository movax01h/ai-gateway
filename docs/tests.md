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

### Running a single test file

Tests run in parallel by default for speed. If you need to debug a specific test, it's recommended to run it
individually. To run a single test file, use the following command:

```shell
poetry run pytest {name of test file}
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

## Prompt Evaluations

We use the
[Centralized Evaluation Framework](https://gitlab.com/gitlab-org/modelops/ai-model-validation-and-research/ai-evaluation/prompt-library)
to evaluate the efficiency and accuracy of our AI features. You can use these tools to validate changes to individual
prompts, locally or on CI, against a given dataset (see
[the eli5 datasets doc](https://gitlab.com/gitlab-org/modelops/ai-model-validation-and-research/ai-evaluation/prompt-library/-/tree/main#step-1-choosing-a-dataset)
for more information on choosing a dataset).

### Running prompt evaluations locally

- Add a `LANGCHAIN_API_KEY` to your `.env` (see
[the eli5 prerequisites doc](https://gitlab.com/gitlab-org/modelops/ai-model-validation-and-research/ai-evaluation/prompt-library/-/tree/main/doc/eli5#prerequisites)
on instructions on how to gain access to LangSmith)
- Run `poetry run eval [prompt-id] [prompt-version] [dataset-name]` (for example:
`poetry run eval generate_description 1.0.0 dataset.generate_description.1`)

### Running prompt evaluations on CI

Each pipeline has a manual `tests:evaluation` job. You can start this job from a Merge Request pipeline to validate
prompt changes before merging them. You'll need to supply the following CI variables:

- `PROMPT_ID`
- `PROMPT_VERSION`
- `DATASET`

In the job output, look for the message "View the evaluation results for experiment" to get a link to the resulting
LangSmith run.

For an example, see [this CI job](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/jobs/9534511015),
which points to [this LangSmith experiment](https://smith.langchain.com/o/477de7ad-583e-47b6-a1c4-c4a0300e7aca/datasets/727e9927-ca44-46a1-83c0-09c59e57d081/compare?selectedSessions=ef174a89-8d5e-403c-b80b-2f30af2d225d),
where you can see the outputs produced by the evaluation and compare them to the reference outputs in the dataset. You
can also see other metrics related to the run, like latency, token usage, and cost estimation. In the future we'll
expand this setup to support LLM judges that can automatically evaluate the adequateness of the responses.

## Code guidelines

- Avoid using [provider overriding](https://python-dependency-injector.ets-labs.org/providers/overriding.html),
since it can lead to divergences between test and runtime behavior. See
[this issue](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/511)
for more details.
