# GitLab AI Gateway

GitLab AI Gateway is a standalone-service that provides access to LLM providers, authentication, model selection, and related features.

## API

See [API](../docs/api.md).

## Prerequisites

You'll need:

- Docker
- `docker compose` >= 1.28
- [`gcloud` CLI](https://cloud.google.com/docs/authentication/provide-credentials-adc#how-to)
- sqlite development libraries
  - This package is usually called `libsqlite3-dev` or `sqlite-devel` (depending on your platform);
    install this _before_ installing Python so it can compile against these libraries.
- `mise` for version management
  - To install `mise`, see [instructions here](https://mise.jdx.dev/getting-started.html).

### Google Cloud SDK

Set up a Google Cloud project with access to the Vertex AI API and authenticate to it locally by following [these instructions](https://docs.gitlab.com/ee/development/ai_features/#gcp-vertex).

### Frameworks

This project is built with the following frameworks:

1. [FastAPI](https://fastapi.tiangolo.com/)
1. [Dependency Injector](https://python-dependency-injector.ets-labs.org/introduction/di_in_python.html)

### Project architecture

This repository follows [The Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html) paradigm,
which define layers present in the system as well as their relations with each other, please refer to the linked article for more details.

### Project structure

For the Code Suggestions feature, most of the code is hosted at `/ai_gateway`.
In that directory, the following artifacts can be of interest:

1. `app.py` - main entry point for web application
1. `code_suggestions/processing/base.py` - that contains base classes for ModelEngine.
1. `code_suggestions/processing/completions.py` and `suggestions/processing/generations.py` - contains `ModelEngineCompletions` and `ModelEngineGenerations` classes respectively.
1. `api/v2/endpoints/code.py` - that houses implementation of main production Code Suggestions API
1. `api/v2/experimental/code.py` - implements experimental endpoints that route requests to fixed external models for experimentation and testing

Middlewares are hosted at `ai_gateway/api/middleware.py` and interact with the `context` global variable that represents the API request.

## Application settings

See [Application settings doc](../docs/application_settings.md)

## How to run the server locally

1. Clone project and change to project directory.
1. Run `mise install` to install all required tools.
1. [Install the `poetry-plugin-shell`](https://github.com/python-poetry/poetry-plugin-shell?tab=readme-ov-file#installation)
1. Init shell: `poetry shell`.
1. [Activate virtualenv](../README.md#how-to-manually-activate-the-virtualenv).
1. Install dependencies: `poetry install`.
1. Copy the `example.env` file to `.env`: `cp example.env .env`
1. Update the `.env` file in the root folder with the following variables:

   ```shell
   ANTHROPIC_API_KEY=<API_KEY>
   ```

1. You can enable hot reload by setting the `AIGW_FASTAPI__RELOAD` environment variable to `true` in the `.env` file.
1. Ensure you're authenticated with the `gcloud` CLI by running `gcloud auth application-default login`.
1. Start the model-gateway server locally: `poetry run ai_gateway`.
1. Open `http://localhost:5052/docs` in your browser and run any requests to the model.

### Mocking AI model responses

If you do not require real models to run and evaluate the input data, you can mock the model responses
by setting the environment variable `AIGW_MOCK_MODEL_RESPONSES=true`.
The models will start echoing the given prompts, while allowing you to run a fully functional AI gateway.

This can be useful for testing middleware, request/response interface contracts, logging, and other
uses cases that do not require an AI model to execute.

Agentic Chat can be mocked by setting the environment variables `AIGW_USE_AGENTIC_MOCK=true` and `AIGW_MOCK_MODEL_RESPONSES=true`. You can specify a sequence of responses to simulate a multi-step flow.
See the [documentation](docs/workflows/agentic_mock.md) for details.

### Logging requests and responses during development

AI Gateway workflow includes additional pre and post-processing steps. By default, the log level is `INFO` and
application writes log to `stdout`. If you want to log data between different steps for development purposes
and to a file, please update the `.env` file by setting the following variables:

```shell
AIGW_LOGGING__LEVEL=debug
AIGW_LOGGING__TO_FILE=../modelgateway_debug.log
```

## Local development using GDK

### Prerequisites

Make sure you have credentials for a Google Cloud project (with the Vertex API enabled) located at `~/.config/gcloud/application_default_credentials.json`.
This should happen automatically when you run `gcloud auth application-default login`. If for any reason this JSON file is at a
different path, you will need to override the `volumes` configuration by creating or updating a `docker-compose.override.yaml` file.

### Running the API

You can either run `make develop-local` or `docker-compose -f docker-compose.dev.yaml up --build --remove-orphans`.
If you need to change configuration for a Docker Compose service, you can add it to `docker-compose.override.yaml`.
Any changes made to services in this file will be merged into the default settings.

Next open the VS Code extension project, and run the development version of the GitLab Workflow extension locally. See [Configuring Development Environment](https://gitlab.com/gitlab-org/gitlab-vscode-extension/-/blob/main/CONTRIBUTING.md#configuring-development-environment) for more information.

In VS Code code, we need to set the `MODEL_GATEWAY_AI_ASSISTED_CODE_SUGGESTIONS_API_URL` constant to `http://localhost:5000/completions`.

Since the feature is only for SaaS, you need to run GDK in SaaS mode:

```shell
export GITLAB_SIMULATE_SAAS=1
gdk restart
```

Then go to `/admin/application_settings/general`, expand `Account and limit`, and enable `Allow use of licensed EE features`.

You also need to make sure that the group you are allowing, is actually `ultimate` as it's an `ultimate` only feature,
go to `/admin/groups` select `Edit` on the group you are using, set `Plan` to `Ultimate`.

## Component overview

### Client

The Client has the following functions:

1. Determine input parameters.
   1. Stop sequences.
   1. Gather code for the prompt.
1. Send the input parameters to the AI Gateway API.
1. Parse results from AI Gateway and present them as `inlineCompletions`.

We are supporting the following clients:

- [GitLab VS Code Extension](https://gitlab.com/gitlab-org/gitlab-vscode-extension).
- [GitLab Language Server for Code Suggestions](https://gitlab.com/gitlab-org/editor-extensions/gitlab-language-server-for-code-suggestions).

## Deployment

### For production AI Gateway environments

AI Gateway is continuously deployed to [Runway](https://about.gitlab.com/handbook/engineering/infrastructure/platforms/tools/runway/).

This deployment is currently available at `https://ai-gateway.runway.gitlab.net`.
Note, however, that clients should not connect to this host directly, but use `cloud.gitlab.com/ai` instead,
which is managed by Cloudflare and is the entry point GitLab instances use instead.

When an MR gets merged, CI will build a new Docker image, and trigger a Runway downstream pipeline that will deploy this image to staging, and then production. Downstream pipelines run against the [deployment project](https://gitlab.com/gitlab-com/gl-infra/platform/runway/deployments/ai-gateway).

The service overview dashboard is available at [https://dashboards.gitlab.net/d/ai-gateway-main/ai-gateway-overview](https://dashboards.gitlab.net/d/ai-gateway-main/ai-gateway-overview).

Note that while the runway pods are running in the `gitlab-runway-production` GCP project, all Vertex API calls target the `gitlab-ai-framework-leg-prod` GCP project for isolation purposes. This project is managed [through terraform](https://ops.gitlab.net/gitlab-com/gl-infra/config-mgmt/-/tree/main/environments/ai-framework-leg-prod?ref_type=heads). Monitoring for those calls is provided through [stackdriver-exporter](https://gitlab.com/gitlab-com/gl-infra/k8s-workloads/gitlab-helmfiles/-/tree/master/releases/stackdriver-exporter/ai-framework?ref_type=heads).

### For production Duo Workflow Service environments

Duo Workflow Service is continuously deployed to [Runway](https://about.gitlab.com/handbook/engineering/infrastructure/platforms/tools/runway/).

This deployment is currently available at `https://duo-workflow-svc.runway.gitlab.net`.

When an MR gets merged, CI will build a new Docker image, and trigger a Runway downstream pipeline that will deploy this image to staging, and then production. Downstream pipelines run against the [deployment project](https://gitlab.com/gitlab-com/gl-infra/platform/runway/deployments/duo-workflow-svc).

See the [service overview dashboard](https://dashboards.gitlab.net/d/runway-service/runway3a-runway-service-metrics?var-PROMETHEUS_DS=mimir-runway&var-environment=gprd&var-type=duo-workflow-svc).

Note that while the runway pods are running in the `gitlab-runway-production` GCP project, all Vertex API calls target the `gitlab-ai-framework-prod` GCP project for isolation purposes. This project is managed [through terraform](https://ops.gitlab.net/gitlab-com/gl-infra/config-mgmt/-/tree/main/environments/ai-framework-prod?ref_type=heads). Monitoring for those calls is provided through [stackdriver-exporter](https://gitlab.com/gitlab-com/gl-infra/k8s-workloads/gitlab-helmfiles/-/tree/master/releases/stackdriver-exporter/ai-framework?ref_type=heads).

### For staging-ref

For [staging-ref](https://staging-ref.gitlab.com/) environment, the deployment is powered by [Runway](https://about.gitlab.com/handbook/engineering/infrastructure/platforms/tools/runway/), and is named as `ai-gateway-custom`.

The deployment for staging-ref differs from other production environments in both its nature and configuration. This deployment specifically powers Code Suggestions and Duo Chat when using Custom Models, and may use a different set of secret variables compared to other production deployments. The Group Custom Models team (`#g_custom_models` on Slack) is responsible for managing changes to deployments in this environment and maintains ownership of it.

Important MRs:

- [Enabling runway deployments for Custom Models as `ai-gateway-custom`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/1407)
- [Using the `ai-gateway-custom`'s URL as the AI Gateway endpoint for staging-ref](https://gitlab.com/gitlab-org/quality/gitlab-environment-toolkit-configs/staging-ref/-/merge_requests/190)
- [Add `ai-gateway-custom` to Runway Provisioner](https://gitlab.com/gitlab-com/gl-infra/platform/runway/provisioner/-/merge_requests/399)

For more information and assistance, please check out:

- [Runway - Handbook](https://about.gitlab.com/handbook/engineering/infrastructure/platforms/tools/runway/).
- [Runway - Group](https://gitlab.com/gitlab-com/gl-infra/platform/runway).
- [Runway - Docs](https://gitlab.com/gitlab-com/gl-infra/platform/runway/docs).
- [Runway - Issue Tracker](https://gitlab.com/groups/gitlab-com/gl-infra/platform/runway/-/issues).
- `#f_runway` in Slack.

## Multiple worker processes

By default, the AI Gateway runs a single process to handle HTTP
requests. To increase throughput, you may want to spawn multiple
workers. To do this, there are a number of environment variables that
need to be set:

- `WEB_CONCURRENCY`: The [number of worker processes](https://www.uvicorn.org/deployment/) to run (1 is default).
- `PROMETHEUS_MULTIPROC_DIR`: This is needed to support scraping of [Prometheus metrics](https://prometheus.github.io/client_python/multiprocess/) from a single endpoint.

This directory holds the metrics from the processes and should be cleared before the application starts.

## GitLab Pages Deployment

On every merge to the `main` branch, a GitLab Pages job automatically deploys the following components:

### Prompt directory structure

The prompt directory structure is deployed to [`/prompt_directory_structure`](https://gitlab-org.gitlab.io/modelops/applied-ml/code-suggestions/ai-assist/prompt_directory_structure.json).

This endpoint exposes the available prompt versions for various AI features and model families supported by the AI Gateway. Introduced in [!2139](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/2139).
