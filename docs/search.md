# Semantic search for public data

You can use `POST /v1/search/(search-app-name)` endpoint in AI Gateway for performing semantic search on public data.

- [Architecture overview](https://docs.gitlab.com/ee/architecture/blueprints/gitlab_rag/vertex_ai_search.html)
- [Implementation details](https://docs.gitlab.com/ee/architecture/blueprints/gitlab_duo_rag/vertex_ai_search.html)

## Data ingestion and refreshing

### GitLab documentation on GitLab hosted AI Gateway

Input:

| Key                             | Type   | Example                                    |
| ------------------------------- | ------ | ------------------------------------------ |
| `GCP_PROJECT_NAME`              | String | `ai-enablement-dev-69497ba7`               |
| `SEARCH_APP_NAME`               | String | `gitlab-docs`                              |
| `GITLAB_DOCS_REPO`              | String | `https://gitlab.com/gitlab-org/gitlab.git` |
| `GITLAB_DOCS_REPO_REF`          | String | `master`                                   |
| `GITLAB_DOCS_CLONE_DIR`         | String | `/tmp/gitlab-org/gitlab`                   |
| `GITLAB_DOCS_JSONL_EXPORT_PATH` | String | `/tmp/gitlab-org/gitlab/docs.jsonl`        |
| `GITLAB_DOCS_WEB_ROOT_URL`      | String | `https://gitlab.com/help`                  |

Process:

1. Download GitLab doc from `GITLAB_DOCS_REPO` and `GITLAB_DOCS_REPO_REF` to `GITLAB_DOCS_CLONE_DIR`.
1. Parse docs and export as JSONL file to `GITLAB_DOCS_JSONL_EXPORT_PATH`.
1. Read the GitLab version from `VERSION` file. For example, `17.0.0-pre`.
1. Convert the GitLab version to datastore version `<major>-<minor>`. For example, `17.0.0-pre` becomes
   `17-0`.
1. Construct the data store ID `SEARCH_APP_NAME-<datastore_version>`. App name is same with this. For example,
   `gitlab-docs-17-0`.`
1. Construct the BigQuery table ID `GCP_PROJECT_NAME.<data_store_id>.<YYYY-MM-dd-HH-mm-ss>`. For example,
   `ai-enablement-dev-69497ba7.gitlab_docs_17_0.2024-04-25-12-43-30`.
1. Create a new BigQuery table from the JSONL file exported to `GITLAB_DOCS_JSONL_EXPORT_PATH`.
1. Create a new data store in Vertex AI Search that imports the BigQuery table. If the data store already
   exists, import the BigQuery table with incremental or full update. See
   <https://cloud.google.com/generative-ai-app-builder/docs/refresh-data>.
   If the import fails and this run created the data store, delete the data store. An empty data store returns
   no results, and it stops the [fallback to a stable data store](#search-api-in-ai-gateway).
1. Create a new search app in Vertex AI Search that connects to the data store. If the search app already exists, skip this step.

As a result, the search app `gitlab-docs-17-0` is connected to the data store `gitlab-docs-17-0`, which was imported from the bigquery table `ai-enablement-dev-69497ba7.gitlab_docs_17_0.2024-04-25-12-43-30`.

A few notes:

- Vertex AI Search was renamed to Agent Builder.
- In AI Gateway project, CI/CD scheduled pipeline runs the ingestion to `ai-enablement-dev-69497ba7` GCP project 03:00 AM UTC every day.
  See `ingest:dev` job in the pipelines.

### Production ingestion

The same scheduled pipeline also ingests the documentation into the production GCP projects. Each job uses the
GCP project from the `INGEST_GCP_PROJECT_NAME` CI/CD variable of its environment.

| Job               | Environment       | GCP project                    | Read by                                                  |
| ----------------- | ----------------- | ------------------------------ | -------------------------------------------------------- |
| `ingest:dev`      | `ingest/dev`      | `ai-enablement-dev-69497ba7`   | Local development                                        |
| `ingest:prod`     | `ingest/prod`     | `gitlab-ai-framework-prod`     | Duo Workflow Service in production (Cloud Run and EKS)   |
| `ingest:leg-prod` | `ingest/leg-prod` | `gitlab-ai-framework-leg-prod` | AI Gateway in production (GitLab.com, Cloud Run and EKS) |

The three jobs run in parallel and do not depend on each other. If one GCP project fails, the import into the
other projects still runs.

The services read the GCP project from `AIGW_GOOGLE_CLOUD_PLATFORM__PROJECT` in their Runway configuration. For
example, see `.runway/ai-gateway/env-production.yml` and `.runway/ai-gateway-eks/eks-service-production.yaml`.

#### Run production ingestion manually

To ingest without waiting for the schedule:

1. Go to **Build > Pipelines** in the AI Gateway project and select **New pipeline**.
1. Select the `main` branch. The `build:ingest-image` job does not run on other branches, and the ingest jobs need
   its image.
1. Add the variable `GITLAB_INGEST_TRIGGER` with the value `dev`, `prod`, or `leg-prod`.
1. Select **New pipeline**.

The pipeline runs `ingest:prod:trigger` or `ingest:leg-prod:trigger` automatically. For `dev`, start the
`ingest:dev:trigger` job manually.

This starts a full `main` pipeline, so other jobs that run on `main` can start too. Check the pipeline and
cancel any jobs you don't need.

#### Fallback data store version

If the data store for the requested GitLab version does not exist, or the version can't be parsed, AI Gateway and
Duo Workflow Service retry the search with `AIGW_VERTEX_SEARCH__FALLBACK_DATASTORE_VERSION`. Other Vertex AI
Search errors are not retried and return HTTP 500. The value is a GitLab `<major>.<minor>` version, for example
`18.9`.

Several Cloud Run and EKS configurations under `.runway/` set this value. To find all of them, run
`git grep AIGW_VERTEX_SEARCH__FALLBACK_DATASTORE_VERSION .runway/`. Staging and production have their own values,
except `.runway/duo-workflow-svc-eks/default-values.yaml`, which applies to both. When you change the value for an
environment, update every file of that environment. Make sure that the data store for that
version exists in the GCP project of that environment. No ingest job writes to the staging GCP projects, so the
staging value can differ from the production value.

Exceptions:

- Duo Workflow Service on EKS staging reads `gitlab-ai-framework-dev` but uses the production fallback value from
  `default-values.yaml`. When you bump that value, make sure the data store also exists in
  `gitlab-ai-framework-dev`, or override the value in `.runway/duo-workflow-svc-eks/eks-service-staging.yaml`.
- `ai-gateway-custom` runs the self-hosted image with `AIGW_CUSTOM_MODELS__ENABLED=true`. It searches a local SQLite
  index instead of Vertex AI Search, so it does not use the fallback version.

### Run locally

Login to the GCP via Application Default Credentials (ADC):

```shell
gcloud auth application-default login
gcloud auth application-default set-quota-project ai-enablement-dev-69497ba7
```

Run the following script in terminal:

```shell
export GCP_PROJECT_NAME="ai-enablement-dev-69497ba7"
export SEARCH_APP_NAME="gitlab-docs"
export GITLAB_DOCS_REPO="https://gitlab.com/gitlab-org/gitlab.git"
export GITLAB_DOCS_REPO_REF="master"
export GITLAB_DOCS_CLONE_DIR="/tmp/gitlab-org/gitlab"
export GITLAB_DOCS_JSONL_EXPORT_PATH="/tmp/gitlab-org/gitlab/docs.jsonl"
export GITLAB_DOCS_WEB_ROOT_URL="https://gitlab.com/help"

make ingest
```

or use Docker image:

```shell
docker run \
    -e 'GCP_PROJECT_NAME=ai-enablement-dev-69497ba7' \
    -e 'SEARCH_APP_NAME=gitlab-docs' \
    -e 'GITLAB_DOCS_REPO=https://gitlab.com/gitlab-org/gitlab.git' \
    -e 'GITLAB_DOCS_REPO_REF=master' \
    -e 'GITLAB_DOCS_CLONE_DIR=/tmp/gitlab-org/gitlab' \
    -e 'GITLAB_DOCS_JSONL_EXPORT_PATH=/tmp/gitlab-org/gitlab/docs.jsonl' \
    -e 'GITLAB_DOCS_WEB_ROOT_URL=https://gitlab.com/help' \
    -e 'GOOGLE_APPLICATION_CREDENTIALS=/gcloud-config/application_default_credentials.json' \
    -v "$HOME/.config/gcloud/application_default_credentials.json:/gcloud-config/application_default_credentials.json" \
    registry.gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/ingest:<SHA>
```

### Test search app in GCP console

You can test a search app in [GCP console](https://console.cloud.google.com/home).

1. Visit **"Your GCP project" > Agent Builder > Apps > "Your app name" > Preview** in GCP.
1. Test a query if it returns relevant chunks.

Here is a [preview app for GitLab documentations](https://console.cloud.google.com/gen-app-builder/locations/global/engines/gitlab-docs-17-0/preview/search?project=ai-enablement-dev-69497ba7).

### Further iteration

- [Generalize ingestion process for any public data](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/446).
- [Replace Ruby parser by Python parser](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/447).

## Search API in AI Gateway

1. `POST /v1/search/(search-app-name)` is requested.
1. If the search app depends on GitLab version:
   1. Get the GitLab version from `metadata.version` parameter.
   1. Convert the GitLab version to datastore version `<major>-<minor>`. For example, `17.0.0-pre` becomes
      `17-0`.
   1. Construct the data store ID `<name>-<version>`. App name is same with this. For example,
      `gitlab-docs-17-0`.
1. Request to the Vertex AI Search with the data store ID.
   1. If the data store is not found, or the GitLab version can't be parsed, it falls back to a stable data store.
      For example, the data store ID doesn't exist yet when GitLab-Rails bumps the `VERSION` and it is deployed to
      GitLab.com before the ingestion pipeline has run. See
      [fallback data store version](#fallback-data-store-version). Other errors return HTTP 500 without a fallback.

## Check usage quota in GCP console

You can check usage quota in [GCP console](https://console.cloud.google.com/home).

1. Visit **"Your GCP project" > API & Services > Discovery Engine API > QUOTAS & SYSTEM LIMITS** in GCP.
1. Check **Current usage percentage**. If the value is saturated, [search API in AI Gateway](#search-api-in-ai-gateway) or [ingestion process](#data-ingestion-and-refreshing) could fail.

Here is a [quota and usage percentage for `ai-enablement-dev-69497ba7` GCP project](https://console.cloud.google.com/apis/api/discoveryengine.googleapis.com/quotas?project=ai-enablement-dev-69497ba7).

## GitLab documentation on self-hosted AI Gateway

Customer that self-hosted their AI Gateway instance do not have access to GitLab VertexAI instance. As an alternative
way of searching for documentation, the AI Gateway self-hosted image already includes embedded search on tags `self-hosted-17.4.0-ee` and
later.

To achieve this, we use [a full-text search using bm25 on a sqlite database](https://www.sqlite.org/fts5.html). The corpus
is generated when building the Docker image and is updated for every version, users are not required to take any action.
This mechanism is light and simple, but the results are not robust towards user query. It is recommended to strip away unnecessary
words and pass only the intent. For example, instead of querying `How can I create an issue`, better results are obtained
by querying `create issue`.

To generate the corpus, run:

```shell
poetry run index_docs -o <tmp/output-database-path> -v <gitlab-version-tag>
```
