# Release process for Self-managed AI Gateway

AI Gateway is deployed by the following ways:

- **Self-managed AI Gateway**: Deployed by customers when a new version of AI Gateway is released. The actual deployment method is TBD. See [this epic](https://gitlab.com/groups/gitlab-org/-/epics/13393) for more information.
- **GitLab-managed AI Gateway**: Deployed via Runway when a new commit is merged into `main` branch.

This release process is necessary to ensure compatibility between Self-managed GitLab Monolith and Self-managed AI Gateway.
It's NOT necessary for GitLab-managed AI Gateway as currently Runway deploys the latest SHA.

## Overview

We follow the [Semantic Versioning guideline](https://semver.org/),
which is rendered in [Conventional Commits](https://www.conventionalcommits.org/en) as an actual practice.
To harness the practice, we use [semantic-release](https://github.com/semantic-release/semantic-release) via the [common-ci-tasks](https://gitlab.com/gitlab-com/gl-infra/common-ci-tasks) template and [commitlint](https://github.com/conventional-changelog/commitlint).

In CI pipelines in AI Gateway:

- On merge requests:
  - `lint:commit` job runs to validate the commits in the feature branch if they are following Conventional Commits.
  - `semantic_release_check` job runs to make sure the commits are releasable via semantic-release.
- On `main` branch:
  - `semantic_release_base` job runs automatically to cut a new release and Git tag when conventional commits are detected.
- On Git tags:
  - `release-docker-image:tag` job runs to pushes a new Docker image.
- On Git tags with format `gitlab-*`:
  - `release-docker-hub-image:tag` job runs to push a new Docker image to DockerHub. Requires `$DOCKERHUB_USERNAME` and `$DOCKERHUB_PASSWORD` to be set as CI variables.

In addition, we have [the expectations on backward compatibility](https://docs.gitlab.com/ee/architecture/blueprints/ai_gateway/#basic-stable-api-for-the-ai-gateway).
Tl;dr;

- We keep the API interfaces backward-compatible for the last 2 major versions.
- If a breaking change happens where we don't have a control (e.g. a depended 3rd party model was removed), we try to find a backward-compatible solution otherwise we bump a major version.

## View released versions of AI Gateway

To view released versions of AI Gateway, visit the following links:

- [Releases](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/releases): This page lists the released versions and changelogs.
- [Container Registry](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/container_registry): This page lists the released Docker images e.g. `registry.gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/model-gateway:v1.0.0`
- [DockerHub](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/container_registry): This page lists the released images that match a GitLab version e.g. `docker/gitlab/model-gateway:gitlab-v17.2`

## Self-hosted AI Gateway release process

When a new minor GitLab version is released (vX.Y.0-ee), a new branch is created with the name `stable-{gitlab-major}-{gitlab-minor}-ee`, and new tag with name `self-hosted-vX.Y.0-ee` is created, which triggers the release of a new image. Users on self-hosted environments can use this to download a version of AI Gateway that is compatible with their GitLab installation. These images are available both on
[GitLab container registry](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/container_registry/3809284)
and on [DockerHub](https://hub.docker.com/repository/docker/gitlab/model-gateway/tags).

### Releasing patches to previous versions

Stable branches will not be receiving updates from main branch. If a bug at a version needs to be addressed, the developer can cherry-pick the necessary commits, and request a maintainer to bump the PATCH version and create a new release tag, publishing a new image. This allows AI teams to release fixes without getting blocked by GitLab-rails patch release process.

## GitLab managed AI Gateway Release process

Releases are now automated using the `semantic-release` job from [common-ci-tasks](https://gitlab.com/gitlab-com/gl-infra/common-ci-tasks). The release process works as follows:

1. When commits following [Conventional Commits](https://www.conventionalcommits.org/en) are pushed to the `main` branch, the `semantic_release_base` job automatically runs.
1. This job calculates the next version based on the commit messages, creates a new Git tag, and publishes a release.
1. No manual intervention is required - releases happen automatically when appropriate conventional commits are detected.

The semantic release job uses the configuration defined in `.releaserc.json` and is managed through the common-ci-tasks template.

## Configure release workflow

The release workflow is configured via the `.releaserc.json` file in the project root. This configuration:

- Supports releases from the `main` branch and maintenance branches following the pattern `+([0-9])?(.{+([0-9]),x}).x`
- Uses only the GitLab plugin for creating releases and tags

The semantic-release job is provided by the [common-ci-tasks](https://gitlab.com/gitlab-com/gl-infra/common-ci-tasks) template.

If you want to set up maintenance/backport releases, see [this recipe](https://github.com/semantic-release/semantic-release/blob/master/docs/recipes/release-workflow/maintenance-releases.md).
