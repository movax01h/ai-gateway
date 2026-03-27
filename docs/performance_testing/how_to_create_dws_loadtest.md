# How to create `dws-loadtest` on Runway

## Background

As part of the [Duo Workflow Service (DWS) load testing effort](https://gitlab.com/groups/gitlab-org/quality/-/work_items/201),
we [set up a `dws-loadtest` instance on Runway](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/1313).
During load testing, we found that deploying directly to Google Cloud Run allowed faster experimentation and iteration,
so we [retired the `dws-loadtest` service](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/2040).

Load testing is not a frequent activity right now, but as usage grows it may be worth integrating into CI/CD.
This document is intended to make that process easier.

## Instructions

This follows the [Runway onboarding process](https://docs.runway.gitlab.com/runtimes/cloud-run/onboarding/).

The file links below point to a specific commit before the `dws-loadtest` configuration was removed.
Use them to retrieve the original file contents as a starting point when recreating each file.
They can be used to get the content needed to re-create the files.

1. Create Runway config in [`ai-assist`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist):
   - Add the following files:
     - [`.runway/dws-loadtest/runway.yml`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/0caedbb7d8463b5480cf52eadc93fb56b3051d3a/.runway/dws-loadtest/runway.yml)
     - [`.runway/dws-loadtest/env-staging.yml`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/0caedbb7d8463b5480cf52eadc93fb56b3051d3a/.runway/dws-loadtest/env-staging.yml)
     - [`.runway/dws-loadtest/env-production.yml`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/0caedbb7d8463b5480cf52eadc93fb56b3051d3a/.runway/dws-loadtest/env-production.yml)
   - Edit [`.gitlab-ci.yml`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/0caedbb7d8463b5480cf52eadc93fb56b3051d3a/.gitlab-ci.yml#L127)
   to include the Runway template for `dws-loadtest` and the include for overrides
   (note deploy from security, unlike before. See: <https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/4891+>):

        ```yaml
        - project: "gitlab-com/gl-infra/platform/runway/runwayctl"
          file: "ci-tasks/service-project/runway.yml"
          inputs:
            runway_service_id: dws-loadtest
            image: "$CI_REGISTRY_IMAGE/model-gateway:${CI_COMMIT_SHORT_SHA}"
            runway_version: v4.18.6 # Update this to the same as other runway services. Or check the runwayctl releases for the latest version
        - local: .gitlab/ci/dws-loadtest.gitlab-ci.yml
          rules: *skip-if-not-security # This is different from the initial setup
        ```

   - Create [`.gitlab/ci/dws-loadtest.gitlab-ci.yml`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/0caedbb7d8463b5480cf52eadc93fb56b3051d3a/.gitlab/ci/dws-loadtest.gitlab-ci.yml#L134) with:

       ```yaml
       # Override these jobs defined from
       # project: "gitlab-com/gl-infra/platform/runway/runwayctl"
       # file: "ci-tasks/service-project/runway.yml"

       🛫 [dws-loadtest] Trigger runway deployment staging:
         rules:
           - if: $CI_COMMIT_BRANCH == $CI_DEFAULT_BRANCH
             when: manual
             allow_failure: true # so this job doesn't block the pipeline

       🛫 [dws-loadtest] Trigger runway deployment production:
         rules:
           - when: never # We only need the staging deployment for load testing
       ```

   - [Allow the Runway deployment project to read CI artifacts](https://docs.runway.gitlab.com/runtimes/cloud-run/onboarding/#update-job-token-permissions)
     - Settings → CI/CD → Job token permissions → CI/CD job token allowlist
     - Add the Runway deployment project for this workload: `gitlab-com/gl-infra/platform/runway/deployments/dws-loadtest`

1. Recreate the Runway workload
   - In [`gitlab-com/gl-infra/platform/runway/provisioner`](https://gitlab.com/gitlab-com/gl-infra/platform/runway/provisioner),
   add the `dws-loadtest` workload entry to [`config/runtimes/cloud-run/workloads.yml`](https://gitlab.com/gitlab-com/gl-infra/platform/runway/provisioner/-/blob/61a116a32443b06c865eacab53a8211f409a7ca1/config/runtimes/cloud-run/workloads.yml#L31):

     ```yaml
     - runway_service_id: dws-loadtest
       project_id: 39903947
       regions:
         - us-east1
       groups:
         - gitlab-org/maintainers/ai-gateway
         - gitlab-org/ai-powered/ai-framework/engineering/all
       exclude_name_suffix: true
     ```

1. Configure secrets in Vault for `dws-loadtest` and make sure the SA can access Vertex AI
   - Add the SA to [`environments/ai-framework-stage/service_accounts.tf`](https://ops.gitlab.net/gitlab-com/gl-infra/config-mgmt/-/blob/3d059158aa2751e0ff7744f613e96168b464fc10/environments/ai-framework-stage/service_accounts.tf#L13) if it doesn't already exist:

      ```tf
      resource "google_project_iam_member" "gitlab-dws-loadtest-staging-vertex-ai" {
        project = var.project
        role    = "roles/aiplatform.user"
        member  = "serviceAccount:crun-dws-loadtest@gitlab-runway-staging.iam.gserviceaccount.com"
      }
      ```

   - In [`environments/vault-production/secrets_policies.tf`](https://ops.gitlab.net/gitlab-com/gl-infra/config-mgmt/-/blob/6bc42f24aedf4101f3ccff43cff9f2763b74a225/environments/vault-production/secrets_policies.tf#L450) add the Vault policy (as in <https://ops.gitlab.net/gitlab-com/gl-infra/config-mgmt/-/merge_requests/11947>):

      ```tf
      "env/+/service/dws-loadtest/*" = {
        admin = {
          groups = local.groups.ai_gateway
        }
      }
      ```

   - In [Vault](https://vault.gitlab.net/ui/vault/secrets/runway/kv/list/env/staging/service/dws-loadtest/), create/clone the secret set for `dws-loadtest` from the existing staging `duo-workflow-svc` Runway service. The keys should be:
     - `ANTHROPIC_API_KEY`
     - `DUO_WORKFLOW_SELF_SIGNED_JWT__SIGNING_KEY`
     - `DUO_WORKFLOW_SELF_SIGNED_JWT__VALIDATION_KEY`
     - `LANGCHAIN_API_KEY`
     - `OPENAI_API_KEY`

## References

- <https://docs.runway.gitlab.com/runtimes/cloud-run/onboarding/>
- <https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/3030+>
- <https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/3337+>
- <https://gitlab.com/gitlab-com/gl-infra/platform/runway/provisioner/-/merge_requests/931+>
- <https://ops.gitlab.net/gitlab-com/gl-infra/config-mgmt/-/merge_requests/11947>
- <https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/1442+>
