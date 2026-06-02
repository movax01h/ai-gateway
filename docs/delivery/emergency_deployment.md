# Emergency deployment

Use this runbook to push a fix to the GitLab-managed AIGW/DWS fleet faster than the normal canary rollout when responding to a production incident.

## Overview

- Production deploys normally use Runway's default `legacy` strategy: a global canary that shifts traffic in steps (25%, 50%, 75%, 100%). This is the safest option and should be used whenever possible.
- For incident response we can switch to the `expedited` strategy, which rolls out to 100% of traffic globally in a single "big bang" deployment. See the Runway [deployment strategies](https://docs.runway.gitlab.com/runtimes/cloud-run/deployment-strategies/#setting-a-deployment-strategy) and [incident response](https://docs.runway.gitlab.com/runtimes/cloud-run/incident-response/#emergency-deployment) docs.
- The strategy is controlled by the `RUNWAY_DEPLOYMENT_STRATEGY` CI/CD variable, which takes precedence over `spec.deployment.strategy` in `.runway/runway.yml`. The variable is already wired into the Runway deploy jobs, so no `.gitlab-ci.yml` change is needed to use it.
- **Set it as a project CI/CD settings variable, not as a `.gitlab-ci.yml` `variables:` entry.** The Runway deploy jobs run with `inherit: variables: false` (staging) and a restricted inherit list (production), so they do not pick up globally-defined `.gitlab-ci.yml`/`workflow` variables. Project CI/CD settings variables are always passed to jobs regardless of `inherit:variables`, so that is the method that reaches the deploy and is forwarded to the downstream Runway pipeline.
- Only Owners and Maintainers can manage project CI/CD variables, so enabling an emergency deployment is restricted to Owners/Maintainers by the GitLab permission model. Variable changes are recorded in the project audit events.

## When to use this

Reserve `expedited` for genuine emergencies where the canary rollout is too slow — for example, shipping a fix for a breaking production issue. The `legacy` strategy receives the most testing and is the most familiar, so prefer it for routine deploys.

## Procedure

### 1. Enable (Owner/Maintainer only)

In **Settings → CI/CD → Variables**, add:

| Key | Value | Flags |
| --- | --- | --- |
| `RUNWAY_DEPLOYMENT_STRATEGY` | `expedited` | **Protected** (applies to protected refs only), unmasked |

The value is not a secret, so it does not need to be masked. Protecting it limits the variable to the default branch and protected tags.

This single variable applies to **both staging and production** deploys, because both deploy jobs forward `RUNWAY_DEPLOYMENT_STRATEGY` to the downstream Runway pipeline.

### 2. Deploy

- Land the fix on the default branch. Production deploys automatically on merge to the default branch unless `RUNWAY_PRODUCTION_AUTO_DEPLOY` is set to `false`, in which case the production deploy is a manual job you trigger from the pipeline. This approval gate is independent of the deployment strategy.
- Alternatively, re-run the staging and `🛫 [...] Trigger runway deployment production` jobs on the latest default-branch pipeline.
- When the CI pipeline itself is the bottleneck (not just the rollout), add `pipeline_expedited: true` to the commit message to skip the lint, test, performance, ingest, and release jobs.

### 3. Disable (required cleanup)

After the incident, an Owner/Maintainer **must remove** the `RUNWAY_DEPLOYMENT_STRATEGY` variable. It is sticky — it stays in effect for every default-branch deploy until removed. Removing it returns deploys to the safe defaults: `legacy` for production and `expedited` for staging (staging is `expedited` by default via `.runway/runway-staging.yml`).

Treat removal as a mandatory incident closeout step.

## Optional hardening

To also restrict who can trigger manual re-runs of the deploy jobs, configure [Protected Environments](https://docs.gitlab.com/ci/environments/protected_environments/) for `production` and `staging` with deploy access limited to Maintainers. This is safe to add: the dry-run jobs use `environment: action: verify`, so they are not treated as deployments and remain available to other roles.

## Related docs

- [Release](./release.md)
- [Security fixes](./security_fixes.md)
- [Delivery process overview](./index.md)
