# Security fixes

The purpose of this page is to guide GitLab engineers in preparing security fixes, deploying to GitLab-managed AIGW/DWS fleet and releasing for self-hosted Duo customers.

## Overview

- This process is based on [Patch release runbook for GitLab engineers: Preparing security fixes for a patch release:](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/security/engineer.md) and customized for the AI Gateway project. When in doubt, follow the original doc.
- It's required to manually synchronize canonical and security repos such as resolving merge conflicts.
- Backports **must** be prepared and merged in the [security fork](https://gitlab.com/gitlab-org/security/modelops/applied-ml/code-suggestions/ai-assist) **before** any canonical sync. See [Process](#process) for details.
- Automations such as `@gitlab-release-tools-bot` are not supported.

## Due Date

The due date for preparing security issues and MRs follows the due date of the security issue.

## DO NOT PUSH TO `gitlab-org/modelops/applied-ml/code-suggestions/ai-assist`

As an engineer working on a fix for a security vulnerability, your main concern
is not disclosing the vulnerability or the fix before we're ready to publicly
disclose it.

To that end, you'll need to be sure that security vulnerabilities:

- For GitLab AI Gateway and Duo Workflow Service, are fixed in the [AIGW Security Repo](https://gitlab.com/gitlab-org/security/modelops/applied-ml/code-suggestions/ai-assist).

This is fundamental to our patch release process because Security repositories are not publicly-accessible.

## Preparation

### Making sure the issue needs to follow the security workflow

- Verify if the issue you're working on `gitlab-org/modelops/applied-ml/code-suggestions/ai-assist` is confidential, if it's public, fix should be placed on AIGW canonical and no backports are required.
- If the issue you're fixing doesn't appear to be something that can be exploited by a malicious person and is instead simply a security enhancement do not hesitate to mention `@gitlab-com/gl-security/product-security/psirt-group` in the issue to discuss whether the fix can be done in a public MR, in the canonical repository.
- If you're updating a dependency that has a known vulnerability that isn't exploitable in GitLab or has very low severity feel free to engage `@gitlab-com/gl-security/product-security/psirt-group` in the related issue to see if the dependency can be updated in the canonical repository.

### Preparing the repository

Before starting, add the new security remote on your local AIGW repository:

```shell
git remote add security git@gitlab.com:gitlab-org/security/modelops/applied-ml/code-suggestions/ai-assist.git
```

## Creating security branches with proper tracking

When creating your security branch, use the `--track` argument To create a security branch that tracks the security remote branch:

1. Fetch from the security remote branch:

   ```shell
   git fetch security
   ```

1. Create and check out a new branch tracking `security/main`:

   ```shell
   git checkout -b security-fix-vulnerability-name --track security/main
   ```

   For backports, track the specific stable branch:

   ```shell
   git checkout -b security-fix-vulnerability-name-18-4 --track security/stable-18-4-ee
   ```

1. When pushing your changes for the first time, use the -u flag to set the upstream:

   ```shell
   git push -u security security-fix-vulnerability-name
   ```

## Process

While most of the process is same with [the original process](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/security/engineer.md?plain=0#process),
there are a couple of additional steps required for AIGW project.

> **Key principle — security fork first, canonical last.**
> All work (main fix **and** all backports) must be completed and merged in the
> [security fork](https://gitlab.com/gitlab-org/security/modelops/applied-ml/code-suggestions/ai-assist)
> before anything is synced to the canonical repository. This prevents premature
> vulnerability disclosure: the vulnerability is only publicly visible once the
> canonical sync lands, at which point every supported version is already patched.

Once an eligible confidential security issue is assigned to an engineer:

1. **Verify stable branches exist in the security fork.**
   Before opening any backport MRs, confirm that the target stable branches
   (e.g. `stable-18-4-ee`) already exist in the security fork. If they do not,
   create them by branching from the corresponding canonical stable branch and
   pushing to the security remote:

   ```shell
   git fetch origin stable-18-4-ee
   git checkout -b stable-18-4-ee --track origin/stable-18-4-ee
   git push -u security stable-18-4-ee
   ```

1. **Open all MRs in the security fork and get approvals.**
   Steps 1 to 4 of [the original process](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/security/engineer.md?plain=0#process) apply, with the following AIGW-specific clarifications:
   - The MR targeting `main` **and** all backport MRs must be opened in the
     **security fork** (`gitlab-org/security/modelops/applied-ml/code-suggestions/ai-assist`),
     not in the canonical repository.
   - Backport MRs target the stable branches of the security fork
     (e.g. `stable-18-4-ee` in the security repo), using branches created as
     described in [Creating security branches with proper tracking](#creating-security-branches-with-proper-tracking).
   - PSIRT approval is not required for backport MRs. PSIRT approval is only
     required for the MR targeting the `main` branch.

1. **Merge all security-fork MRs.**
   Once the MR targeting `main` and all backport MRs are approved, merge them
   **all in the security fork**. At this point the branches
   (e.g. `main`, `stable-18-4-ee`) are diverged between canonical and security
   repos — the following steps bring them back in sync.

1. **Activate the merge-train keep-alive schedule.**
   Merging the security fix in the previous step diverges `main` (and any
   affected stable branches) between the canonical repository and the security
   fork. The push mirror then rejects new canonical commits until the
   **Sync security fork → canonical** step below brings the two back together.

   As described in [#2370](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/2370),
   this can silently prevent Runway deployments from the security fork for
   several days.

   In the security fork, go to **Build > Pipeline schedules** and activate the
   merge-train schedule. The schedule sets `MERGE_TRAIN` to `true`, which runs
   the `merge-train-trigger` job in `.gitlab-ci.yml`. That job triggers
   [`gitlab-org/merge-train`](https://gitlab.com/gitlab-org/merge-train), which
   copies new canonical commits into `security/main` on every run.

   Deactivate the schedule again in the **Sync security fork → canonical** step,
   once the sync back to canonical completes. Leaving it active only wastes
   pipeline minutes; it does not create a disclosure risk, because commits flow
   from canonical into the security fork, never the other way.

   > The schedule and its `MERGE_TRAIN` variable are both created by hand on the
   > security fork, and activating and deactivating it is manual too. The
   > upstream `release-platform` component can automate that, but only off a
   > `vX.Y.Z` tag, and our patch tags are `self-hosted-vX.Y.Z-ee`. The
   > security-to-canonical sync stays manual for the same reason. See
   > [#2370](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/2370)
   > for background.

1. **Cut the private patch tags in the fork.**
   Once every backport MR for this release is merged, open the latest pipeline
   of each stable branch in the security fork and run the manual
   `tag-stable-patch` job ([`scripts/tag_stable_patch.py`](../../scripts/tag_stable_patch.py)).
   It creates the next patch tag (e.g. `self-hosted-v18.4.2-ee`) **in the
   security fork**, where it stays private. The tag pipeline builds the
   self-hosted images into the fork's private container registry, so the
   patched artifacts exist before disclosure. Nothing is published yet: the
   Docker Hub jobs in that pipeline are also manual (see the publish step
   below).

   The job is manual in the fork (it is automatic in canonical) so that one
   release can group several backport MRs — run it **once per branch**, after
   the last MR merges. Cutting more than one fork tag per branch between syncs
   desynchronizes the patch numbers from the canonical re-cut.

   > This requires the `AIGW_TAGGING_ACCESS_TOKEN` CI/CD variable to be set in
   > the security fork, holding a token that can create tags **on the fork**
   > (the canonical token is group-inherited and does not reach the
   > `gitlab-org/security` subtree). If it is not set, the job does not run and
   > the patch tag is only cut in canonical after the sync.
   >
   > If the variable is marked **Protected**, note that the fork currently
   > protects `*-stable` branches, not `stable-*-ee` — a protected variable is
   > invisible on the stable-branch pipelines where this job runs. Either add a
   > `stable-*` protected-branch rule in the fork or create the variable
   > unprotected.

1. **Ensure the security patch is deployed to the GitLab-managed fleet.**
   Check the post-merge pipeline of the `main`-targeting MR in the security fork.
   Verify the `[duo-workflow-svc]` job in the `runway_production` stage succeeded.
   **Do NOT proceed to the next steps until you've confirmed it.**

1. **Publish the patched images (disclosure starts here).**
   In the security-fork tag pipeline of each patch tag, run the manual
   `release-docker-hub-image:self-managed-tag` and
   `release-docker-hub-self-hosted-fips-image:tag` jobs. These push the
   pre-built images to Docker Hub, so self-hosted users can pull the patched
   version the moment the code becomes public. A published image can be diffed
   against the previous patch, so only run these jobs when you are ready to
   complete the canonical sync immediately afterwards.

   > These jobs need the `DOCKERHUB_USERNAME` and `DOCKERHUB_PASSWORD` CI/CD
   > variables in the security fork. They are project variables in canonical
   > and are not inherited by the fork.
   >
   > Do not retry the stable-branch pipeline's build jobs after the tag
   > pipeline ran: they overwrite the per-commit registry image that these
   > publish jobs pull. If that happened, re-run the tag pipeline's build jobs
   > before publishing.

1. **Sync security fork → canonical.**
   Open sync MRs in [the canonical project](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/1703)
   for every branch that was updated in the security fork. For example:
   - Open a new MR targeting `main` of the canonical repository from `main` of the security repository.
   - Open a new MR targeting `stable-18-4-ee` of the canonical repository from `stable-18-4-ee` of the security repository.
   - Merge all of these MRs.

   Merging the sync MR for a stable branch triggers `tag-stable-patch` in
   canonical, which cuts the same patch tag name there. The sync MR merges
   with a merge commit, so the canonical tag can point at a different commit
   than the fork's tag, with identical content. The canonical tag pipeline
   publishes the canonical GitLab Container Registry images (including the
   cosign-signed FIPS image used by Dedicated) and re-pushes the Docker Hub
   tags with a functionally identical rebuild. If repository mirroring later
   reports a divergence on the tag, delete the tag in the security fork; the
   mirror recreates it from canonical.

   Once `canonical:main` contains `security:main` again, deactivate the
   merge-train keep-alive schedule in **Build > Pipeline schedules** on the
   security fork. The embargo window is over, and the ordinary push mirror
   can take back over.

1. **Verify the release artifacts.**
   After the sync MRs are merged and the canonical tag pipelines complete,
   confirm the patched images are available. See
   [How to backport a fix](./release.md#how-to-backport-a-fix) for details on
   verifying images in the GitLab Container Registry and DockerHub.

1. **Verify repository mirroring.**
   In [the repository settings](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/settings/repository#js-push-remote-settings)
   (**Maintainer** role required), under **Mirroring repositories**, verify that
   mirroring between the security and canonical repositories succeeded.

1. If the mirroring is failing with an error like `Some refs have diverged...`, use the below shell commands to bring the two repos into sync. Note that squash commits should not be enabled on the resulting MR.

```shell
# Example for <branch>, which could be 'main', 'stable-19-0-ee', etc
git checkout -b <branch_name> origin/<branch>
git merge security/<branch> -m "chore: sync security fork into canonical"
git push origin <branch_name>
```

- **NOTE:**
  - If a prior sync merged canonical into the security repo, merging would drag that pollution into canonical.
    Instead, cherry-pick only the real fix commits onto canonical, then force-push canonical over the security
    mirror as a last-resort reset. See [this issue](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/2153) for an example.
    Related information can be found at [How to sync Security repository with Canonical repository](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/security/how_to_sync_security_with_canonical.md).
  - You could encounter a merge conflict at the **Sync security fork → canonical** step if other developers changed the same code.
    You need to manually fix the merge conflict and ask a maintainer to merge it.
  - Other developers may notice their change is not deployed to production because of a mirroring failure due to a merge conflict.
    This can happen if they changed the same code while you were working on the steps from **Merge all security-fork MRs** through **Publish the patched images**.
    To resolve this, finish the **Sync security fork → canonical** step and ask them to rebase their feature branches.

## References

- [Patch release runbook for GitLab engineers: Preparing security fixes for a patch release:](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/security/engineer.md)
- [How to sync Security repository with Canonical repository?](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/security/how_to_sync_security_with_canonical.md)
- [GitLab release and maintenance policy](https://docs.gitlab.com/policy/maintenance/)
- [Note on AI security fixes](https://gitlab.com/gitlab-com/content-sites/internal-handbook/-/blob/main/content/handbook/security/product_security/application_security/_index.md#note-on-ai-security-fixes)
