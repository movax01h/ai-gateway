# Security fixes

The purpose of this page is to guide GitLab engineers in preparing security fixes, deploying to GitLab-managed AIGW/DWS fleet and releasing for self-hosted Duo customers.

## Overview

- This process is based on [Patch release runbook for GitLab engineers: Preparing security fixes for a patch release:](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/security/engineer.md) and customized for the AI Gateway project. When in doubt, follow the original doc.
- It's required to manually synchronize canonical and security repos such as resolving merge conflicts.
- It's required to manually create [backports and releases](release.md#how-to-backport-a-fix).
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

Once an eligible confidential security issue is assigned to an engineer:

1. Steps 1 to 4 are same. Open MRs and get approvals from a maintainer and an PSIRT team member.
   - As stated in [the original process](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/security/engineer.md?plain=0#process) sections, PSIRT approval is not required for backport MRs. PSIRT approval is only required for the MR targeting the `main` branch.
1. Once the merge request targeting the default branch and all backports are ready, merge these MRs at once.
   At this moment, these branches (e.g. `main`, `stable-18-4-ee`, etc) are diverged between canonical and security repos, so we're going to fix in the following steps.
1. Ensure that the security patch has been deployed to GitLab-managed AIGW and DWS fleet.
   One way to confirm it is to check the post-merge pipeline in the MR that targets `main` branch. Check `[duo-workflow-svc]` job status in `runway_production` stage if it succeeded.
   **Do NOT proceed to the next steps until you've confirmed it.**
1. Open new merge requests in [the canonical project](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/1703) to sync the security repository and the canonical repository. For example,
   - Open new MR that targets `main` branch of the canonical repository from the `main` branch of the security repository.
   - Open new MR that targets `stable-18-4-ee` branch of the canonical repository from the `stable-18-4-ee` branch of the security repository.
   - And merge these MRs.
1. Cut Git-Tags for backports by following [How to backport a fix](./release.md#how-to-backport-a-fix). This will release patched Docker images for self-hosted Duo customers.
1. Done.

- **NOTE:**
  - You could encounter a merge conflict at step 4 if the other developers have changed the same code.
    You need to manually fix the merge conflict and ask a maintainer to merge it.
  - The other developers could notice that their change is not deployed to production because of mirroring failure due to merge conflict.
    This could happen if they changed the same code while you're working on step 2~3.
    To resolve this issue, finish the step 4 and ask them to rebase their feature branches.

## References

- [Patch release runbook for GitLab engineers: Preparing security fixes for a patch release:](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/security/engineer.md)
- [How to sync Security repository with Canonical repository?](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/security/how_to_sync_security_with_canonical.md)
- [GitLab release and maintenance policy](https://docs.gitlab.com/policy/maintenance/)
- [Note on AI security fixes](https://gitlab.com/gitlab-com/content-sites/internal-handbook/-/blob/main/content/handbook/security/product_security/application_security/_index.md#note-on-ai-security-fixes)
