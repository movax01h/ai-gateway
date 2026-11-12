# Security fixes

This page guides GitLab engineers through preparing a security fix for the AI Gateway (AIGW) and Duo Workflow Service (DWS), deploying them to the GitLab Runway deployments, and releasing it to self-hosted Duo customers.

It follows the [patch release runbook for GitLab engineers](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/security/engineer.md). When in doubt, follow the runbook.

## Rules

- **Security fork first, canonical last.** Do all work in the [security fork](https://gitlab.com/gitlab-org/security/modelops/applied-ml/code-suggestions/ai-assist). Nothing reaches canonical until every supported version is patched.
- **Never push a security fix to canonical** (`gitlab-org/modelops/applied-ml/code-suggestions/ai-assist`). It is public.
- **Syncing the fork back to canonical is manual.** `@gitlab-release-tools-bot` and other release automation do not work for this project. See [How canonical and the fork stay in sync](#how-canonical-and-the-fork-stay-in-sync).
- **The owning team's Engineering Manager runs the release.** The EM of the [AI Engineering team](https://handbook.gitlab.com/handbook/engineering/ai/#ai-engineering-teams) that owns the affected code schedules the release and decides with PSIRT whether it needs an announcement. See the [note on AI security fixes](https://gitlab.com/gitlab-com/content-sites/internal-handbook/-/blob/main/content/handbook/security/product_security/application_security/_index.md#note-on-ai-security-fixes).
- **Open fork MRs with `glab api`, not `glab mr create`.** `glab mr create` ignores `-R` and uses the `origin` remote, which is canonical.
- **If a fix is pushed to canonical by accident**, [engage the Security Incident Response Team (SIRT) on-call engineer](https://handbook.gitlab.com/handbook/security/security-operations/sirt/engaging-security-on-call/) and share links to the commit and merge request.

## Due date

The due date for preparing security issues and MRs follows the due date of the security issue.

## Do not push to canonical

As an engineer working on a fix for a security vulnerability, your main concern is not disclosing the vulnerability or the fix before we're ready to publicly disclose it.

To that end, fix security vulnerabilities for GitLab AI Gateway and Duo Workflow Service in the [AIGW security repo](https://gitlab.com/gitlab-org/security/modelops/applied-ml/code-suggestions/ai-assist), never in `gitlab-org/modelops/applied-ml/code-suggestions/ai-assist`. This is fundamental to our patch release process because security repositories are not publicly accessible.

## How canonical and the fork stay in sync

The two projects are connected in both directions, and the directions work differently.

- **Canonical to fork: automatic.** A scheduled [merge-train](https://gitlab.com/gitlab-org/merge-train) job copies new canonical commits into the fork's `main` every hour, as a new `Automatic merge of ... main` commit. It covers `main` only: the `merge-train-trigger` job in `.gitlab-ci.yml` sets `SOURCE_BRANCH: main`, so a canonical commit on a `stable-*-ee` branch does not reach the fork's copy of that branch, and the two can conflict when you sync back. Without the job, the fork's `main` would fall behind during an embargo and Runway would stop deploying. The commits only flow toward the fork, so this discloses nothing.
  - A run that copies commits also starts a pipeline on the fork's `main`, which is what deploys them ([merge-train!94](https://gitlab.com/gitlab-org/merge-train/-/merge_requests/94), [!7875](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/merge_requests/7875)). If no pipeline starts, the copied commits deploy with the next MR a Maintainer merges into the fork's `main`. Nobody can push to the fork's `main` directly, and a pipeline started from **Run pipeline** does not run the Runway deploy jobs.
- **Fork to canonical: manual, on purpose.** After the fix is deployed and the images are published, an engineer opens one sync MR per touched branch from the fork into canonical and merges it. This is the moment the fix becomes public, so it is a decision, not a schedule.
- **Why not automate it.** The release automation that syncs the two projects only runs after `vX.Y.Z` tags, and our patch tags are `self-hosted-vX.Y.Z-ee`. The Delivery team also keeps that sync manual by design, because when to release is up to the team. See [#2370](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/2370).
- **The push mirror.** Canonical pushes to the fork through a push mirror that skips any branch that has diverged. After you merge a fix in the fork, the mirror reports `Some refs have diverged` for that branch until you sync back. That is expected.

## Before you start

1. Check that the issue needs the security workflow:
   - If the issue in `gitlab-org/modelops/applied-ml/code-suggestions/ai-assist` is public, or carries the `security-fix-in-public` label, fix it with a normal MR on canonical and skip the backports. Stop here.
   - If the fix looks like a security enhancement that nobody can exploit, mention `@gitlab-com/gl-security/product-security/psirt-group` in the issue and ask whether the fix can be a public MR on canonical.
   - If you are updating a dependency with a known vulnerability that isn't exploitable in GitLab or has very low severity, ask PSIRT in the issue whether the update can go to canonical.
1. Add the security remote to your local checkout:

   ```shell
   git remote add security git@gitlab.com:gitlab-org/security/modelops/applied-ml/code-suggestions/ai-assist.git
   git fetch security
   ```

## Process

1. **Check that every target stable branch exists in the fork.** If one is missing, create it from the tip of canonical's branch. Branching from the tip keeps the fork's branch an ancestor of canonical's, so the push mirror does not mark it as diverged. A push cannot create a protected branch, so use the API:

   ```shell
   # <branch> is stable-19-0-ee and so on
   SHA=$(glab api "projects/gitlab-org%2Fmodelops%2Fapplied-ml%2Fcode-suggestions%2Fai-assist/repository/branches/<branch>" | jq -r .commit.id)

   glab api --method POST \
     "projects/gitlab-org%2Fsecurity%2Fmodelops%2Fapplied-ml%2Fcode-suggestions%2Fai-assist/repository/branches" \
     -f branch=<branch> -f ref="$SHA"

   # Fetch the new branch so the next step can track it
   git fetch security
   ```

1. **Create your branches from the security remote.**

   ```shell
   # Fix for main
   git checkout -b security-fix-<name> --track security/main

   # Backport for a stable branch
   git checkout -b security-fix-<name>-19-0 --track security/stable-19-0-ee

   # Push each fix branch to the fork, never to origin
   git push -u security security-fix-<name>
   git push -u security security-fix-<name>-19-0
   ```

1. **Open all MRs in the fork and get approvals.** Follow steps 1 to 4 of the [original process](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/security/engineer.md#process), with these differences:
   - Open the `main` MR and every backport MR in the **security fork**, not in canonical. Use the API so the MR cannot land in canonical by mistake:

     ```shell
     glab api --method POST \
       "projects/gitlab-org%2Fsecurity%2Fmodelops%2Fapplied-ml%2Fcode-suggestions%2Fai-assist/merge_requests" \
       -f source_branch=security-fix-<name> -f target_branch=main \
       -f title="fix: <generic title>" -F squash=false -F remove_source_branch=true
     ```

   - Backport MRs target the stable branches of the fork (for example `stable-19-0-ee` in the security repo), using the branches from the first two steps.
   - The `main` MR needs a maintainer and a PSIRT member to approve. Ping `@gitlab-com/gl-security/product-security/psirt-group` on the MR, or `@psirt-team` in `#security-help`, and share your test results from Self-Managed and GitLab Dedicated (or an emulation of them). PSIRT approves based on those results.
   - The fork has no approval rule that requires PSIRT, so check that a PSIRT member approved before you merge.
   - Backport MRs need approval from the same maintainer who approved the `main` MR. They don't need PSIRT approval, so never wait on PSIRT for a backport.
   - Copy labels (`security`, severity) from the issue, not only from the `main` MR.
   - Set each backport's milestone to the version of its target branch (`stable-19-2-ee` gets `19.2`).

1. **Merge all fork MRs.** `main` and every touched stable branch now differ between canonical and the fork. This is expected.

1. **Cut the private patch tags in the fork.** After the last backport MR for a branch merges, run the manual `tag-stable-patch` job on that stable branch's latest pipeline in the fork. It creates the next `self-hosted-vX.Y.Z-ee` tag in the fork only and builds the images into the fork's private registry.
   - Run it once per branch. A second fork tag before the sync desyncs the patch numbers from canonical.
   - The job only appears when the `AIGW_TAGGING_ACCESS_TOKEN` variable is set. It is a project variable in both canonical and the fork, so the fork does not inherit it from canonical. If the job is missing, check that the fork still has the variable. If the variable is protected, the fork must also protect `stable-*` branches, or the job never appears.

1. **Confirm the fix is deployed.** In the post-merge pipeline of the `main` MR in the fork, check that `[ai-gateway]`, `[duo-workflow-svc]` and `[ai-gateway-custom]` in the `runway_production` stage succeeded. **Do not continue until all three are green.**

   Canonical commits that merge during the embargo reach the fork by themselves. A scheduled merge-train job copies them into the fork's `main` every hour and starts the pipeline that deploys them. You don't activate or pause anything; see [How canonical and the fork stay in sync](#how-canonical-and-the-fork-stay-in-sync).

1. **Decide on the announcement.** Before anything is published, the EM of the owning team pings `@gitlab-com/gl-security/product-security/psirt-group` on the security issue to decide whether the fix gets a note in the next GitLab patch-release blog post, and when. Record the decision on the issue.

1. **Publish the images. Disclosure starts here.** In each fork tag pipeline, run the manual `release-docker-hub-image:self-managed-tag` and `release-docker-hub-self-hosted-fips-image:tag` jobs. The jobs need `DOCKERHUB_USERNAME` and `DOCKERHUB_PASSWORD` in the fork, which are also not inherited. Check that both exist in the fork's CI/CD variables before you cut the tag; if they are missing, add them through infra-mgmt. A published image can be compared with the previous patch, so only publish when you can sync to canonical right away. Do not retry a stable branch's build jobs after its tag pipeline ran, because that overwrites the image the publish jobs pull. If that happened, re-run the tag pipeline's build jobs before you publish.

1. **Sync the fork to canonical.** This is the last step of "security fork first, canonical last". It makes the fix public, and it makes canonical the source of the release: canonical's tag pipeline publishes the GitLab Container Registry images, including the signed FIPS image that Dedicated verifies. It also ends the embargo divergence, so the push mirror works again. Open one MR per touched branch from the fork into canonical (`security:main` into `canonical:main`, `security:stable-19-0-ee` into `canonical:stable-19-0-ee`, and so on), then merge them all.
   - **Turn squash off.** Canonical defaults new MRs to squash. Squash replaces the fork's commits with a new one, and the push mirror compares commit SHAs. Right after opening each sync MR, run:

     ```shell
     glab api --method PUT \
       "projects/gitlab-org%2Fmodelops%2Fapplied-ml%2Fcode-suggestions%2Fai-assist/merge_requests/<iid>" \
       -F squash=false
     ```

   - **Merge the `main` sync within the hour, from the fork's current tip.** The hourly merge-train job keeps adding commits to the fork's `main`. If one lands after you cut `sync-main`, canonical never gets the fork's newest commit and the mirror keeps rejecting `main`, even though the content matches. Right before merging, check that the sync branch still contains the fork's tip:

     ```shell
     git fetch security main && git fetch origin sync-main
     git merge-base --is-ancestor security/main origin/sync-main && echo "up to date" || echo "re-merge security/main"
     ```

     If it is behind, merge `security/main` into `sync-main` again, push, and merge.
   - A stable-branch sync cannot pass the merge train, because `feature-tests` fails on older branches. Merge it immediately instead of adding it to the train. Canonical allows this (`merge_train_enforcement: allow_bypass`), and it does not require a green pipeline:

     ```shell
     glab api --method PUT \
       "projects/gitlab-org%2Fmodelops%2Fapplied-ml%2Fcode-suggestions%2Fai-assist/merge_requests/<iid>/merge" \
       -f sha=<head sha of the sync branch> -F squash=false
     ```

   - On a merge conflict, resolve it by hand and ask a maintainer to merge. Ask whoever changed the same code to rebase after your sync merges.
   - To prepare a sync branch locally:

     ```shell
     # <branch> is main, stable-19-0-ee, and so on
     git fetch origin <branch> && git fetch security <branch>
     git checkout -b sync-<branch> origin/<branch>
     git merge security/<branch> -m "chore: sync security fork into canonical"
     git push origin sync-<branch>
     ```

1. **Verify the release.**
   - The patched images are in the GitLab Container Registry and on Docker Hub. See [How to backport a fix](./release.md#how-to-backport-a-fix).
   - In [the repository settings](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/settings/repository#js-push-remote-settings) (Maintainer role), under **Mirroring repositories**, the push mirror shows a finished update with no error.

## If the mirror says "Some refs have diverged"

Check these causes:

1. **Canonical is missing the fork's commits.** Open a sync MR from the fork's branch into canonical with squash off, and merge it. Cherry-picking the fix does not help, because the mirror compares SHAs, not content.
1. **The fork's `main` moved after the last sync.** If canonical already has the fix but `main` is still rejected, an hourly merge-train commit probably landed on the fork after the sync branch was cut. `git log --oneline origin/main..security/main` lists it. Sync again from the fork's current tip, or ask a fork Maintainer to reset the fork's `main` to canonical's after checking that `git diff origin/main security/main` shows nothing the fork has that canonical lacks.
1. **A tag exists on both sides at different commits.** Delete the fork's tag. The mirror recreates it from canonical. The published image stays in the registry.
1. **The mirror cannot push to a stable branch.** The fork's `stable-*` protection must list the mirror service account under push access, and each `stable-*-ee` branch must already exist in the fork.

For details and an example, see [how to sync the security repository with canonical](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/security/how_to_sync_security_with_canonical.md) and [this issue](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/work_items/2153).

## References

- [Patch release runbook for GitLab engineers](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/security/engineer.md)
- [How to sync the security repository with canonical](https://gitlab.com/gitlab-org/release/docs/-/blob/master/general/security/how_to_sync_security_with_canonical.md)
- [GitLab release and maintenance policy](https://docs.gitlab.com/policy/maintenance/)
- [Note on AI security fixes](https://gitlab.com/gitlab-com/content-sites/internal-handbook/-/blob/main/content/handbook/security/product_security/application_security/_index.md#note-on-ai-security-fixes)
