# Maintainership

## How to become a reviewer

All Backend, Full Stack, and Machine learning engineers in AI Engineering should be reviewers (or maintainers) for the AI Gateway / Duo Workflow Service.

An exception to this is any engineers who are already maintaining multiple other projects in the GitLab AI domain.

### Steps to become a reviewer

1. **Update your team member YAML file** in [`www-gitlab-com`](https://gitlab.com/gitlab-com/www-gitlab-com) with the following format:

   ```yaml
   projects:
     ai-gateway: reviewer
   ```

## How to become a maintainer

### Prerequisites for maintainer role

We generally recommend the following activities before submitting the request:

- familiarize yourself with the [preferred domain knowledge](#preferred-domain-knowledge)
- **Author 5+ MRs** that demonstrate understanding of the codebase.
- **Review 5+ MRs** showing the ability to provide quality feedback.

### Steps to become a maintainer

1. **Create a merge request** from [this MR template](https://gitlab.com/gitlab-com/www-gitlab-com/-/blob/master/.gitlab/merge_request_templates/AI%20Gateway%20maintainer.md) to update your [team member entry](https://gitlab.com/gitlab-com/www-gitlab-com/blob/master/doc/team_database.md) and indicate your role as the following:
   - `ai-gateway: maintainer`
1. **Assign the MR** to your manager and `@gitlab-org/maintainers/duo-workflow-service` for approval and merge.
1. **Request group membership** after the MR is merged:
   - Ask an Owner of [`@gitlab-org/maintainers/duo-workflow-service`](https://gitlab.com/groups/gitlab-org/maintainers/duo-workflow-service/-/group_members?with_inherited_permissions=exclude) group to add you
   - This makes you one of the [Code Owners](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/.gitlab/CODEOWNERS?ref_type=heads) whose approval is required for an MR to be merged.
   - Do **not** add members directly as `Owners` to the `https://gitlab.com/groups/gitlab-org/maintainers/duo-workflow-service/` project
1. **Enable dependency review notifications**: Add your username to `additionalReviewers` in [`renovate.json`](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/renovate.json)
   - This includes you as a reviewer for dependency update merge requests.

## Preferred domain knowledge

In order to maintain the AI Gateway project, familiarize yourself with the following:

- [Python](https://www.python.org) as the primary programming language.
- [Poetry](https://python-poetry.org) as the package and virtual environment manager.
- [FastAPI](https://fastapi.tiangolo.com/) as the modern web framework.
- [LangGraph](https://langchain-ai.github.io/langgraph/) bedrock of Flow Registry framework used by Duo Agent Platform.
- [gRPC](https://grpc.io/docs/what-is-grpc/introduction/) as the primary method to send messages between the Executor and the Service
- [Architectural blueprint for AI Gateway](https://docs.gitlab.com/ee/architecture/blueprints/ai_gateway/) to understand how the AI Gateway is integrated with the other components.
- [Architectural blueprint for Duo Workflow Service](https://handbook.gitlab.com/handbook/engineering/architecture/design-documents/duo_workflow/) to understand how Duo Workflow is designed and how the service integrates with the overall design.
- [Architectural blueprint for Flow Registry](https://handbook.gitlab.com/handbook/engineering/architecture/design-documents/ai_agent_registry/) to understand how Flow Registry framework used by Duo Agent Platform is designed
- [Flow Registry documentation](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/tree/main/docs/flow_registry) to understand how Duo Agent Platform features are built
- [Flow Registry contribution guidelines](https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/blob/main/docs/flow_registry/contribution_guidelines.md) to understand how to develop new features for Flow Registry framework
- [Consolidating Evaluation Tooling](https://handbook.gitlab.com/handbook/engineering/architecture/design-documents/ai_evaluation_consolidation/)
- [Prompts Migration Versioning](https://handbook.gitlab.com/handbook/engineering/architecture/design-documents/prompts_migration/#versioning)
- [Self-Hosted Model Deployment](https://handbook.gitlab.com/handbook/engineering/architecture/design-documents/custom_models/)
- [Duo Agent Platform](https://docs.gitlab.com/user/duo_agent_platform/) product documentation
- [AI Model Selection](https://handbook.gitlab.com/handbook/engineering/architecture/design-documents/ai_model_selection/)
- [AI Context Abstraction Layer](https://handbook.gitlab.com/handbook/engineering/architecture/design-documents/ai_context_abstraction_layer/)

## New team members

When you join the AI Engineering department as a backend or fullstack engineer, complete the following onboarding checklist:

### Onboarding Checklist

- [ ] **Study the codebase**: Familiarize yourself with the [preferred domain knowledge](#preferred-domain-knowledge).
- [ ] **Add reviewer permissions**: Update your team member YAML file in [`www-gitlab-com`](https://gitlab.com/gitlab-com/www-gitlab-com) to become a reviewer.
- [ ] **Configure deployment access**: Add your username to the `members` of the `ai-gateway` and `duo-workflow-service` entries in [`workloads.yml`](https://gitlab.com/gitlab-com/gl-infra/platform/runway/provisioner/-/blob/main/config/runtimes/cloud-run/workloads.yml).
  - This makes you a deployer of the [AI Gateway](https://gitlab.com/gitlab-com/gl-infra/platform/runway/deployments/ai-gateway) and [Duo Workflow Service](https://gitlab.com/gitlab-com/gl-infra/platform/runway/deployments/duo-workflow-svc).
  - Deployments happen on merge (only maintainers can merge, but this prepares you for that role).
