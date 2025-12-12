# Delivery process overview of GitLab AI Gateway

```mermaid
flowchart LR
  GitLabLSP["GitLab-LSP"]
  subgraph AIGWProj["AIGW Project"]
    subgraph AIGWRepo["Repository"]
      AIGWRepoMain["main"]
      subgraph AIGWGT["Git Tags"]
        AIGWv1["self-hosted-v18.6.0-ee, etc [^1]"]
      end
      subgraph AIGWSB["Stable Branches"]
          AIGWb1["stable-18-6-ee, etc [^2]"]
      end
    end
    subgraph AIGWCR["Container Registry"]
        AIGWCR1["model-gateway:SHA [^8]"]
    end
    subgraph AIGWPR["Package Registry"]
        AIGWPR1["gitlab-org/duo-workflow-service [^5]"]
    end
  end
  subgraph AIGWSecurityProj["AIGW Security Project"]
    subgraph AIGWSRepo["Repository"]
      AIGWSRepoMain["main"]
    end
    subgraph AIGWSCR["Container Registry"]
        AIGWSCR1["model-gateway:SHA"]
    end
  end
  subgraph Runway["Runway"]
    subgraph RunwayDeployments["Runway Deployments"]
      RunwayAIGW["AIGW deployment project [^6]"]
      RunwayDWS["DWS deployment project [^7]"]
    end
  end
  subgraph GCP["GCP"]
    subgraph CloudRun
      AIGWFleet["AIGW fleet [^9]"]
      DWSFleet["DWS fleet [^9]"]
    end
  end
  subgraph DockerHub["Docker Hub"]
    DHub1["model-gateway-self-hosted:self-hosted-v18.6.0-ee, etc [^10]"]
  end
  AIGWRepo -- push mirror [^3] --> AIGWSRepo
  AIGWRepoMain -- build --> AIGWCR1
  AIGWRepoMain -- build --> AIGWPR1
  AIGWSRepoMain -- trigger --> RunwayAIGW
  AIGWSRepoMain -- trigger --> RunwayDWS
  AIGWSRepoMain -- build --> AIGWSCR1
  RunwayDeployments -- pull --> AIGWSCR1
  RunwayAIGW -- deploy --> AIGWFleet
  RunwayDWS -- deploy --> DWSFleet
  GitLabLSP -- depends --> AIGWPR1
  AIGWGT -- push [^4] --> DHub1
```

Notes:

- `[^1]`: Git-Tags that are created by [GitLab-Rails](https://gitlab.com/gitlab-org/gitlab). See [Release](./release.md) for more information.
- `[^2]`: Git-Branches that are created by [GitLab-Rails](https://gitlab.com/gitlab-org/gitlab). See [Release](./release.md) for more information.
- `[^3]`: Changes on the canonical repository are mirrored to the security repository.
  - e.g. When a merge request is merged into `main` branch of the canonical repository:
    - The merge commit in the canonical repository triggers a CI/CD pipeline in the canonical project.
    - The merge commit is immediately mirrored to the security repository. The merge commit in the security project triggers a CI/CD pipeline in the security project.
- `[^4]`: When a Git-Tag is created, a CI/CD pipeline builds and push the Docker image to Docker Hub. See [Release](./release.md) for more information.
- `[^5]`: Node gRPC client lib that is used by [GitLab-LSP](https://gitlab.com/gitlab-org/editor-extensions/gitlab-lsp).
- `[^6]`: [AIGW deployment project](https://gitlab.com/gitlab-com/gl-infra/platform/runway/deployments/ai-gateway)
- `[^7]`: [DWS deployment project](https://gitlab.com/gitlab-com/gl-infra/platform/runway/deployments/duo-workflow-svc)
- `[^8]`: Docker images that are used for non-production environments e.g. `dws-loadtest` Runway deployment.
- `[^9]`: Requested by GitLab.com, Dedicated and Self-managed GitLab customers (except self-hosted Duo customers).
- `[^10]`: Pulled by self-hosted Duo customers who host AIGW/DWS fleet on their own (e.g. airgapped). See [Install the GitLab AI gateway](https://docs.gitlab.com/install/install_ai_gateway/) and [GitLab Duo Self-Hosted](https://docs.gitlab.com/administration/gitlab_duo_self_hosted/) for more information.

Related docs:

- [Release](./release.md)
- [Security fixes](./security_fixes.md)
- [GitLab AI Gateway project provisioning process](https://internal.gitlab.com/handbook/engineering/ai/ai-gateway/provisioning/) (Internal only)
