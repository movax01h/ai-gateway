ARG SRC_IMAGE="registry.gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/model-gateway:latest"

FROM ${SRC_IMAGE}
ARG TAG
RUN mkdir "tmp"

# Specific configurations for self-hosted models
ENV AIGW_CUSTOM_MODELS__ENABLED=true
ENV AIGW_FASTAPI__API_PORT=5052
ENV AIGW_FASTAPI__OPENAPI_URL="/openapi.json"
ENV AIGW_FASTAPI__DOCS_URL="/docs"
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1
ENV HF_HUB_OFFLINE=true

# Used to run the Duo Workflow Service.
# For self-hosted Duo customers, it makes sense to
# run this service by default, rather than having them to
# specify one more environment variable during deployment.
ENV ENABLE_DUO_WORKFLOW_SERVICE=true

# Disable LangSmith tracing by default for self-hosted/air-gapped environments.
# This prevents failed connection attempts to api.smith.langchain.com.
ENV LANGSMITH_TRACING_V2=false

RUN poetry run index_docs -o "./tmp/docs.db" -v ${TAG##self-hosted-}

CMD ["./scripts/run.sh"]
