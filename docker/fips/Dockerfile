# Use official Red Hat UBI9 Python 3.12 minimal image for FIPS compliance
FROM registry.access.redhat.com/ubi9/python-312-minimal:latest AS base-image

USER 0

ARG FIPS_MODE=1
ARG POETRY_VERSION=2.2.1
ARG OPENSSL_FORCE_FIPS_MODE=1
ARG GNUTLS_FORCE_FIPS_MODE=3
ARG GOLANG_FIPS=1

ENV POETRY_VERSION=${POETRY_VERSION} \
    FIPS_MODE=${FIPS_MODE} \
    OPENSSL_FORCE_FIPS_MODE=${OPENSSL_FORCE_FIPS_MODE} \
    GNUTLS_FORCE_FIPS_MODE=${GNUTLS_FORCE_FIPS_MODE} \
    GOLANG_FIPS=${GOLANG_FIPS} \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/aigateway/.hf \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VIRTUALENVS_PATH=/home/aigateway/app/venv \
    POETRY_CONFIG_DIR=/home/aigateway/app/.config/pypoetry \
    POETRY_DATA_DIR=/home/aigateway/app/.local/share/pypoetry \
    POETRY_CACHE_DIR=/home/aigateway/app/.cache/pypoetry \
    ENABLE_DUO_WORKFLOW_SERVICE=true

RUN useradd --create-home --user-group aigateway \
    && install --owner aigateway --group aigateway -d /home/aigateway/app

WORKDIR /home/aigateway/app

RUN curl -JLO --retry 6 https://dl.fedoraproject.org/pub/epel/epel-release-latest-9.noarch.rpm \
    && rpm -i epel-release-latest-9.noarch.rpm \
    && microdnf upgrade --nodocs --best --assumeyes --setopt=install_weak_deps=0 \
    && microdnf install --nodocs --best --assumeyes --setopt=install_weak_deps=0 \
        findutils \
        parallel \
    && microdnf clean all

RUN pip install "poetry==$POETRY_VERSION"

RUN mkdir -p -m 777 $POETRY_CONFIG_DIR $POETRY_DATA_DIR $POETRY_CACHE_DIR

COPY --chown=aigateway:aigateway poetry.lock pyproject.toml ./
COPY --chown=aigateway:aigateway README.md README.md
COPY --chown=aigateway:aigateway ai_gateway/ ai_gateway/
COPY --chown=aigateway:aigateway duo_workflow_service/ duo_workflow_service/
COPY --chown=aigateway:aigateway lib/ lib/
COPY --chown=aigateway:aigateway contract/ contract/
COPY --chown=aigateway:aigateway scripts/ scripts/
COPY --chown=aigateway:aigateway vendor/ vendor/

##
## Intermediate image contains build dependencies
##
FROM base-image AS install-image

RUN microdnf install --nodocs --best --assumeyes --setopt=install_weak_deps=0 \
        gcc \
        gcc-c++ \
        make \
        python3.12-devel \
        libstdc++-static \
    && microdnf clean all

# Specifically compile dependency_injector, instead of relying upon the wheel.
# This prevents: __Pyx_GetCurrentInterpreterId failed. Try setting the C define CYTHON_PEP489_MULTI_PHASE_INIT=0
RUN POETRY_INSTALLER_NO_BINARY=dependency_injector poetry install --compile --no-interaction --no-ansi --no-cache --only main

RUN find /home/aigateway/app/venv -type d -name "tests" -exec rm -rf {} + 2>/dev/null || true && \
    rm -rf /home/aigateway/app/venv/*/lib*/python3.12/site-packages/{pip,setuptools,wheel}*

##
## Final image
##
FROM base-image AS final

USER aigateway

COPY --chown=aigateway:aigateway --from=install-image /home/aigateway/app/venv/ /home/aigateway/app/venv/

# Ensure poetry recognizes the venv copied from install-image not the
# Poetry default /opt/app-root and can't find installed entry points
RUN poetry env use venv/ai-gateway-*/bin/python

RUN poetry run python scripts/bootstrap.py

USER 0

RUN --mount=type=bind,rw,from=hardening,source=/,target=/hardening \
    set -ex; \
    for f in /hardening/*.sh; do \
        sh "$f"; \
    done

USER aigateway

EXPOSE 5052

CMD ["./scripts/run.sh"]
