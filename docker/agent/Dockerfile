# Dockerfile for building Duo Agent Environment container for container registry
# For use with agent-config.yml
# Runs as root as Duo expects this

FROM python:3.12.12-slim

ARG GIT_VERSION=2.43.7
ARG GIT_SHA256=e3515ee128ce124d69f925d8062d165c3599b172593a9e0cf4a830f682a94b9b

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=2.3.1 \
    POETRY_VIRTUALENVS_CREATE=false

# Temp dir for installing dependencies
WORKDIR /tmp/build

# Install system dependencies including git build requirements
RUN apt-get update && apt-get install -y --no-install-recommends \
    npm \
    build-essential \
    libsqlite3-dev \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    libexpat1-dev \
    gettext \
    libz-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install git from source with checksum verification
RUN curl -fsSL https://github.com/git/git/archive/v${GIT_VERSION}.tar.gz -o git.tar.gz \
    && echo "${GIT_SHA256}  git.tar.gz" | sha256sum -c - \
    && tar -xf git.tar.gz \
    && cd git-${GIT_VERSION} \
    && make -j$(nproc) prefix=/usr/local all \
    && make prefix=/usr/local install \
    && cd .. \
    && rm -rf git-${GIT_VERSION} git.tar.gz

COPY poetry.lock pyproject.toml ./
COPY vendor/ vendor/
RUN pip install "poetry==$POETRY_VERSION"
RUN poetry install --with test,lint,eval --no-root --no-interaction --no-ansi

RUN poetry --version && \
    python --version && \
    git --version && \
    npm --version && \
    echo "Duo Agent image build complete"
