# Dockerfile for building Duo Agent Environment container for container registry
# For use with agent-config.yml
# Runs as root as Duo expects this

FROM python:3.12.12-slim

ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VERSION=2.3.1 \
    POETRY_VIRTUALENVS_CREATE=false

# Temp dir for installing dependencies
WORKDIR /tmp/build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    npm \
    build-essential \
    libsqlite3-dev \
    curl \
    ca-certificates \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY poetry.lock pyproject.toml ./
COPY vendor/ vendor/
RUN pip install "poetry==$POETRY_VERSION"
RUN poetry install --with test,lint,eval --no-root --no-interaction --no-ansi

RUN poetry --version && \
    python --version && \
    git --version && \
    npm --version && \
    echo "Duo Agent image build complete"
