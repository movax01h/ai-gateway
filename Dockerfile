FROM python:3.12.12-slim AS base-image

# HF_HOME and POETRY_* are needed to allow the docker image to be run from a non-root users
# without having issues with permissions
ENV PYTHONUNBUFFERED=1 \
  HF_HOME=/home/aigateway/.hf \
  PIP_NO_CACHE_DIR=1 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  POETRY_VERSION=2.2.1 \
  POETRY_VIRTUALENVS_PATH=/home/aigateway/app/venv \
  POETRY_CONFIG_DIR=/home/aigateway/app/.config/pypoetry \
  POETRY_DATA_DIR=/home/aigateway/app/.local/share/pypoetry \
  POETRY_CACHE_DIR=/home/aigateway/app/.cache/pypoetry

WORKDIR /home/aigateway/app

COPY poetry.lock pyproject.toml ./
RUN pip install "poetry==$POETRY_VERSION"
RUN mkdir -p -m 777 $POETRY_CONFIG_DIR $POETRY_DATA_DIR $POETRY_CACHE_DIR

COPY README.md README.md
COPY ai_gateway/ ai_gateway/
COPY duo_workflow_service/ duo_workflow_service/
COPY lib/ lib/
COPY contract/ contract/
COPY scripts/ scripts/
COPY vendor/ vendor/

##
## Intermediate image contains build-essential for installing
## google-cloud-profiler's dependencies
##
FROM base-image AS install-image

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN poetry install --compile --no-interaction --no-ansi --no-cache --only main

##
## Final image copies dependencies from install-image
##
FROM base-image AS final

RUN apt-get update && apt-get install -y --no-install-recommends \
    parallel \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd aigateway
RUN chown -R aigateway:aigateway /home/aigateway/
USER aigateway

COPY --chown=aigateway:aigateway --from=install-image /home/aigateway/app/venv/ /home/aigateway/app/venv/

RUN poetry run python scripts/bootstrap.py

EXPOSE 5052

CMD ["./scripts/run.sh"]
