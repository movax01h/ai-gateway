FROM debian:12-slim

RUN apt-get update  \
    && apt-get -y --no-install-recommends install  \
        # install any other dependencies you might need
        sudo curl git ca-certificates build-essential libz-dev \
        # For ruby https://github.com/rbenv/ruby-build/wiki#suggested-build-environment
        autoconf patch rustc libssl-dev libyaml-dev libreadline6-dev zlib1g-dev libgmp-dev libncurses5-dev libffi-dev libgdbm6 libgdbm-dev libdb-dev uuid-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the locale UTF8
ENV LANG=C.UTF-8
ENV LANGUAGE=C
ENV LC_ALL=C.UTF-8

WORKDIR /app

# Install dependencies via mise
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV MISE_DATA_DIR="/mise"
ENV MISE_CONFIG_DIR="/mise"
ENV MISE_CACHE_DIR="/mise/cache"
ENV MISE_INSTALL_PATH="/usr/local/bin/mise"
ENV PATH="/mise/shims:$PATH"
RUN curl https://mise.run | sh

COPY .tool-versions /app/.tool-versions

# Set up Python first since gcloud requires it.
RUN mise install python
RUN mise install

# Install python dependencies
ENV VIRTUAL_ENV=/usr/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY poetry.lock pyproject.toml /app/
COPY vendor/ /app/vendor
RUN python -m venv $VIRTUAL_ENV && poetry install --no-interaction --no-ansi --only main --no-root

# Copy scripts
COPY Makefile /app/Makefile
COPY scripts /app/scripts

CMD ["make", "ingest"]
