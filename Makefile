ROOT_DIR := $(shell pwd)
AI_GATEWAY_DIR := ${ROOT_DIR}/ai_gateway
EVAL_DIR := ${ROOT_DIR}/eval
LINTS_DIR := ${ROOT_DIR}/lints
SCRIPTS_DIR := ${ROOT_DIR}/scripts
TESTS_DIR := ${ROOT_DIR}/tests
INTEGRATION_TESTS_DIR := ${ROOT_DIR}/integration_tests

LINT_WORKING_DIR ?= ${AI_GATEWAY_DIR} \
	${EVAL_DIR} \
	${LINTS_DIR} \
	${SCRIPTS_DIR} \
	${TESTS_DIR} \
	${INTEGRATION_TESTS_DIR}

MYPY_LINT_TODO_DIR ?= --exclude "ai_gateway/models/*" \
	--exclude "ai_gateway/api/snowplow_context.py" \
	--exclude "ai_gateway/api/v3/code/completions.py" \
	--exclude "ai_gateway/api/v2/code/completions.py" \
	--exclude "ai_gateway/api/v2/chat/agent.py" \
	--exclude "ai_gateway/api/v1/x_ray/libraries.py" \
	--exclude "ai_gateway/api/v1/prompts/invoke.py" \
	--exclude "ai_gateway/api/v1/chat/agent.py" \
	--exclude "ai_gateway/api/server.py" \
	--exclude "ai_gateway/code_suggestions/completions.py" \
	--exclude "ai_gateway/code_suggestions/container.py" \
	--exclude "ai_gateway/code_suggestions/generations.py" \
	--exclude "ai_gateway/code_suggestions/processing/completions.py" \
	--exclude "ai_gateway/code_suggestions/processing/ops.py" \
	--exclude "ai_gateway/code_suggestions/processing/post/completions.py" \
	--exclude "tests/code_suggestions/test_completions.py" \
	--exclude "tests/code_suggestions/test_container.py" \
	--exclude "tests/code_suggestions/test_engine.py" \
	--exclude "tests/code_suggestions/test_generation.py"

COMPOSE_FILES := -f docker-compose.dev.yaml
ifneq (,$(wildcard docker-compose.override.yaml))
COMPOSE_FILES += -f docker-compose.override.yaml
endif
COMPOSE := docker-compose $(COMPOSE_FILES)
ifeq (, $(shell command -v $(COMPOSE) 2> /dev/null))
COMPOSE := docker compose $(COMPOSE_FILES)
endif
TEST_PATH_ARG ?=

.PHONY: develop-local
develop-local:
	$(COMPOSE) up --build --remove-orphans

.PHONY: test-local
test-local:
	$(COMPOSE) run -v "$(ROOT_DIR):/app" api bash -c 'poetry install --with test && poetry run pytest $(TEST_PATH_ARG)'

.PHONY: lint-local
lint-local:
	$(COMPOSE) run -v "$(ROOT_DIR):/app" api bash -c 'poetry install --only lint && poetry run flake8 ai_gateway'

.PHONY: clean
clean:
	$(COMPOSE) rm -s -v -f

.PHONY: install-lint-deps
install-lint-deps:
	@echo "Installing lint dependencies..."
	@poetry install --only lint

.PHONY: black
black: install-lint-deps
	@echo "Running black format..."
	@poetry run black ${LINT_WORKING_DIR}

.PHONY: isort
isort: install-lint-deps
	@echo "Running isort format..."
	@poetry run isort ${LINT_WORKING_DIR}

.PHONY: format
format: black isort

.PHONY: lint
lint: lint-code lint-doc

.PHONY: lint-code
lint-code: flake8 check-black check-isort check-pylint check-mypy

.PHONY: lint-commit
lint-commit:
	@npm install
	@npx commitlint --from=$$(git rev-parse main) --help-url

.PHONY: flake8
flake8: install-lint-deps
	@echo "Running flake8..."
	@poetry run flake8 ${LINT_WORKING_DIR}

.PHONY: check-black
check-black: install-lint-deps
	@echo "Running black check..."
	@poetry run black --check ${LINT_WORKING_DIR}

.PHONY: check-isort
check-isort: install-lint-deps
	@echo "Running isort check..."
	@poetry run isort --check-only ${LINT_WORKING_DIR}

.PHONY: check-pylint
check-pylint: install-lint-deps
	@echo "Running pylint check..."
	@poetry run pylint ${LINT_WORKING_DIR} --ignore=vendor

.PHONY: check-mypy
check-mypy: install-lint-deps
ifeq ($(TODO),true)
	@echo "Running mypy check todo..."
	@poetry run mypy ${LINT_WORKING_DIR}
else
	@echo "Running mypy check..."
	@poetry run mypy ${LINT_WORKING_DIR} ${MYPY_LINT_TODO_DIR} --exclude "scripts/vendor/*"
endif

.PHONY: install-test-deps
install-test-deps:
	@echo "Installing test dependencies..."
	@poetry install --with test

.PHONY: test
test: install-test-deps
	@echo "Running tests..."
	@poetry run pytest -n auto

.PHONY: test-watch
test-watch: install-test-deps
	@echo "Running tests in watch mode..."
	@poetry run ptw . -n auto

.PHONY: test-coverage
test-coverage: install-test-deps
	@echo "Running tests with coverage..."
	@poetry run pytest --cov=ai_gateway --cov=lints --cov-report term --cov-report html -n auto

.PHONY: test-coverage-ci
test-coverage-ci: install-test-deps
	@echo "Running tests with coverage on CI..."
	@poetry run pytest --cov=ai_gateway --cov=lints --cov-report term --cov-report xml:.test-reports/coverage.xml --junitxml=".test-reports/tests.xml" -n auto

.PHONY: test-integration
test-integration: install-test-deps
	@echo "Running integration tests..."
	@poetry run pytest integration_tests/ -n auto

.PHONY: lint-doc
lint-doc: vale markdownlint

.PHONY: vale
vale:
	@echo "Running vale..."
	@vale --minAlertLevel error docs *.md

.PHONY: markdownlint
markdownlint:
	@echo "Running markdownlint..."
	@markdownlint-cli2 "docs/**/*.md" *.md

.PHONY: ingest
ingest:
	@echo "Running data ingestion and refreshing for search APIs..."
	@$(ROOT_DIR)/scripts/ingest/gitlab-docs/run.sh

.PHONY: install-eval-deps
install-eval-deps:
	@echo "Installing evaluation dependencies..."
	@poetry install --with eval

.PHONY: eval
eval: install-eval-deps
	@echo "Running evaluation..."
	@poetry run eval $(PROMPT_ID) $(PROMPT_VERSION) $(DATASET)
