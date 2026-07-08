# AGENTS.md

This file provides guidance to GitLab Duo when working with this repository.

## Architectural Overview

### Project Structure

This repository contains two main services:

1. **AI Gateway** (`ai_gateway/`): Standalone service providing AI features (Code Suggestions, Duo Chat) to all GitLab instances (SaaS, self-managed, dedicated)
1. **Duo Workflow Service** (`duo_workflow_service/`): Python-based LangGraph service managing AI-powered workflows, handling communication between UI, LLM providers, and executors

### Design Patterns

**Clean Architecture**: The project follows [The Clean Architecture](https://blog.cleancoder.com/uncle-bob/2012/08/13/the-clean-architecture.html) paradigm with clear layer separation and dependencies flowing inward.

**Dependency Injection**: Uses [Dependency Injector](https://python-dependency-injector.ets-labs.org/) framework extensively. Container definitions are in `**/container.py` files. The main application container is `ai_gateway/container.py` with nested containers for different domains (models, prompts, chat, code suggestions, etc.).

**LangGraph Abstraction Layers** (Duo Workflow Service):

- **LangGraph APIs Layer**: Low-level LangGraph package interactions
- **Components Layer**: Reusable graph components built on LangGraph APIs (`duo_workflow_service/components/`, `duo_workflow_service/agent_platform/`)
- **Workflows Layer**: Ready-to-use workflow configurations combining components (`duo_workflow_service/workflows/`)

Each layer only uses entities from the layer directly below it (enforced via CI static scanning).

### Key Dependencies

- **FastAPI**: Web framework for API endpoints
- **Pydantic**: Data validation and settings management
- **LangGraph** + **LangChain** ecosystem (Anthropic, Community, Google Vertex AI, OpenAI): Workflow orchestration and LLM provider integrations
- **LiteLLM**: Unified interface for multiple LLM providers
- **Anthropic**: Claude model integration
- **Google Cloud AI Platform**: Vertex AI integration
- **Tree-sitter** + tree-sitter-languages: Code parsing for suggestions
- **Transformers**: Tokenization
- **gRPC**: Communication protocol for Duo Workflow Service
- **Python-GitLab**: GitLab API client
- **Prometheus** (`prometheus-client`, `prometheus-fastapi-instrumentator`): Metrics collection

> See `pyproject.toml` for exact pinned versions (updated frequently via Renovate).

### Component Interactions

**AI Gateway**:

- API endpoints (`ai_gateway/api/v{1,2,3,4}/`) handle HTTP requests
- Middleware (`ai_gateway/api/middleware/`) processes authentication, feature flags, internal events, usage quotas
- Code Suggestions flow: API → Processing (pre/post) → Model Engine → Prompt Builder → LLM
- Prompts are versioned templates in `ai_gateway/prompts/definitions/` using Jinja2

**Duo Workflow Service**:

- gRPC server receives workflow requests
- Workflows (`duo_workflow_service/workflows/`) compile LangGraph graphs from components
- Checkpointer (`duo_workflow_service/checkpointer/`) saves state to GitLab
- Tools (`duo_workflow_service/tools/`) provide LLM capabilities (filesystem, Git, GitLab API, etc.)
- Agents (`duo_workflow_service/agents/`) orchestrate planning and execution

**Configuration**: Environment-based using Pydantic Settings with `AIGW_` prefix. See `ai_gateway/config.py` and `example.env`.

## Essential Commands

### Development Setup

```shell
# Install dependencies
poetry install

# Activate virtualenv
poetry shell
# or manually: . ./.venv/bin/activate

# Copy environment template
cp example.env .env
# Edit .env with required credentials (ANTHROPIC_API_KEY, GCP auth, etc.)

# Authenticate with GCP
gcloud auth application-default login
```

### Running Services

```shell
# AI Gateway (with hot reload)
poetry run ai_gateway
# or: poetry run ai-gateway

# Duo Workflow Service
poetry run duo-workflow-service

# Local development with Docker Compose
make develop-local
```

### Testing

```shell
# Run all tests (parallel)
make test
# or: poetry run pytest -n auto

# Run specific test file/directory
poetry run pytest tests/path/to/test_file.py

# Run with coverage
make test-coverage

# Run integration tests
make test-integration

# Watch mode (auto-rerun on changes)
make test-watch
```

### Linting and Formatting

```shell
# Run all linters (code + docs)
make lint

# Auto-fix formatting issues (codespell, ruff, docformatter)
make format

# Individual linters
make check-ruff          # Ruff lint + format check (replaces flake8/black/isort)
make check-pylint        # Code analysis
make check-mypy          # Type checking
make check-codespell     # Spell checking
make check-docformatter  # Docstring formatting
make check-editorconfig  # Editorconfig conformance
make check-graphql       # GraphQL schema validation
make lint-proto          # buf lint for protobuf contracts

# Auto-format code
make ruff-fix       # ruff check --fix + ruff format
make docformatter
make codespell      # Auto-fix spelling

# Lint documentation
make lint-doc  # Runs vale + markdownlint
```

Always run these `make` targets rather than invoking `pytest`, `mypy`,
`ruff`, or `pylint` directly. `make test`, `check-mypy`, `check-ruff`, and
`check-pylint` install required dependencies first via `install-test-deps`/
`install-lint-deps`, and `check-mypy` also passes `--exclude` flags (for
`scripts/vendor/*` and the known-noncompliant files listed under
`MYPY_LINT_TODO_DIR` in the Makefile) that a bare `mypy` invocation would
silently skip, causing local results to disagree with CI.

### Pre-commit Hooks

```shell
# Install Lefthook hooks
lefthook install

# Run pre-commit manually
lefthook run pre-commit

# Disable temporarily
LEFTHOOK=0 git commit ...
```

### Protocol Buffers

```shell
# Generate all proto files (Python, Go, Ruby, Node)
make gen-proto

# Generate specific language
make gen-proto-python
make gen-proto-go
make gen-proto-ruby
make gen-proto-node

# Clean proto artifacts
make clean-proto
```

> **Apple Silicon (arm64) note:** `make gen-proto-ruby` runs `grpc_tools_ruby_protoc`
> from the `grpc-tools` gem, which ships an **x86_64-only** `protoc`/`grpc_ruby_plugin`.
> On Apple Silicon you must have Rosetta 2 installed to execute it, otherwise the step
> fails with `Bad CPU type in executable`. Install it once with:
>
> ```shell
> softwareupdate --install-rosetta --agree-to-license
> ```
>
> The Makefile runs the Ruby generator via `bundle exec` against `clients/ruby/Gemfile`,
> so it uses the **pinned** `grpc-tools` version (matching CI) rather than any globally
> installed gem. Don't substitute Homebrew's `protoc` — its protoc version differs and
> produces a diff (e.g. `::Google::Protobuf` vs `Google::Protobuf`) that breaks the
> `check-proto-ruby` CI job.

### Other Utilities

```shell
# Generate Duo Workflow graph documentation
make duo-workflow-docs

# Validate model selection config
poetry run validate-model-selection-config
```

## Code Style

### Formatting Rules

- **Line length**: `ruff format` targets 88 characters (default); not hard-enforced as a lint error (`E501` is ignored) — Pylint and docformatter allow up to 120 characters
- **Import sorting**: Handled by Ruff's `isort`-compatible rule group (`I`), configured in `[tool.ruff.lint.isort]`
- **Code formatter**: `ruff format` (replaced Black; migration tracked in work item #2237)
- **Docstrings**: Google-style, formatted with `docformatter` (max 120 chars)

### Python Conventions

Use type hints (mypy enforced incrementally).

```python
# Good
def process_data(data: str, transform: bool = True) -> str:
    return data.upper() if transform else data

# Bad
def process_data(data, transform=True):
    return data.upper() if transform else data
```

Use Pydantic models for configuration and validation instead of raw
dicts or ad hoc `os.environ` reads.

```python
# Good
class ConfigLogging(BaseModel):
    level: str = "INFO"
    format_json: bool = True

# Bad
LOGGING_LEVEL = os.environ.get("LOGGING_LEVEL", "INFO")
LOGGING_FORMAT_JSON = os.environ.get("LOGGING_FORMAT_JSON", "true") == "true"
```

Wire dependencies through the container with `@inject`/`Provide[...]`
rather than importing and instantiating them directly.

```python
# Good
@inject
def __init__(
    self,
    model: BaseModel = Provide[ContainerModels.model],
):
    self._model = model

# Bad
def __init__(self):
    self._model = build_model_from_config(load_config())
```

Use structured logging with bound context instead of string-formatted
messages.

```python
# Good
self.log = structlog.stdlib.get_logger("component").bind(request_id=request_id)
self.log.info("Processing request", user_id=user_id)

# Bad
logging.info(f"Processing request {request_id} for user {user_id}")
```

Use async/await for I/O operations; don't block the event loop with
synchronous clients.

```python
# Good
async def fetch_data(self, url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()

# Bad
def fetch_data(self, url: str) -> dict:
    response = requests.get(url)
    return response.json()
```

### Naming Conventions

- **Files**: `snake_case.py` (test files: `test_*.py`)
- **Classes**: `PascalCase`
- **Functions/methods**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Private members**: `_leading_underscore`
- **Environment variables**: `AIGW_SECTION__SUBSECTION__KEY` (double underscore for nesting)

### Linting Suppressions

**Avoid inline pylint/mypy/ruff disables** (`# pylint: disable=...`, `# type: ignore`, `# noqa`). They negate agreed-upon code standards. If absolutely necessary, provide a comment explaining why:

```python
# Protobuf generated code can't be parsed by pylint
# pylint: disable=no-name-in-module
from contract.contract_pb2 import Action
```

### Testing Conventions

- Test files mirror source structure: `tests/path/to/test_module.py` for `path/to/module.py`
- Prefer `@pytest.mark.parametrize` over multiple near-identical test functions that only vary
  in input/expected values. This convention is used extensively already, but it requires actively
  noticing and restructuring duplicated tests — no linter flags the alternative, so don't default
  to a new test function per case.

```python
# Good
@pytest.mark.parametrize("value,expected", [("a", "A"), ("b", "B")])
def test_upper(value, expected):
    assert value.upper() == expected

# Bad
def test_upper_a():
    assert "a".upper() == "A"

def test_upper_b():
    assert "b".upper() == "B"
```

- Name fixtures with `@pytest.fixture(name="...")` so call sites reference them as plain values,
  keeping the defining function's name free to describe its implementation.

```python
# Good
@pytest.fixture(name="mock_app_dependencies")
def mock_app_dependencies_fixture():
    ...

def test_something(mock_app_dependencies):
    ...

# Bad
@pytest.fixture
def mock_app_dependencies():
    ...
```

## Git Workflow

Use `glab` for MR/issue CLI operations (or the GitLab MCP tools, if available).

### Branch Naming

No strict convention enforced, but descriptive names recommended:

- `feature/add-new-tool`
- `fix/authentication-bug`
- `refactor/cleanup-prompts`

### Commit Guidelines

**Conventional Commits** enforced via `commitlint`:

```plaintext
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**: `feat`, `fix`, `docs`, `style`, `refactor`, `perf`, `test`, `chore`, `ci`, `build`, `revert`

**Examples**:

```plaintext
feat(duo-chat): add support for custom models

fix(code-suggestions): handle empty prefix correctly

docs: update AGENTS.md with architecture details
```

MR titles follow the same `<type>(<scope>): <subject>` format, for example
`fix(auth): resolve JWT signature validation`. MR titles are linted in CI
(`lint:commit` job runs commitlint against `$CI_MERGE_REQUEST_TITLE`) with the
same rules as commit messages, including a 100-character header limit.

### Pre-commit Checklist

Lefthook automatically runs on commit (skipped on `main`), scoped to staged files:

1. **Python files** (`*.py`): `check-mypy` (filtered to mypy-safe files), `check-ruff`, `check-pylint`, `check-codespell`, `check-docformatter`, `check-editorconfig`
1. **GraphQL files** (`*.graphql`): `check-graphql`
1. **Proto files** (`contract/*.proto`): `lint-proto` (buf lint)
1. **Markdown files** (`*.md`): vale and markdownlint
1. **All other files**: `check-codespell`, `check-editorconfig`
1. **Pre-push**: `lint-commit` validates commit messages with `commitlint` against `main`

**Before committing**:

```shell
# Run auto-formatters
make format

# Run linters
make lint
```

### MR Template

The default template (`.gitlab/merge_request_templates/Default.md`) includes:

**Required sections**:

- What does this MR do and why?
- How to set up and validate locally (numbered steps strongly suggested)

**Checklist**:

- Tests added for new functionality
- Documentation added/updated
- Executor implementation verified (if applicable)

**Labels**: Auto-applies `~"group::ai core infra"`. The author must also select exactly one
type label: `~"type::bug"`, `~"type::feature"`, or `~"type::maintenance"` (the template includes
commented-out `/label` quick actions for each — uncomment the one that applies).

### Review Process

**Streamlined process** (differs from standard GitLab guidelines):

1. **Reviewers**: All AI Engineering department engineers (no prerequisites)
1. **Maintainers**: Senior+ Backend/Full Stack/ML engineers in AI Engineering
1. **Review flow**: Reviewer → Maintainer (or directly to Maintainer for urgent MRs)
1. **Deployment**: Merged MRs auto-deploy via [Runway](https://handbook.gitlab.com/handbook/engineering/infrastructure/platforms/tools/runway/)

**Required practices**:

- Authors cannot approve their own MRs
- Never merge code you don't understand
- Request second review for: unfamiliar code, complex areas (auth/authz), or when author requests

**Domain expertise**:

- Recommendations: See `Dangerfile`
- Requirements: See `.gitlab/CODEOWNERS`

### CI/CD

Pipelines defined in `.gitlab-ci.yml` and `.gitlab/ci/*.gitlab-ci.yml`:

- **Lint**: Code style, type checking, documentation
- **Test**: Unit tests with coverage, integration tests
- **Build**: Docker images for AI Gateway and Duo Workflow Service
- **Deploy**: Automatic deployment to staging/production via Runway

## Project-Specific Details

### Prompt Management

Prompts are versioned Jinja2 templates in `ai_gateway/prompts/definitions/`:

```plaintext
prompts/definitions/
├── chat/
│   ├── explain_code/
│   │   ├── base/1.0.0.yml          # Base config
│   │   ├── system/1.0.0.jinja      # System prompt template
│   │   └── user/1.0.0.jinja        # User prompt template
```

**Updating prompts**: Use `scripts/update_prompt_version.sh` to create new versions.

### Model Selection

Model configuration in `ai_gateway/model_selection/models.yml`:

- Defines available models per feature and deployment type
- Unit primitives in `unit_primitives.yml`
- Validate with: `poetry run validate-model-selection-config`

### Deployment Environments

- **Production**: `https://cloud.gitlab.com/ai` (via Cloudflare) → `https://ai-gateway.runway.gitlab.net`
- **Staging**: Automatic deployment on merge to `main`
- **Staging-ref**: Custom models environment (`ai-gateway-custom`)
- **Monitoring**: [Service overview dashboard](https://dashboards.gitlab.net/d/ai-gateway-main/ai-gateway-overview)

### Internal Events

Track events using GitLab Internal Events:

- Event definitions: `config/events/*.yml`
- Client: `lib/internal_events/client.py`
- Enum: `lib/internal_events/event_enum.py`

### Security

- **Authentication**: JWT-based (Cloud Connector), bypass options for local dev
- **Prompt security**: Tools in `duo_workflow_service/security/` validate and sanitize prompts

### Troubleshooting

```shell
# Mock model responses (no LLM calls)
AIGW_MOCK_MODEL_RESPONSES=true poetry run ai_gateway

# Enable debug logging
AIGW_LOGGING__LEVEL=debug poetry run ai_gateway

# Troubleshoot self-hosted installation
poetry run troubleshoot

# Check Poetry dependency conflicts
poetry install --sync
```

### Documentation

- **API docs**: `docs/api.md`
- **Architecture**: `docs/duo_workflow_service.md`, [Blueprint](https://docs.gitlab.com/ee/architecture/blueprints/duo_workflow/)
- **Testing**: `docs/tests.md`
- **Authentication**: `docs/auth.md`
- **Workflows**: `docs/workflows/*.md`
- **Markdown linting**: Uses `markdownlint-cli2` with config in `.markdownlint-cli2.yaml`
