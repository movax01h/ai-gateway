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

- **FastAPI** (^0.116.0): Web framework for API endpoints
- **Pydantic** (^2.7.4): Data validation and settings management
- **LangGraph** (0.4.8): Workflow orchestration for Duo Workflow Service
- **LangChain** ecosystem: Anthropic (^0.3.17), Community (^0.3.5), Google Vertex AI (^2.0.8), OpenAI (^0.3.30)
- **LiteLLM** (^1.35.20): Unified interface for multiple LLM providers
- **Anthropic** (^0.71.0): Claude model integration
- **Google Cloud AI Platform** (^1.36.4): Vertex AI integration
- **Tree-sitter** (^0.21.0) + tree-sitter-languages (^1.10.2): Code parsing for suggestions
- **Transformers** (^4.37.2): Tokenization
- **gRPC** (^1.68.1): Communication protocol for Duo Workflow Service
- **Python-GitLab** (^6.0.0): GitLab API client
- **Prometheus** (^0.22.0): Metrics collection

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
# Run all linters
make lint

# Auto-fix formatting issues
make format

# Individual linters
make flake8           # Style guide enforcement
make check-black      # Code formatting check
make check-isort      # Import sorting check
make check-pylint     # Code analysis
make check-mypy       # Type checking
make check-codespell  # Spell checking
make check-docformatter  # Docstring formatting

# Auto-format code
make black
make isort
make docformatter
make codespell  # Auto-fix spelling

# Lint documentation
make lint-doc  # Runs vale + markdownlint
```

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

### Other Utilities

```shell
# Generate Duo Workflow graph documentation
make duo-workflow-docs

# Validate model selection config
poetry run validate-model-selection-config

# Generate dataset for evaluation
poetry run generate-dataset

# Run evaluation
make eval PROMPT_ID=<id> PROMPT_VERSION=<version> DATASET=<path> EVALUATORS=<evaluators>
```

## Code Style

### Formatting Rules

- **Line length**: 120 characters (enforced by Black and Pylint)
- **Import sorting**: Use `isort` with Black-compatible profile
- **Code formatter**: Black (no configuration needed, opinionated)
- **Docstrings**: Google-style, formatted with `docformatter` (max 120 chars)

### Python Conventions

```python
# Use type hints (mypy enforced incrementally)
def process_data(data: str, transform: bool = True) -> str:
    return data.upper() if transform else data

# Pydantic models for configuration and validation
class ConfigLogging(BaseModel):
    level: str = "INFO"
    format_json: bool = True

# Dependency injection via containers
@inject
def __init__(
    self,
    model: BaseModel = Provide[ContainerModels.model],
):
    self._model = model

# Structured logging with context
self.log = structlog.stdlib.get_logger("component").bind(request_id=request_id)
self.log.info("Processing request", user_id=user_id)

# Async/await for I/O operations
async def fetch_data(self, url: str) -> dict:
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
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

**Avoid inline pylint/mypy disables**. They negate agreed-upon code standards. If absolutely necessary, provide a comment explaining why:

```python
# Protobuf generated code can't be parsed by pylint
# pylint: disable=no-name-in-module
from contract.contract_pb2 import Action
```

### Testing Conventions

- Test files mirror source structure: `tests/path/to/test_module.py` for `path/to/module.py`

## Git Workflow

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

### Pre-commit Checklist

Lefthook automatically runs on commit:

1. **Python files**: Lints with mypy (filtered), flake8, black, isort, pylint, codespell, docformatter, editorconfig
1. **Markdown files**: Lints with vale and markdownlint
1. **All files**: Spell check with codespell, editorconfig validation

**Before committing**:

```shell
# Run auto-formatters
make format

# Run linters
make lint
```

### Title Format

Use Conventional Commits format:

```plaintext
<type>(<scope>): <subject>
```

Examples:

- `feat(workflows): implement issue-to-MR workflow`
- `fix(auth): resolve JWT signature validation`
- `docs: add troubleshooting guide for local setup`

### MR Template

The default template (`.gitlab/merge_request_templates/Default.md`) includes:

**Required sections**:

- What does this MR do and why?
- How to set up and validate locally (numbered steps strongly suggested)

**Checklist**:

- Tests added for new functionality
- Documentation added/updated
- Executor implementation verified (if applicable)

**Labels**: Auto-applies `~"group::ai framework"`

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
│   │   ├── claude_3/1.0.0.yml      # Model-specific override
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
