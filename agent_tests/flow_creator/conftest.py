"""Fixtures for the Flow Creator benchmark suite.

The agent under test is the single-agent flow at
``duo_workflow_service/agent_platform/v1/flows/configs/flow_creator``.
Its system prompt instructs it to research the Flow Registry documentation
before answering, so the suite gives it read-only tools backed by this
repository's own ``docs/flow_registry/`` files: real content, deterministic, and
no network access.

The declared toolset in the flow config is deliberately not used to build the
agent. Its tools reach live GitLab and filesystem resources that are not
available here, and none of them affect whether the emitted YAML is correct.
Substituting doc-serving mocks keeps the benchmark hermetic while exercising the
same research-then-answer loop the real agent runs.
"""

# pylint: disable=redefined-outer-name,import-outside-toplevel

from __future__ import annotations

from pathlib import Path
from typing import Any, Type

import pytest
import yaml
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agent_tests.conftest import (
    make_passthrough_optimizer_pipeline,
    make_prompt_adapter_class,
)
from agent_tests.flow_creator import helpers
from agent_tests.flow_creator.cases import CASES_BY_ID, FOLLOW_UPS, FlowCase

REPO_ROOT = Path(__file__).resolve().parents[2]

FLOW_CONFIG_PATH = (
    REPO_ROOT
    / "duo_workflow_service"
    / "agent_platform"
    / "v1"
    / "flows"
    / "configs"
    / "flow_creator"
    / "1.0.0.yml"
)

# Documentation the mocked tools will serve, keyed by repository path. The agent
# prompt names a doc in gitlab-org/gitlab as its primary source, which is not
# available here; these are the secondary sources it also names.
DOC_PATHS = (
    "docs/flow_registry/index.md",
    "docs/flow_registry/v1.md",
    "docs/flow_registry/experimental.md",
    "docs/flow_registry/contribution_guidelines.md",
)

# A safety net only: the real files are well under this, so docs are served
# whole. Truncating them would unfairly penalise the supervisor and human-input
# cases, whose reference material sits late in `v1.md`.
MAX_DOC_CHARS = 200_000

AGENT_NAME = "flow_creator"


def pytest_collection_modifyitems(items):
    """Auto-apply the flow_registry marker to all tests in this directory."""
    for item in items:
        if "/agent_tests/flow_creator/" in str(item.path):
            item.add_marker(pytest.mark.flow_registry)


def _load_docs() -> dict[str, str]:
    """Read the served documentation from disk, keyed by repository path."""
    documents: dict[str, str] = {}
    for path in DOC_PATHS:
        file_path = REPO_ROOT / path
        try:
            documents[path] = file_path.read_text()[:MAX_DOC_CHARS]
        except OSError:
            continue
    return documents


_DOCS = _load_docs()


def _resolve_doc(requested: str) -> str | None:
    """Return the doc a requested path refers to, tolerating path variations."""
    normalized = requested.strip().lstrip("/")
    if normalized in _DOCS:
        return _DOCS[normalized]

    for path, content in _DOCS.items():
        if normalized.endswith(path) or normalized == Path(path).name:
            return content
    return None


def _not_found(requested: str) -> str:
    """Return an error that tells the agent what it can read instead."""
    available = "\n".join(f"- {path}" for path in _DOCS)
    return (
        f"File not found: {requested}\n\n"
        f"Only the Flow Registry documentation is available in this "
        f"environment:\n{available}"
    )


class ReadFileInput(BaseModel):
    file_path: str = Field(description="Path of the file to read")


class MockReadFile(BaseTool):
    """Mock of read_file, serving this repository's Flow Registry docs."""

    name: str = "read_file"
    description: str = (
        "Read the contents of a file. Use this to read the Flow Registry "
        "documentation before designing a flow."
    )
    args_schema: Type[BaseModel] = ReadFileInput

    def _run(self, file_path: str, **_: Any) -> str:
        return _resolve_doc(file_path) or _not_found(file_path)

    async def _arun(self, file_path: str, **kwargs: Any) -> str:
        return self._run(file_path, **kwargs)


class GetRepositoryFileInput(BaseModel):
    file_path: str = Field(description="Path of the file in the repository")
    project_id: str | None = Field(default=None)
    ref: str | None = Field(default=None)
    url: str | None = Field(default=None)


class MockGetRepositoryFile(BaseTool):
    """Mock of get_repository_file, serving this repository's Flow Registry docs."""

    name: str = "get_repository_file"
    description: str = (
        "Read a file from a GitLab repository. Use this to read reference "
        "documentation such as doc/user/duo_agent_platform/flows/*.md."
    )
    args_schema: Type[BaseModel] = GetRepositoryFileInput

    def _run(self, file_path: str, **_: Any) -> str:
        return _resolve_doc(file_path) or _not_found(file_path)

    async def _arun(self, file_path: str, **kwargs: Any) -> str:
        return self._run(file_path, **kwargs)


class DocumentationSearchInput(BaseModel):
    search: str = Field(description="Search terms")


class MockDocumentationSearch(BaseTool):
    """Mock of gitlab_documentation_search, listing the docs that can be read."""

    name: str = "gitlab_documentation_search"
    description: str = (
        "Search the GitLab documentation. Returns the documentation pages "
        "available for the Flow Registry framework."
    )
    args_schema: Type[BaseModel] = DocumentationSearchInput

    def _run(self, search: str, **_: Any) -> str:
        available = "\n".join(f"- {path}" for path in _DOCS)
        return (
            f"Results for {search!r}. The Flow Registry documentation available "
            f"in this environment is listed below. Read a page with read_file to "
            f"get its full contents.\n{available}"
        )

    async def _arun(self, search: str, **kwargs: Any) -> str:
        return self._run(search, **kwargs)


@pytest.fixture
def doc_research_tools():
    """Read-only documentation tools the agent under test is given."""
    return [MockReadFile(), MockGetRepositoryFile(), MockDocumentationSearch()]


@pytest.fixture
def flow_registry_config():
    """The flow config under test, parsed."""
    with open(FLOW_CONFIG_PATH, encoding="utf-8") as file:
        return yaml.safe_load(file)


@pytest.fixture
def flow_registry_system_template(flow_registry_config):
    """The agent's system prompt, loaded from the flow config."""
    return flow_registry_config["prompts"][0]["prompt_template"]["system"]


@pytest.fixture
def flow_registry_max_tokens(flow_registry_config):
    """The output token budget declared by the flow config.

    The shared ``real_llm`` fixture caps output at 4096 tokens, which is not
    enough for a complete multi-component flow YAML. Honouring the config's own
    ``max_tokens`` keeps the benchmark from measuring a truncation artifact.
    """
    # `or {}` rather than a `.get` default: a `model:` or `params:` key present
    # but empty parses as None, which has no `.get`.
    model = flow_registry_config["prompts"][0].get("model") or {}
    params = model.get("params") or {}
    return params.get("max_tokens", 8192)


@pytest.fixture
def flow_registry_llm(execution_model, flow_registry_max_tokens):
    """The execution model, with the flow config's output token budget."""
    return ChatAnthropic(  # type: ignore[call-arg]
        model=execution_model,
        max_tokens=flow_registry_max_tokens,
    )


@pytest.fixture
def flow_registry_agent(
    flow_registry_llm,
    flow_registry_system_template,
    doc_research_tools,
    mock_tools_registry,
):
    """The Flow Creator agent, wired to the real system prompt."""
    from duo_workflow_service.agents.chat_agent import ChatAgent
    from duo_workflow_service.tools.toolset import Toolset

    prompt_adapter_class = make_prompt_adapter_class()
    adapter = prompt_adapter_class(
        model=flow_registry_llm,
        system_template=flow_registry_system_template,
        tools=doc_research_tools,
        agent_name=AGENT_NAME,
    )

    tools_dict = {tool.name: tool for tool in doc_research_tools}
    return ChatAgent(
        name=AGENT_NAME,
        prompt_adapter=adapter,
        tools_registry=mock_tools_registry,
        system_template_override=None,
        toolset=Toolset(pre_approved=set(), all_tools=tools_dict),
        optimizer_pipeline=make_passthrough_optimizer_pipeline(),
    )


# Generated flows are memoised per worker as well as cached on disk, so that a
# run only ever pays for one conversation per case even if the cache directory
# is not writable.
_GENERATED: dict[tuple[str, str], helpers.GeneratedFlow] = {}


@pytest.fixture
def generated_flow(
    flow_registry_agent,
    initial_state,
    execution_model,
    validation_model,
    request,
):
    """Return an async accessor for the flow YAML a test case produced.

    The same generated YAML is shared by ``test_smoke`` and ``test_hard_rules``:
    the first test to ask for a case generates it, and every later test reuses
    it from the in-process memo or the on-disk cache. Pass ``--refresh-cache``
    to force regeneration.

    ``--refresh-cache`` bypasses the *on-disk* cache only. The in-process memo is
    still consulted, so a refreshing run pays for one conversation per case
    rather than one per test, and every test in the run scores the same YAML.
    """
    refresh = request.config.getoption("--refresh-cache")

    async def _generated_flow(case: FlowCase | str) -> helpers.GeneratedFlow:
        if isinstance(case, str):
            case = CASES_BY_ID[case]

        key = (case.case_id, execution_model)
        if key in _GENERATED:
            return _GENERATED[key]

        flow = await helpers.generate_flow(
            flow_registry_agent,
            initial_state,
            case,
            follow_ups=FOLLOW_UPS,
            execution_model=execution_model,
            validation_model=validation_model,
            use_cache=not refresh,
        )
        _GENERATED[key] = flow
        return flow

    return _generated_flow
