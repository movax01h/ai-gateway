"""Shared fixtures for agent tests.

These fixtures provide common setup for testing DWS agents with real LLMs.
The execution and validation models are configurable via CLI options.

To run agent tests in parallel for speed:
    pytest agent_tests/ -n auto

Requires ANTHROPIC_API_KEY environment variable.
"""

# pylint: disable=redefined-outer-name,import-outside-toplevel,super-init-not-called

from __future__ import annotations

import datetime
import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Type
from unittest.mock import AsyncMock, MagicMock, Mock

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from ai_gateway.model_selection.models import ChatAnthropicParams
    from duo_workflow_service.conversation.history_optimizer.pipeline import (
        HistoryOptimizerPipeline,
    )
    from duo_workflow_service.entities.state import ChatWorkflowState

REPORT_DIR = Path(__file__).resolve().parents[1] / ".test-reports" / "agent_tests"


def pytest_addoption(parser):
    """Add custom CLI options for model selection."""
    parser.addoption(
        "--execution-model",
        default="claude-sonnet-4-6",
        help="Anthropic model for agent execution (default: claude-sonnet-4-6)",
    )
    parser.addoption(
        "--validation-model",
        default="claude-haiku-4-5-20251001",
        help="Anthropic model for LLM-as-judge validation (default: claude-haiku-4-5-20251001)",
    )
    parser.addoption(
        "--refresh-cache",
        action="store_true",
        default=False,
        help=(
            "Ignore cached agent responses and generate them again. Suites that "
            "score the same response from several test files cache it on disk "
            "under .test-reports/agent_tests/."
        ),
    )


def pytest_configure(config):
    """Register custom markers and set model defaults from CLI options."""
    config.addinivalue_line(
        "markers", "analytics: mark test as an analytics agent test"
    )
    config.addinivalue_line(
        "markers", "flow_registry: mark test as a Flow Creator agent test"
    )
    config.addinivalue_line(
        "markers",
        "flow_versions(*versions): flow config versions this test runs against",
    )
    # Registered defensively: pytest-xdist owns this marker, but suites use it to
    # keep every test that scores the same cached LLM response on one worker, and
    # an unregistered marker would be an error under `filterwarnings`.
    config.addinivalue_line(
        "markers",
        "xdist_group(name): run tests sharing a group on the same xdist worker",
    )
    validation_model = config.getoption("--validation-model", default=None)
    if validation_model:
        from agent_tests import helpers

        helpers.DEFAULT_VALIDATION_MODEL = validation_model


def make_passthrough_optimizer_pipeline() -> HistoryOptimizerPipeline:
    """Build a HistoryOptimizerPipeline mock that returns history unchanged.

    Agent tests exercise prompt and tool behavior, so history optimization is disabled to keep the conversation the LLM
    sees identical to the one the test built.
    """
    from duo_workflow_service.conversation.history_optimizer.pipeline import (
        HistoryOptimizerPipeline,
    )
    from duo_workflow_service.conversation.history_optimizer.schema import (
        OptimizationResult,
    )

    mock_pipeline = Mock(spec=HistoryOptimizerPipeline)

    async def optimize(history):
        return history, [OptimizationResult(messages=history, was_modified=False)]

    mock_pipeline.optimize = AsyncMock(side_effect=optimize)
    return mock_pipeline


def _make_prompt_adapter_class():
    from duo_workflow_service.agents.prompt_adapter import BasePromptAdapter

    class RealLLMPromptAdapter(BasePromptAdapter):
        """Prompt adapter that uses a real LLM for testing."""

        def __init__(
            self,
            model: ChatAnthropic,
            system_template: str,
            tools: list[Any],
            agent_name: str = "agent",
        ):
            self._model = model.bind_tools(tools)
            self._system_template = system_template
            self._tools = tools
            self._agent_name = agent_name

        def get_model(self):
            return self._model

        async def get_response(
            self,
            input: ChatWorkflowState,
            **kwargs,
        ) -> AIMessage:
            from langchain_core.messages import BaseMessage, SystemMessage

            messages: list[BaseMessage] = [SystemMessage(content=self._system_template)]

            agent_name = kwargs.get("agent_name", self._agent_name)
            if agent_name in input.get("conversation_history", {}):
                messages.extend(input["conversation_history"][agent_name])

            response = await self._model.ainvoke(messages)
            return response

    return RealLLMPromptAdapter


make_prompt_adapter_class = _make_prompt_adapter_class


@pytest.fixture
def execution_model(request):
    """Model name for agent execution, from --execution-model CLI option."""
    return request.config.getoption("--execution-model")


@pytest.fixture
def validation_model(request):
    """Model name for LLM-as-judge validation, from --validation-model CLI option."""
    return request.config.getoption("--validation-model")


@lru_cache(maxsize=None)
def _production_params(model_name: str) -> ChatAnthropicParams:
    """Params from the direct-Anthropic models.yml entry matching this API model name.

    Raises:
        ValueError: if no anthropic-provider entry in models.yml matches model_name.
    """
    from ai_gateway.model_selection import ModelSelectionConfig
    from ai_gateway.model_selection.models import ModelClassProvider

    definitions = ModelSelectionConfig(default_models_override={}).get_llm_definitions()
    for definition in definitions.values():
        if (
            definition.model_class_provider == ModelClassProvider.ANTHROPIC
            and definition.params.model == model_name
        ):
            return definition.params
    raise ValueError(
        f"No model_class_provider=anthropic entry in ai_gateway/model_selection/"
        f"models.yml has params.model == {model_name!r}. Agent tests only run "
        f"against production-configured models."
    )


@pytest.fixture
def real_llm(execution_model):
    """Real Anthropic model configured via --execution-model.

    Temperature is read from the direct-Anthropic row (model_class_provider: anthropic) in
    ai_gateway/model_selection/models.yml for the model name, because agent tests call the Anthropic API directly.
    Production resolves model_tags to the vertex/litellm row, which shares the same temperature today. Only
    temperature is applied from the row; other params (top_p, top_k, extra_headers) are not, and max_tokens=4096 is a
    deliberate test cost cap versus production's higher limit.
    """
    kwargs: dict[str, Any] = {"model": execution_model, "max_tokens": 4096}
    params = _production_params(execution_model)
    if params.temperature is not None:
        kwargs["temperature"] = params.temperature
    return ChatAnthropic(**kwargs)  # type: ignore[call-arg]


@pytest.fixture
def mock_tools_registry():
    """Tools registry that doesn't require approval."""
    from duo_workflow_service.components.tools_registry import ToolsRegistry

    registry = MagicMock(spec=ToolsRegistry)
    registry.approval_required.return_value = False
    return registry


@pytest.fixture
def initial_state():
    """Factory for creating initial workflow state.

    The agent_name parameter controls the conversation history key.
    """

    def _create_state(goal: str, agent_name: str = "agent") -> ChatWorkflowState:
        from langchain_core.messages import HumanMessage

        from duo_workflow_service.entities.state import (
            ChatWorkflowState,
            WorkflowStatusEnum,
        )

        return ChatWorkflowState(
            plan={"steps": []},
            status=WorkflowStatusEnum.EXECUTION,
            conversation_history={agent_name: [HumanMessage(content=goal)]},
            ui_chat_log=[],
            last_human_input=None,
            goal=goal,
            project=None,
            namespace=None,
            approval=None,
            preapproved_tools=[],
            denied_tools=[],
        )

    return _create_state


# ===== Mock Orbit MCP tools =====
# The real Orbit MCP tools (`orbit_list_commands`, `orbit_invoke_command`) are
# injected at runtime by Rails/Workhorse — no Python tool classes exist in
# this repo to import. These mocks mirror the real tool names, descriptions,
# and shapes so the LLM sees a realistic surface during tests. The canned
# graph schema reflects the live Orbit graph as of 2026-05-14 — 6 domains,
# the most-traversed edges — so routing decisions are made against
# representative data.


class OrbitListCommandsInput(BaseModel):
    command_names: list[str] | None = Field(default=None)
    format: str = Field(default="llm")


class MockOrbitListCommands(BaseTool):
    """Mock of orbit_list_commands — lists available Orbit commands."""

    name: str = "orbit_list_commands"
    description: str = (
        "List Orbit Knowledge Graph commands with descriptions and input "
        "schemas. Use this before invoke_command to discover available "
        "command details."
    )
    args_schema: Type[BaseModel] = OrbitListCommandsInput

    def _run(self, **_: Any) -> str:
        return json.dumps(
            {
                "commands": [
                    {
                        "name": "query_graph",
                        "description": (
                            "Execute a graph query. Before composing a "
                            "query, call get_query_dsl for the DSL and "
                            "get_graph_schema for the node and edge names."
                        ),
                    },
                    {
                        "name": "get_graph_schema",
                        "description": (
                            "Return the graph schema. Use expand_nodes for "
                            "node types to include properties and "
                            "relationships."
                        ),
                    },
                    {
                        "name": "get_query_dsl",
                        "description": "Return the query_graph JSON DSL grammar.",
                    },
                    {
                        "name": "get_response_format",
                        "description": (
                            "Return the JSON Schema for query_graph responses."
                        ),
                    },
                ]
            }
        )

    async def _arun(self, **kwargs: Any) -> str:
        return self._run(**kwargs)


class OrbitInvokeCommandInput(BaseModel):
    command_name: str
    parameters: dict[str, Any] | None = Field(default=None)


_ORBIT_COMMAND_RESPONSES: dict[str, dict[str, Any]] = {
    "get_graph_schema": {
        "domains": [
            {
                "name": "ci",
                "nodes": [
                    "Deployment",
                    "Environment",
                    "Job",
                    "JobMetadata",
                    "Pipeline",
                    "Runner",
                    "Stage",
                ],
            },
            {
                "name": "code_review",
                "nodes": [
                    "MergeRequest",
                    "MergeRequestDiff",
                    "MergeRequestDiffFile",
                ],
            },
            {"name": "core", "nodes": ["Group", "Note", "Project", "User"]},
            {"name": "plan", "nodes": ["Label", "Milestone", "WorkItem"]},
            {
                "name": "security",
                "nodes": [
                    "Finding",
                    "SecurityScan",
                    "Vulnerability",
                    "VulnerabilityIdentifier",
                    "VulnerabilityOccurrence",
                    "VulnerabilityScanner",
                ],
            },
            {
                "name": "source_code",
                "nodes": [
                    "Branch",
                    "Definition",
                    "Directory",
                    "File",
                    "ImportedSymbol",
                ],
            },
        ],
        "edges": [
            {"name": "APPROVED", "from": ["User"], "to": ["MergeRequest"]},
            {
                "name": "ASSIGNED",
                "from": ["User"],
                "to": ["MergeRequest", "WorkItem"],
            },
            {
                "name": "AUTHORED",
                "from": ["User"],
                "to": ["MergeRequest", "Note", "Vulnerability", "WorkItem"],
            },
            {
                "name": "CALLS",
                "from": ["Definition", "File"],
                "to": ["Definition", "ImportedSymbol"],
            },
            {
                "name": "CLOSED",
                "from": ["User"],
                "to": ["MergeRequest", "WorkItem"],
            },
            {"name": "CLOSES", "from": ["MergeRequest"], "to": ["WorkItem"]},
            {
                "name": "CONTAINS",
                "from": ["Branch", "Directory", "Group", "Project", "WorkItem"],
                "to": ["Branch", "Directory", "File", "Group", "Project", "WorkItem"],
            },
            {
                "name": "DEFINES",
                "from": ["File", "Definition"],
                "to": ["Definition"],
            },
            {"name": "DEPLOYED_TO", "from": ["MergeRequest"], "to": ["Deployment"]},
            {"name": "EXTENDS", "from": ["Definition"], "to": ["Definition"]},
            {"name": "FIXES", "from": ["MergeRequest"], "to": ["Vulnerability"]},
            {"name": "HAS_DIFF", "from": ["MergeRequest"], "to": ["MergeRequestDiff"]},
            {
                "name": "HAS_FILE",
                "from": ["MergeRequestDiff"],
                "to": ["MergeRequestDiffFile"],
            },
            {"name": "HAS_JOB", "from": ["Pipeline", "Stage"], "to": ["Job"]},
            {
                "name": "HAS_LABEL",
                "from": ["MergeRequest", "WorkItem"],
                "to": ["Label"],
            },
            {
                "name": "HAS_NOTE",
                "from": ["MergeRequest", "Vulnerability", "WorkItem"],
                "to": ["Note"],
            },
            {"name": "HAS_STAGE", "from": ["Pipeline"], "to": ["Stage"]},
            {
                "name": "IMPORTS",
                "from": ["File", "ImportedSymbol"],
                "to": ["Definition", "ImportedSymbol"],
            },
            {
                "name": "IN_GROUP",
                "from": ["Label", "Milestone", "WorkItem"],
                "to": ["Group"],
            },
            {
                "name": "IN_MILESTONE",
                "from": ["MergeRequest", "WorkItem"],
                "to": ["Milestone"],
            },
            {
                "name": "IN_PROJECT",
                "from": [
                    "Branch",
                    "Deployment",
                    "Environment",
                    "Job",
                    "Label",
                    "MergeRequest",
                    "Milestone",
                    "Pipeline",
                    "Vulnerability",
                    "WorkItem",
                ],
                "to": ["Project"],
            },
            {"name": "MEMBER_OF", "from": ["User"], "to": ["Group", "Project"]},
            {"name": "MERGED", "from": ["User"], "to": ["MergeRequest"]},
            {"name": "RELATED_TO", "from": ["WorkItem"], "to": ["WorkItem"]},
            {"name": "REVIEWER", "from": ["User"], "to": ["MergeRequest"]},
            {
                "name": "TRIGGERED",
                "from": ["MergeRequest", "User"],
                "to": ["Job", "Pipeline"],
            },
        ],
    },
    "get_query_dsl": {
        "version": "0.1",
        "grammar": {
            "type": "object",
            "properties": {
                "node_type": {"type": "string"},
                "filters": {"type": "object"},
                "expand": {"type": "array"},
                "node_ids": {"type": "array"},
            },
        },
    },
    "get_response_format": {
        "version": "0.1",
        "schema": {
            "type": "object",
            "properties": {
                "results": {"type": "array"},
                "page_info": {"type": "object"},
            },
        },
    },
    "query_graph": {
        "results": [
            {
                "id": "gid://gitlab/MergeRequest/1",
                "title": "Refactor GLQL frontend renderer",
            },
            {
                "id": "gid://gitlab/MergeRequest/2",
                "title": "Fix GLQL embedded view bug",
            },
        ]
    },
}


class MockOrbitInvokeCommand(BaseTool):
    """Mock of orbit_invoke_command — runs a named Orbit command."""

    name: str = "orbit_invoke_command"
    description: str = (
        "Execute an Orbit command. This is a wrapper tool: keep only "
        "command_name and parameters at the top level, and put downstream "
        "command inputs inside parameters."
    )
    args_schema: Type[BaseModel] = OrbitInvokeCommandInput

    def _run(self, command_name: str, parameters: dict | None = None) -> str:
        del parameters  # unused — mock ignores command-specific inputs
        response = _ORBIT_COMMAND_RESPONSES.get(
            command_name,
            {"error": f"Unknown orbit command: {command_name}"},
        )
        return json.dumps(response)

    async def _arun(self, command_name: str, parameters: dict | None = None) -> str:
        return self._run(command_name, parameters)


@pytest.fixture
def orbit_list_commands_tool():
    """Mock orbit_list_commands tool — reusable across agent test suites."""
    return MockOrbitListCommands()


@pytest.fixture
def orbit_invoke_command_tool():
    """Mock orbit_invoke_command tool — reusable across agent test suites."""
    return MockOrbitInvokeCommand()


# ===== Pass-rate summary =====
# Agent tests are benchmarks as much as they are tests: a run is only useful if
# its score can be compared against previous runs and against other agent
# variants. These hooks record the final outcome of every test, grouped by test
# file, and write a per-suite summary to .test-reports/agent_tests/.
#
# Retries from pytest-rerunfailures are ignored so a flaky-then-passing test
# counts once. Skipped tests are reported but excluded from the pass rate
# denominator, because suites skip rules that do not apply to a given case.

_OUTCOMES: dict[str, str] = {}


def pytest_runtest_logreport(report):
    """Record the final outcome of each agent test, ignoring reruns."""
    if report.outcome == "rerun" or not report.nodeid.startswith("agent_tests/"):
        return

    if report.when == "call":
        _OUTCOMES[report.nodeid] = report.outcome
    elif report.when == "setup" and report.outcome == "skipped":
        # A module or fixture level skip never reaches the call phase.
        _OUTCOMES[report.nodeid] = "skipped"
    elif report.when == "setup" and report.outcome == "failed":
        # A fixture that raises is an error, not a test-level failure.
        _OUTCOMES[report.nodeid] = "error"
    elif report.when == "teardown" and report.outcome == "failed":
        _OUTCOMES.setdefault(report.nodeid, "error")


def _suite_and_file(nodeid: str) -> tuple[str, str]:
    """Split an agent test nodeid into its suite directory and test file name."""
    path = nodeid.split("::", maxsplit=1)[0]
    parts = Path(path).parts
    # agent_tests/<suite>/<file>.py — fall back to the directory name for tests
    # that live directly under agent_tests/.
    suite = parts[1] if len(parts) > 2 else "agent_tests"
    return suite, Path(path).name


NO_CASE = "(no case)"


def _case_id(nodeid: str) -> str:
    """Return the parametrize id a row scores, or a placeholder when the test takes no parameters.

    Agent suites parametrize one test per benchmark case, so the parametrize id names the case. pytest-xdist's
    loadgroup scheduler appends "@<group>" after the closing bracket, which is dropped here.
    """
    name = nodeid.split("::", maxsplit=1)[-1]
    if "]" in name:
        name = name[: name.rindex("]") + 1]
        if "[" in name:
            return name[name.index("[") + 1 : -1]
    return NO_CASE


def _summarize() -> dict[str, dict[str, dict[str, int]]]:
    """Aggregate recorded outcomes into {suite: {test_file: {outcome: count}}}."""
    summary: dict[str, dict[str, dict[str, int]]] = {}
    for nodeid, outcome in sorted(_OUTCOMES.items()):
        suite, test_file = _suite_and_file(nodeid)
        counts = summary.setdefault(suite, {}).setdefault(
            test_file, {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
        )
        counts[outcome] = counts.get(outcome, 0) + 1
    return summary


def _summarize_cases() -> dict[str, dict[str, dict[str, int]]]:
    """Aggregate recorded outcomes into {suite: {case: {outcome: count}}}."""
    summary: dict[str, dict[str, dict[str, int]]] = {}
    for nodeid, outcome in sorted(_OUTCOMES.items()):
        suite, _ = _suite_and_file(nodeid)
        counts = summary.setdefault(suite, {}).setdefault(
            _case_id(nodeid), {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
        )
        counts[outcome] = counts.get(outcome, 0) + 1
    return summary


def _pass_rate(counts: dict[str, int]) -> tuple[int, int]:
    """Return (passed, scored) where scored excludes skipped tests."""
    passed = counts["passed"]
    scored = passed + counts["failed"] + counts["error"]
    return passed, scored


def _format_rate(passed: int, scored: int) -> str:
    """Render a pass rate, or "n/a" when nothing was scored."""
    if scored == 0:
        return "n/a"
    return f"{100 * passed / scored:.0f}% ({passed}/{scored})"


def _render_rows(header: str, rows: dict[str, dict[str, int]]) -> list[str]:
    """Render one pass-rate table as Markdown lines, with a totals row."""
    lines = [
        f"| {header} | Passed | Failed | Errors | Skipped | Pass rate |",
        "|---|---|---|---|---|---|",
    ]
    totals = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
    for label, counts in sorted(rows.items()):
        for key in totals:
            totals[key] += counts[key]
        passed, scored = _pass_rate(counts)
        lines.append(
            f"| `{label}` | {counts['passed']} | {counts['failed']} | "
            f"{counts['error']} | {counts['skipped']} | "
            f"{_format_rate(passed, scored)} |"
        )

    passed, scored = _pass_rate(totals)
    lines.append(
        f"| **Total** | {totals['passed']} | {totals['failed']} | "
        f"{totals['error']} | {totals['skipped']} | "
        f"**{_format_rate(passed, scored)}** |"
    )
    return lines


def _render_markdown(
    suite: str,
    files: dict[str, dict[str, int]],
    cases: dict[str, dict[str, int]],
    meta: dict,
) -> str:
    """Render one suite's pass rate as a Markdown report."""
    lines = [
        f"# Agent test pass rate — `{suite}`",
        "",
        f"- Run at: {meta['run_at']}",
        f"- Execution model: `{meta['execution_model']}`",
        f"- Validation model: `{meta['validation_model']}`",
        "",
        "## By test file",
        "",
    ]
    lines.extend(_render_rows("Test file", files))
    lines.extend(
        [
            "",
            "## By case",
            "",
            "Which cases regressed is more stable than the headline rate, so compare "
            "this table first when scoring a variant.",
            "",
        ]
    )
    lines.extend(_render_rows("Case", cases))
    lines.extend(
        [
            "",
            "Skipped tests are excluded from the pass rate denominator.",
            "",
        ]
    )
    return "\n".join(lines)


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Print and persist a per-test-file pass rate for each agent test suite."""
    del exitstatus

    if hasattr(config, "workerinput") or not _OUTCOMES:
        # Only the xdist controller (or a plain single-process run) reports.
        return

    meta = {
        "run_at": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"
        ),
        "execution_model": config.getoption("--execution-model"),
        "validation_model": config.getoption("--validation-model"),
    }

    summary = _summarize()
    case_summary = _summarize_cases()
    terminalreporter.write_sep("=", "agent test pass rate")

    for suite, files in sorted(summary.items()):
        for test_file, counts in sorted(files.items()):
            passed, scored = _pass_rate(counts)
            terminalreporter.write_line(
                f"{suite}/{test_file}: {_format_rate(passed, scored)} "
                f"passed, {counts['skipped']} skipped"
            )

        cases = case_summary.get(suite, {})
        # Only worth printing when the suite parametrizes by case; an
        # unparametrized suite would just repeat the per-file numbers.
        if set(cases) - {NO_CASE}:
            terminalreporter.write_line(f"{suite} by case:")
            for case_id, counts in sorted(cases.items()):
                passed, scored = _pass_rate(counts)
                terminalreporter.write_line(
                    f"  {case_id}: {_format_rate(passed, scored)} "
                    f"passed, {counts['skipped']} skipped"
                )

        report = {"suite": suite, **meta, "test_files": files, "cases": cases}
        try:
            REPORT_DIR.mkdir(parents=True, exist_ok=True)
            (REPORT_DIR / f"{suite}-summary.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            (REPORT_DIR / f"{suite}-summary.md").write_text(
                _render_markdown(suite, files, cases, meta)
            )
        except OSError as exc:  # pragma: no cover - reporting must never fail a run
            terminalreporter.write_line(f"Could not write pass-rate summary: {exc}")
        else:
            terminalreporter.write_line(f"Wrote {REPORT_DIR / f'{suite}-summary.md'}")
