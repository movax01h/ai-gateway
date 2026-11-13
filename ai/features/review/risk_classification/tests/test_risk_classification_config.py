"""Tests for the submit step of the risk_classification 2.0.0 flow config.

The step runs as a DeterministicStepComponent, so these tests build it from the shipped config, attach it to a real
graph with the real submit tool, and check that the classifier's answer reaches GitLab unchanged.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from langgraph.graph import END, StateGraph

from ai.features.review.risk_classification.components.submit_merge_request_risk_classification import (
    RESULTS_PATH,
    SubmitMergeRequestRiskClassification,
)
from duo_workflow_service.agent_platform.v1.components.deterministic_step.component import (
    DeterministicStepComponent,
)
from duo_workflow_service.agent_platform.v1.flows import flow_config
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from duo_workflow_service.agent_platform.v1.state import FlowState
from duo_workflow_service.components.tools_registry import agent_privileges_for_tool
from duo_workflow_service.entities.state import WorkflowStatusEnum
from duo_workflow_service.gitlab.http_client import GitLabHttpResponse
from duo_workflow_service.tools.toolset import Toolset
from lib.events import GLReportingEventContext
from lib.internal_events import InternalEventsClient

TOOL_NAME = "submit_merge_request_risk_classification"
DIFF_SHA = "b83d6e391c22777fca1ed3012fce84f633d7fed0"
CLAIMS = [
    {
        "name": "authorization",
        "value": "true",
        "evidence": "app/policies/project_policy.rb:44",
    },
    {"name": "database_migration", "value": "false", "evidence": None},
]
SUMMARY = "Adds a session token refresh path."

# Privileges that grant tools which need a repository checkout: files, commands and git.
CHECKOUT_PRIVILEGES = {"read_only_files", "read_write_files", "run_commands", "use_git"}


def no_checkout_versions() -> list[str]:
    """Every shipped version from 2.0.0 on, taken from the config filenames.

    Rails runs these versions with coding_environment none, so none of them may use a tool that needs a checkout.
    """
    config_dir = Path(__file__).parent.parent / "config"
    return sorted(
        path.stem
        for path in config_dir.glob("*.yml")
        if int(path.stem.split(".")[0]) >= 2
    )


@pytest.fixture(name="discover_flow_configs", autouse=True)
def discover_flow_configs_fixture():
    # The flow lives under ai/features/, which the service registers at startup.
    # Restore the registry so tests that list flow configs never see these roots.
    with patch.dict(flow_config._FEATURE_FLOW_ROOTS):
        flow_config.discover_feature_flow_configs()
        yield


@pytest.fixture(name="gitlab_client")
def gitlab_client_fixture():
    client = Mock()
    client.apost = AsyncMock(
        return_value=GitLabHttpResponse(status_code=204, body=None)
    )
    return client


@pytest.fixture(name="submit_step")
def submit_step_fixture(gitlab_client):
    config = FlowConfig.from_yaml_config("risk_classification", "2.0.0")
    step = next(
        c for c in config.components if c["name"] == "submit_risk_classification"
    )
    tool = SubmitMergeRequestRiskClassification(
        metadata={
            "gitlab_client": gitlab_client,
            "gitlab_host": "gitlab.com",
            "workflow_id": "4242",
        }
    )

    return DeterministicStepComponent(
        name=step["name"],
        flow_id="4242",
        flow_type=GLReportingEventContext.from_workflow_definition(
            "risk_classification"
        ),
        user=CloudConnectorUser(
            authenticated=True,
            is_debug=False,
            claims=UserClaims(scopes=[], gitlab_instance_uid="unique-instance-uid"),
        ),
        inputs=step["inputs"],
        tool_name=step["tool_name"],
        toolset=Toolset(pre_approved={TOOL_NAME}, all_tools={TOOL_NAME: tool}),
        ui_log_events=step["ui_log_events"],
        internal_event_client=Mock(spec=InternalEventsClient),
    )


@pytest.fixture(name="compiled_graph")
def compiled_graph_fixture(submit_step):
    router = Mock()
    router.route = Mock(return_value=END)

    graph = StateGraph(FlowState)
    submit_step.attach(graph, router)
    graph.set_entry_point(submit_step.__entry_hook__())
    return graph.compile()


@pytest.fixture(name="initial_state")
def initial_state_fixture():
    return {
        "status": WorkflowStatusEnum.EXECUTION,
        "conversation_history": {},
        "ui_chat_log": [],
        "context": {
            "project_id": 13,
            "inputs": {
                "agent_platform_resource_context": {
                    # Rails sends the IID as a string.
                    "merge_request_id": "9",
                    "merge_request_diff_sha": DIFF_SHA,
                }
            },
            "classify_risk": {
                "final_answer": {"claims": CLAIMS, "summary": SUMMARY},
            },
        },
    }


def test_no_checkout_versions_are_discovered():
    """A bad glob would silently parametrize the guard below into nothing."""
    assert "2.0.0" in no_checkout_versions()


@pytest.mark.parametrize("version", no_checkout_versions())
def test_no_checkout_version_uses_only_gitlab_api_tools(version):
    """A checkout tool such as read_file needs a new major version, and Rails must drop coding_environment none."""
    config = FlowConfig.from_yaml_config("risk_classification", version)
    tool_names = {
        tool
        for component in config.components
        for tool in [*component.get("toolset", []), component.get("tool_name")]
        if tool
    }

    for tool in sorted(tool_names):
        privileges = set(agent_privileges_for_tool(tool))
        assert privileges, f"{tool} is not granted by any agent privilege"
        assert not privileges & CHECKOUT_PRIVILEGES, (
            f"{tool} needs a repository checkout ({sorted(privileges & CHECKOUT_PRIVILEGES)})"
        )


def test_submit_step_is_deterministic():
    config = FlowConfig.from_yaml_config("risk_classification", "2.0.0")
    step = next(
        c for c in config.components if c["name"] == "submit_risk_classification"
    )

    assert step["type"] == "DeterministicStepComponent"
    assert step["tool_name"] == TOOL_NAME


def test_classify_step_limits_its_tool_use():
    """The default of 280 cycles with no identical-call guard lets a run read files without bound."""
    config = FlowConfig.from_yaml_config("risk_classification", "2.0.0")
    step = next(c for c in config.components if c["name"] == "classify_risk")

    assert step["max_cycles"] == 10
    assert step["identical_tool_call_limit"] == 3


@pytest.mark.parametrize(
    "from_component,success_target",
    [
        ("fetch_mr_diffs", "classify_risk"),
        ("submit_risk_classification", "end"),
    ],
)
def test_deterministic_step_failure_aborts(from_component, success_target):
    """A failed step records execution_result="failed" and returns normally, so only the router can stop the flow."""
    config = FlowConfig.from_yaml_config("risk_classification", "2.0.0")
    router = next(r for r in config.routers if r.get("from") == from_component)

    assert "to" not in router
    assert router["condition"]["input"] == f"context:{from_component}.execution_result"
    assert router["condition"]["routes"] == {
        "success": success_target,
        "default_route": "abort",
    }


@pytest.mark.asyncio
async def test_submit_step_sends_the_classification_unchanged(
    compiled_graph, initial_state, gitlab_client
):
    result = await compiled_graph.ainvoke(initial_state)

    assert result["context"]["submit_risk_classification"]["execution_result"] == (
        "success"
    )
    gitlab_client.apost.assert_called_once()
    assert gitlab_client.apost.call_args.kwargs["path"] == RESULTS_PATH
    assert json.loads(gitlab_client.apost.call_args.kwargs["body"]) == {
        "project_id": 13,
        "merge_request_iid": 9,
        "diff_sha": DIFF_SHA,
        "workflow_id": 4242,
        "claims": CLAIMS,
        "summary": SUMMARY,
    }


@pytest.mark.asyncio
async def test_submit_step_records_a_rejected_submission(
    compiled_graph, initial_state, gitlab_client
):
    gitlab_client.apost.return_value = GitLabHttpResponse(
        status_code=422, body={"message": "diff_sha is not a revision of this MR"}
    )

    result = await compiled_graph.ainvoke(initial_state)

    step_state = result["context"]["submit_risk_classification"]
    assert step_state["execution_result"] == "failed"
    assert step_state["error"]
