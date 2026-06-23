# pylint: disable=file-naming-for-tests
"""Tests for tool approval functionality in SupervisorAgentComponent."""

from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END

from duo_workflow_service.agent_platform.v1.components.agent.component import (
    RoutingError,
)
from duo_workflow_service.agent_platform.v1.state import FlowEventType, FlowStateKeys
from duo_workflow_service.entities.state import WorkflowStatusEnum

from .conftest import _AGENT_COMPONENT_MODULE, _compile

# --- Tool approval node mock fixtures ---


@pytest.fixture(name="mock_tool_approval_request_node_cls")
def mock_tool_approval_request_node_cls_fixture(supervisor_name):
    """Fixture for mocked ToolApprovalRequestNode class in agent component module."""
    with patch(f"{_AGENT_COMPONENT_MODULE}.ToolApprovalRequestNode") as mock_cls:
        mock_cls.return_value.name = f"{supervisor_name}#tool_approval_request"
        yield mock_cls


@pytest.fixture(name="mock_tool_approval_fetch_node_cls")
def mock_tool_approval_fetch_node_cls_fixture(supervisor_name):
    """Fixture for mocked ToolApprovalFetchNode class in agent component module."""
    with patch(f"{_AGENT_COMPONENT_MODULE}.ToolApprovalFetchNode") as mock_cls:
        mock_cls.return_value.name = f"{supervisor_name}#tool_approval_fetch"
        yield mock_cls


@pytest.fixture(name="all_approval_node_mocks")
def all_approval_node_mocks_fixture(
    mock_agent_node_cls,
    mock_tool_node_cls,
    mock_final_response_node_cls,
    mock_delegation_node_cls,
    mock_subagent_return_node_cls,
    mock_tool_approval_request_node_cls,
    mock_tool_approval_fetch_node_cls,
):
    """Activate all supervisor node mocks including tool approval nodes."""
    return {
        "agent": mock_agent_node_cls.return_value,
        "tools": mock_tool_node_cls.return_value,
        "final_response": mock_final_response_node_cls.return_value,
        "delegation": mock_delegation_node_cls.return_value,
        "subagent_return": mock_subagent_return_node_cls.return_value,
        "tool_approval_request": mock_tool_approval_request_node_cls.return_value,
        "tool_approval_fetch": mock_tool_approval_fetch_node_cls.return_value,
    }


class TestSupervisorAgentComponentToolApproval:
    """Tests for tool approval functionality in SupervisorAgentComponent.

    All tests use a real compiled StateGraph (via ``_compile``) and assert on
    which nodes were actually visited during execution.  This catches wiring
    errors (e.g. a node referenced in a conditional edge that was never added)
    that a mocked graph cannot detect.
    """

    def test_agent_routes_to_tool_approval_request_for_regular_tools(
        self,
        all_approval_node_mocks,
        mock_router,
        base_flow_state,
        supervisor_name,
        regular_tool_call,
        make_supervisor,
    ):
        """Agent routes to #tool_approval_request for regular tool calls when approval is enabled."""
        nodes = all_approval_node_mocks

        supervisor = make_supervisor(require_tool_approval=True, pre_approved_tools=[])

        # First agent call: regular tool call → should route to tool_approval_request
        # Second agent call (after approval loop): text-only → exit
        nodes["agent"].run.side_effect = [
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    supervisor_name: [
                        AIMessage(content="", tool_calls=[regular_tool_call])
                    ]
                },
            },
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    supervisor_name: [AIMessage(content="All done.", tool_calls=[])]
                },
            },
        ]
        # tool_approval_request sets status=TOOL_CALL_APPROVAL_REQUIRED → routes to fetch
        nodes["tool_approval_request"].run.return_value = {
            **base_flow_state,
            "status": WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED,
        }
        # tool_approval_fetch sets decision=REJECT → routes back to agent
        nodes["tool_approval_fetch"].run.return_value = {
            **base_flow_state,
            "context": {
                supervisor_name: {
                    "tool_approval_decision": FlowEventType.REJECT,
                }
            },
        }
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        compiled = _compile(supervisor, mock_router)
        compiled.invoke(base_flow_state)

        nodes["tool_approval_request"].run.assert_called_once()
        nodes["tool_approval_fetch"].run.assert_called_once()
        nodes["tools"].run.assert_not_called()

    def test_approval_nodes_not_visited_when_disabled(
        self,
        all_approval_node_mocks,
        mock_router,
        base_flow_state,
        supervisor_name,
        regular_tool_call,
        make_supervisor,
    ):
        """When require_tool_approval=False, tool approval nodes are never visited."""
        nodes = all_approval_node_mocks

        supervisor = make_supervisor()  # require_tool_approval=False by default

        nodes["agent"].run.side_effect = [
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    supervisor_name: [
                        AIMessage(content="", tool_calls=[regular_tool_call])
                    ]
                },
            },
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    supervisor_name: [AIMessage(content="All done.", tool_calls=[])]
                },
            },
        ]
        nodes["tools"].run.return_value = {**base_flow_state}
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        compiled = _compile(supervisor, mock_router)
        compiled.invoke(base_flow_state)

        nodes["tool_approval_request"].run.assert_not_called()
        nodes["tool_approval_fetch"].run.assert_not_called()
        nodes["tools"].run.assert_called_once()

    @pytest.mark.parametrize(
        ("decision", "expect_tools_called"),
        [
            (FlowEventType.APPROVE, True),
            (FlowEventType.REJECT, False),
            (FlowEventType.MODIFY, False),
        ],
        ids=["approve", "reject", "modify"],
    )
    def test_tool_approval_fetch_router(
        self,
        all_approval_node_mocks,
        mock_router,
        base_flow_state,
        supervisor_name,
        regular_tool_call,
        make_supervisor,
        decision,
        expect_tools_called,
    ):
        """_tool_approval_fetch_router routes to #tools on APPROVE and to #agent on REJECT/MODIFY."""
        nodes = all_approval_node_mocks

        supervisor = make_supervisor(require_tool_approval=True, pre_approved_tools=[])

        nodes["agent"].run.side_effect = [
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    supervisor_name: [
                        AIMessage(content="", tool_calls=[regular_tool_call])
                    ]
                },
            },
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    supervisor_name: [AIMessage(content="All done.", tool_calls=[])]
                },
            },
        ]
        nodes["tool_approval_request"].run.return_value = {
            **base_flow_state,
            "status": WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED,
        }
        nodes["tool_approval_fetch"].run.return_value = {
            **base_flow_state,
            "context": {
                supervisor_name: {
                    "tool_approval_decision": decision,
                }
            },
        }
        nodes["tools"].run.return_value = {**base_flow_state}
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        compiled = _compile(supervisor, mock_router)
        compiled.invoke(base_flow_state)

        if expect_tools_called:
            nodes["tools"].run.assert_called_once()
        else:
            nodes["tools"].run.assert_not_called()

    @pytest.mark.parametrize(
        ("request_status", "fetch_context", "expected_error"),
        [
            (
                WorkflowStatusEnum.PAUSED,
                None,
                "Unexpected approval status",
            ),
            (
                WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED,
                {},
                "No approval decision found in state",
            ),
            (
                WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED,
                {"tool_approval_decision": "unknown_decision"},
                "Unexpected approval decision",
            ),
        ],
        ids=[
            "request_router_unexpected_status",
            "fetch_router_missing_decision",
            "fetch_router_unexpected_decision",
        ],
    )
    def test_approval_router_raises_on_invalid_state(
        self,
        all_approval_node_mocks,
        mock_router,
        base_flow_state,
        supervisor_name,
        regular_tool_call,
        make_supervisor,
        request_status,
        fetch_context,
        expected_error,
    ):
        """Approval routers raise RoutingError for invalid status or decision values."""
        nodes = all_approval_node_mocks

        supervisor = make_supervisor(require_tool_approval=True, pre_approved_tools=[])

        nodes["agent"].run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                supervisor_name: [AIMessage(content="", tool_calls=[regular_tool_call])]
            },
        }
        nodes["tool_approval_request"].run.return_value = {
            **base_flow_state,
            "status": request_status,
        }
        if fetch_context is not None:
            nodes["tool_approval_fetch"].run.return_value = {
                **base_flow_state,
                "context": {supervisor_name: fetch_context},
            }

        compiled = _compile(supervisor, mock_router)
        with pytest.raises(RoutingError, match=expected_error):
            compiled.invoke(base_flow_state)

    @pytest.mark.parametrize(
        ("status", "expect_fetch_called", "expect_tools_called"),
        [
            (WorkflowStatusEnum.TOOL_CALL_APPROVAL_REQUIRED, True, False),
            (WorkflowStatusEnum.EXECUTION, False, True),
        ],
        ids=["needs_approval", "pre_approved"],
    )
    def test_tool_approval_request_router(
        self,
        all_approval_node_mocks,
        mock_router,
        base_flow_state,
        supervisor_name,
        regular_tool_call,
        make_supervisor,
        status,
        expect_fetch_called,
        expect_tools_called,
    ):
        """_tool_approval_request_router routes to #tool_approval_fetch or #tools depending on status."""
        nodes = all_approval_node_mocks

        supervisor = make_supervisor(require_tool_approval=True, pre_approved_tools=[])

        nodes["agent"].run.side_effect = [
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    supervisor_name: [
                        AIMessage(content="", tool_calls=[regular_tool_call])
                    ]
                },
            },
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    supervisor_name: [AIMessage(content="All done.", tool_calls=[])]
                },
            },
        ]
        nodes["tool_approval_request"].run.return_value = {
            **base_flow_state,
            "status": status,
        }
        # Only needed when fetch is called (TOOL_CALL_APPROVAL_REQUIRED path)
        nodes["tool_approval_fetch"].run.return_value = {
            **base_flow_state,
            "context": {
                supervisor_name: {
                    "tool_approval_decision": FlowEventType.REJECT,
                }
            },
        }
        nodes["tools"].run.return_value = {**base_flow_state}
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        compiled = _compile(supervisor, mock_router)
        compiled.invoke(base_flow_state)

        if expect_fetch_called:
            nodes["tool_approval_fetch"].run.assert_called_once()
        else:
            nodes["tool_approval_fetch"].run.assert_not_called()

        if expect_tools_called:
            nodes["tools"].run.assert_called_once()
        else:
            nodes["tools"].run.assert_not_called()
