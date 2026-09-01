"""Test suite for SupervisorAgentComponentV2."""

from unittest.mock import Mock, patch

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END

from duo_workflow_service.agent_platform.v1.components.agent.component import (
    AgentComponentBase,
    MaxCyclesConfig,
    RoutingError,
)
from duo_workflow_service.agent_platform.v1.components.agent.ui_log import (
    UILogEventsAgent,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.component import (
    SubagentConfig,
    extract_subagent_names,
)
from duo_workflow_service.agent_platform.v1.components.supervisor_v2.ui_log import (
    UILogEventsSupervisor,
)
from duo_workflow_service.agent_platform.v1.state import FlowStateKeys
from duo_workflow_service.agent_platform.v1.state.base import NoneIOKey
from duo_workflow_service.entities.state import MessageTypeEnum
from duo_workflow_service.tracking.subagent_delegation import (
    DelegationRejectionReason,
    SubagentDelegationTracker,
)

from .conftest import (
    _AGENT_COMPONENT_MODULE,
    MockSubagentComponent,
    RoutingMockSubagentComponent,
    _compile,
)


class TestExtractSubagentNames:
    """Tests for extract_subagent_names."""

    def test_extracts_names_in_order(self):
        subagents: list[SubagentConfig] = [{"name": "developer"}, {"name": "tester"}]
        assert extract_subagent_names(subagents) == ["developer", "tester"]

    def test_non_dict_entry_raises(self):
        with pytest.raises(ValueError, match="must be a dict with a 'name' key"):
            extract_subagent_names([{"name": "developer"}, "tester"])

    def test_missing_name_key_raises(self):
        with pytest.raises(ValueError, match="must be a dict with a 'name' key"):
            extract_subagent_names([{"role": "developer"}])

    def test_duplicate_name_raises(self):
        with pytest.raises(ValueError, match="Duplicate subagent name 'developer'"):
            extract_subagent_names([{"name": "developer"}, {"name": "developer"}])


class TestSupervisorAgentComponentV2Init:
    """Tests for SupervisorAgentComponentV2 construction/validation."""

    @pytest.mark.parametrize(
        (
            "subagents_override",
            "max_delegations_override",
            "subagent_components_key",
            "match",
        ),
        [
            ([], 5, "empty", "at least one managed agent"),
            (None, 0, None, "max_delegations must be at least 1"),
            (None, -1, None, "max_delegations must be at least 1"),
            (None, 5, "empty", "not found in subagent_components"),
            (None, 5, "missing_tester", "Managed agent 'tester' not found"),
            (
                None,
                5,
                "wrong_type",
                "does not have a compile_as_subagent method",
            ),
            (
                [{"name": "developer"}, "tester"],
                5,
                None,
                "must be a dict with a 'name' key",
            ),
            (
                [{"name": "developer"}, {"name": "developer"}],
                5,
                None,
                "Duplicate subagent name 'developer'",
            ),
        ],
        ids=[
            "empty_subagents",
            "zero_max_delegations",
            "negative_max_delegations",
            "empty_subagent_components",
            "missing_subagent",
            "wrong_type_subagent",
            "entry_is_plain_string",
            "duplicate_subagent_name",
        ],
    )
    def test_invalid_params_raise(
        self,
        make_supervisor,
        mock_sub_agents,
        developer_name,
        tester_name,
        subagent_names,
        subagents_override,
        max_delegations_override,
        subagent_components_key,
        match,
    ):
        class NonCompilableComponent:
            def __init__(self, name: str):
                self.name = name
                self.description = "Not compilable."

        subagent_components_by_key = {
            None: mock_sub_agents,
            "empty": {},
            "missing_tester": {
                developer_name: MockSubagentComponent(name=developer_name)
            },
            "wrong_type": {
                developer_name: NonCompilableComponent(name=developer_name),
                tester_name: MockSubagentComponent(name=tester_name),
            },
        }
        with pytest.raises(ValueError, match=match):
            make_supervisor(
                subagents=subagents_override
                if subagents_override is not None
                else subagent_names,
                max_delegations=max_delegations_override,
                subagent_components=subagent_components_by_key[subagent_components_key],
            )

    def test_none_subagents_raises(self, make_supervisor):
        with pytest.raises(ValueError, match="at least one managed agent"):
            make_supervisor(subagents=None)

    def test_none_max_delegations_is_valid(self, make_supervisor):
        supervisor = make_supervisor(max_delegations=None)
        assert supervisor.max_delegations is None

    def test_builds_dynamic_delegate_task_cls(self, make_supervisor):
        supervisor = make_supervisor()
        assert supervisor._delegate_task_cls.tool_title == "delegate_task"


class TestSupervisorAgentComponentV2Properties:
    """Tests for SupervisorAgentComponentV2's derived properties."""

    def test_managed_agent_names(self, make_supervisor, developer_name, tester_name):
        supervisor = make_supervisor()
        assert set(supervisor.managed_agent_names) == {developer_name, tester_name}

    def test_subagents_config_derives_name_and_description(
        self,
        make_supervisor,
        developer_name,
        developer_description,
        tester_name,
        tester_description,
    ):
        supervisor = make_supervisor()
        configs = {c["name"]: c["description"] for c in supervisor.subagents_config}
        assert configs[developer_name] == developer_description
        assert configs[tester_name] == tester_description

    def test_outputs_include_final_answer_and_orchestration_metadata(
        self, make_supervisor, supervisor_name
    ):
        supervisor = make_supervisor()
        output_subkey_sets = [tuple(o.subkeys or []) for o in supervisor.outputs]
        assert (supervisor_name, "delegation_count") in output_subkey_sets
        assert (supervisor_name, "max_subsession_id") in output_subkey_sets


class TestAgentNodeRouter:
    """Tests for SupervisorAgentComponentV2._agent_node_router."""

    def test_text_only_response_routes_to_final_response(
        self,
        make_supervisor,
        supervisor_name,
        base_flow_state,
        ai_message_no_tool_calls,
    ):
        supervisor = make_supervisor()
        state = {**base_flow_state}
        state["conversation_history"] = {supervisor_name: [ai_message_no_tool_calls]}

        assert (
            supervisor._agent_node_router(state) == f"{supervisor_name}#final_response"
        )

    def test_delegate_task_call_routes_to_delegation_prepare(
        self,
        make_supervisor,
        supervisor_name,
        base_flow_state,
        ai_message_with_delegate,
    ):
        supervisor = make_supervisor()
        state = {**base_flow_state}
        state["conversation_history"] = {supervisor_name: [ai_message_with_delegate]}

        assert (
            supervisor._agent_node_router(state)
            == f"{supervisor_name}#delegation_prepare"
        )

    def test_regular_tool_call_routes_to_tools(
        self,
        make_supervisor,
        supervisor_name,
        base_flow_state,
        ai_message_with_regular_tool,
    ):
        supervisor = make_supervisor()
        state = {**base_flow_state}
        state["conversation_history"] = {
            supervisor_name: [ai_message_with_regular_tool]
        }

        assert supervisor._agent_node_router(state) == f"{supervisor_name}#tools"

    def test_regular_tool_call_routes_to_tool_approval_when_required(
        self,
        make_supervisor,
        supervisor_name,
        base_flow_state,
        ai_message_with_regular_tool,
    ):
        supervisor = make_supervisor(require_tool_approval=True)
        state = {**base_flow_state}
        state["conversation_history"] = {
            supervisor_name: [ai_message_with_regular_tool]
        }

        assert (
            supervisor._agent_node_router(state)
            == f"{supervisor_name}#tool_approval_request"
        )

    def test_schema_tool_call_routes_to_final_response(
        self,
        make_supervisor,
        supervisor_name,
        base_flow_state,
        mock_toolset,
    ):
        mock_toolset.__contains__ = Mock(
            side_effect=lambda name: name != "custom_response_tool"
        )
        supervisor = make_supervisor(
            response_schema_id="general/structured_response",
            response_schema_version="1.0.0",
        )
        ai_msg = Mock(spec=AIMessage)
        ai_msg.tool_calls = [
            {
                "id": "schema_call",
                "name": "custom_response_tool",
                "args": {"summary": "done", "score": 9},
            }
        ]
        state = {**base_flow_state}
        state["conversation_history"] = {supervisor_name: [ai_msg]}

        assert (
            supervisor._agent_node_router(state) == f"{supervisor_name}#final_response"
        )

    def test_no_history_raises_routing_error(self, make_supervisor, base_flow_state):
        supervisor = make_supervisor()
        with pytest.raises(RoutingError, match="Conversation history not found"):
            supervisor._agent_node_router(base_flow_state)

    def test_last_message_not_ai_message_raises_routing_error(
        self, make_supervisor, supervisor_name, base_flow_state
    ):
        from langchain_core.messages import HumanMessage

        supervisor = make_supervisor()
        state = {**base_flow_state}
        state["conversation_history"] = {supervisor_name: [HumanMessage(content="hi")]}
        with pytest.raises(RoutingError, match="not AIMessage"):
            supervisor._agent_node_router(state)


class TestSupervisorAgentComponentV2Attach:
    """Tests for SupervisorAgentComponentV2.attach's graph wiring."""

    def test_subagents_are_compiled_and_dispatch_nodes_registered(
        self,
        all_node_mocks,
        mock_router,
        mock_sub_agents,
        base_flow_state,
        supervisor_name,
        developer_name,
        tester_name,
        make_supervisor,
    ):
        supervisor = make_supervisor()

        all_node_mocks["agent"].run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                supervisor_name: [AIMessage(content="All done.", tool_calls=[])]
            },
        }
        all_node_mocks["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        _compile(supervisor, mock_router)

        mock_sub_agents[developer_name].compile_as_subagent.assert_called_once()
        mock_sub_agents[tester_name].compile_as_subagent.assert_called_once()

    def test_edges_from_every_subagent_node_reach_delegation_collect(
        self,
        all_node_mocks,
        mock_router,
        mock_state_graph,
        make_supervisor,
        supervisor_name,
        developer_name,
        tester_name,
    ):
        supervisor = make_supervisor()
        supervisor.attach(mock_state_graph, mock_router)

        edge_calls = [call.args for call in mock_state_graph.add_edge.call_args_list]
        assert (developer_name, f"{supervisor_name}#delegation_collect") in edge_calls
        assert (tester_name, f"{supervisor_name}#delegation_collect") in edge_calls

    def test_delegation_prepare_declares_every_destination_it_dispatches_to(
        self,
        all_node_mocks,
        mock_router,
        mock_state_graph,
        make_supervisor,
        supervisor_name,
        developer_name,
        tester_name,
    ):
        """DelegationPrepareNode dispatches via `Command`, so its edges come from `destinations`.

        Without them the compiled graph has no edges out of the node at all -- nothing to render, and nothing for
        LangGraph to validate the targets against.

        The collect node is deliberately not among them: prepare either dispatches subagents (which reach collect by
        their own edges) or, when it rejected every call and answered them itself, returns straight to the agent with
        nothing for collect to collect.
        """
        supervisor = make_supervisor()
        supervisor.attach(mock_state_graph, mock_router)

        prepare = f"{supervisor_name}#delegation_prepare"
        destinations = next(
            call.kwargs["destinations"]
            for call in mock_state_graph.add_node.call_args_list
            if call.args[0] == prepare
        )
        assert set(destinations) == {
            developer_name,
            tester_name,
            f"{supervisor_name}#agent",
        }

    def test_dispatch_node_added_for_every_managed_subagent(
        self,
        all_node_mocks,
        mock_router,
        mock_state_graph,
        make_supervisor,
        developer_name,
        tester_name,
    ):
        supervisor = make_supervisor()
        supervisor.attach(mock_state_graph, mock_router)

        added_node_names = {
            call.args[0] for call in mock_state_graph.add_node.call_args_list
        }
        assert developer_name in added_node_names
        assert tester_name in added_node_names


class TestSupervisorExecutionFlow:
    """Tests for SupervisorAgentComponentV2 execution via a real compiled graph."""

    def test_agent_routes_directly_to_final_response(
        self,
        all_node_mocks,
        mock_router,
        base_flow_state,
        supervisor_name,
        make_supervisor,
    ):
        nodes = all_node_mocks
        supervisor = make_supervisor()

        nodes["agent"].run.return_value = {
            **base_flow_state,
            FlowStateKeys.CONVERSATION_HISTORY: {
                supervisor_name: [AIMessage(content="All done.", tool_calls=[])]
            },
        }
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        compiled = _compile(supervisor, mock_router)
        compiled.invoke(base_flow_state)

        nodes["agent"].run.assert_called_once()
        nodes["final_response"].run.assert_called_once()
        nodes["tools"].run.assert_not_called()
        nodes["delegation_prepare"].run.assert_not_called()
        mock_router.route.assert_called_once()

    def test_agent_routes_to_tools_then_back(
        self,
        all_node_mocks,
        mock_router,
        base_flow_state,
        supervisor_name,
        regular_tool_call,
        make_supervisor,
    ):
        nodes = all_node_mocks
        supervisor = make_supervisor()

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

        assert nodes["agent"].run.call_count == 2
        nodes["tools"].run.assert_called_once()
        nodes["final_response"].run.assert_called_once()

    def test_routing_errors_propagate(
        self, all_node_mocks, mock_router, base_flow_state, make_supervisor
    ):
        nodes = all_node_mocks
        supervisor = make_supervisor()
        nodes["agent"].run.return_value = {**base_flow_state}

        compiled = _compile(supervisor, mock_router)

        with pytest.raises(RoutingError, match="Conversation history not found"):
            compiled.invoke(base_flow_state)

    @pytest.mark.asyncio
    async def test_full_single_delegation_loop_via_real_subagent_subgraph(
        self,
        mock_agent_node_cls,
        mock_tool_node_cls,
        mock_final_response_node_cls,
        mock_router,
        base_flow_state,
        supervisor_name,
        developer_name,
        tester_name,
        delegate_tool_call,
        make_supervisor,
    ):
        """Agent -> delegation_prepare -> Send(developer) -> delegation_collect -> agent -> final_response.

        Uses RoutingMockSubagentComponent for the developer subagent so a real native Send dispatch, and
        SubagentDispatchNode's translation of its result, is exercised end-to-end without a fully wired AgentComponent.
        DelegationPrepareNode/DelegationCollectNode are deliberately left un-mocked here (unlike the other tests in this
        class) so the actual dispatch/collect logic runs.
        """
        nodes = {
            "agent": mock_agent_node_cls.return_value,
            "tools": mock_tool_node_cls.return_value,
            "final_response": mock_final_response_node_cls.return_value,
        }
        routing_sub_agents = {
            developer_name: RoutingMockSubagentComponent(
                name=developer_name, answer="Implementation complete."
            ),
            tester_name: MockSubagentComponent(name=tester_name),
        }
        supervisor = make_supervisor(subagent_components=routing_sub_agents)

        nodes["agent"].run.side_effect = [
            {
                **base_flow_state,
                FlowStateKeys.CONVERSATION_HISTORY: {
                    supervisor_name: [
                        AIMessage(content="", tool_calls=[delegate_tool_call])
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
        nodes["final_response"].run.return_value = {**base_flow_state}
        mock_router.route.return_value = END

        # delegation_prepare/collect nodes are real here (not mocked) so the
        # Send-based dispatch and result-collection actually run.
        compiled = _compile(supervisor, mock_router)
        await compiled.ainvoke(base_flow_state)

        assert nodes["agent"].run.call_count == 2
        nodes["final_response"].run.assert_called_once()
        # The developer subagent's real result reached the supervisor as a
        # completed ToolMessage in its own conversation history by the time
        # the agent node is re-invoked for its second turn.
        second_agent_call_state = nodes["agent"].run.call_args_list[1].args[0]
        supervisor_history = second_agent_call_state[
            FlowStateKeys.CONVERSATION_HISTORY
        ][supervisor_name]
        tool_messages = [
            m for m in supervisor_history if m.__class__.__name__ == "ToolMessage"
        ]
        assert len(tool_messages) == 1
        assert "Implementation complete." in tool_messages[0].content
        assert "<status>completed</status>" in tool_messages[0].content


class TestSupervisorMaxCycles:
    """Tests for max_cycles/iteration_warning_offset on SupervisorAgentComponentV2.

    Mirrors the equivalent v1 SupervisorAgentComponent test suite -- the two components' own AgentNode construction must
    stay in sync since only v1's was updated when the soft cycle-limit warning was introduced.
    """

    @pytest.mark.usefixtures(
        "mock_tool_node_cls",
        "mock_final_response_node_cls",
        "mock_delegation_prepare_node_cls",
        "mock_delegation_collect_node_cls",
    )
    def test_max_cycles_passed_to_agent_node(
        self,
        mock_agent_node_cls,
        mock_router,
        make_supervisor,
        mock_state_graph,
        supervisor_name,
    ):
        supervisor = make_supervisor(max_cycles=7)
        supervisor.attach(mock_state_graph, mock_router)

        call_kwargs = mock_agent_node_cls.call_args[1]
        assert call_kwargs["cycle_budget"].max_cycles == 7
        assert call_kwargs["cycle_budget"].iteration_warning_offset == 6

    @pytest.mark.usefixtures(
        "mock_tool_node_cls",
        "mock_final_response_node_cls",
        "mock_delegation_prepare_node_cls",
        "mock_delegation_collect_node_cls",
    )
    def test_max_cycles_config_form_resolved_to_int_threshold(
        self,
        mock_agent_node_cls,
        mock_router,
        make_supervisor,
        mock_state_graph,
    ):
        """A MaxCyclesConfig max_cycles must resolve to its int threshold, not the config object itself.

        AgentNode's max_cycles/cycle-count comparisons require an int; passing the raw MaxCyclesConfig through would
        break at runtime the first time a cycle-count check ran.
        """
        supervisor = make_supervisor(
            max_cycles=MaxCyclesConfig(threshold=10, iteration_warning_offset=3)
        )
        supervisor.attach(mock_state_graph, mock_router)

        call_kwargs = mock_agent_node_cls.call_args[1]
        assert call_kwargs["cycle_budget"].max_cycles == 10
        assert isinstance(call_kwargs["cycle_budget"].max_cycles, int)
        assert call_kwargs["cycle_budget"].iteration_warning_offset == 3

    @pytest.mark.usefixtures(
        "mock_tool_node_cls",
        "mock_final_response_node_cls",
        "mock_delegation_prepare_node_cls",
        "mock_delegation_collect_node_cls",
    )
    def test_max_cycles_default_passed_to_agent_node_when_unset(
        self,
        mock_agent_node_cls,
        mock_router,
        make_supervisor,
        mock_state_graph,
    ):
        supervisor = make_supervisor()
        supervisor.attach(mock_state_graph, mock_router)

        call_kwargs = mock_agent_node_cls.call_args[1]
        assert (
            call_kwargs["cycle_budget"].max_cycles
            == AgentComponentBase._DEFAULT_MAX_CYCLES
        )


class TestAgentNodeInvokeConfig:
    """Tests for SupervisorAgentComponentV2._agent_node_invoke_config's TAG_NOSTREAM logic."""

    def test_returns_streaming_enabled_config_when_both_events_declared(
        self, make_supervisor
    ):
        supervisor = make_supervisor(
            ui_log_events=[
                UILogEventsSupervisor.ON_AGENT_FINAL_ANSWER,
                UILogEventsSupervisor.ON_AGENT_REASONING,
            ]
        )

        assert (
            supervisor._agent_node_invoke_config()
            == AgentComponentBase.STREAMING_ENABLED_CONFIG
        )

    @pytest.mark.parametrize(
        "ui_log_events",
        [
            [],
            [UILogEventsSupervisor.ON_AGENT_FINAL_ANSWER],
            [UILogEventsSupervisor.ON_AGENT_REASONING],
        ],
        ids=["no_events", "only_final_answer", "only_reasoning"],
    )
    def test_returns_streaming_disabled_config_unless_both_events_declared(
        self, make_supervisor, ui_log_events
    ):
        supervisor = make_supervisor(ui_log_events=ui_log_events)

        assert (
            supervisor._agent_node_invoke_config()
            == AgentComponentBase.STREAMING_DISABLED_CONFIG
        )


class TestValidateAndBuildDelegateTaskCls:
    """Tests for the defensive post-construction guard on subagent_components."""

    def test_raises_if_subagent_components_is_empty(self, make_supervisor):
        """Guards against a subclass/programmatic path bypassing the `before` validator.

        ``validate_and_consume_subagents`` (mode="before") already prevents an
        empty ``subagent_components`` from surviving normal construction, so
        this defensive check can only be exercised by invoking the `after`
        validator directly against a mutated instance.
        """
        supervisor = make_supervisor()
        supervisor.subagent_components = {}

        with pytest.raises(
            ValueError, match="requires at least one subagent component"
        ):
            supervisor.validate_and_build_delegate_task_cls()


class TestAgentNodeRouterSchemaMode:
    """Tests for _agent_node_router's schema-mode-specific RoutingError."""

    def test_text_only_response_in_schema_mode_raises_routing_error(
        self,
        make_supervisor,
        mock_toolset,
        supervisor_name,
        base_flow_state,
        ai_message_no_tool_calls,
    ):
        mock_toolset.__contains__ = Mock(
            side_effect=lambda name: name != "custom_response_tool"
        )
        supervisor = make_supervisor(
            response_schema_id="general/structured_response",
            response_schema_version="1.0.0",
        )
        state = {**base_flow_state}
        state["conversation_history"] = {supervisor_name: [ai_message_no_tool_calls]}

        with pytest.raises(RoutingError, match="Schema mode requires a tool call"):
            supervisor._agent_node_router(state)


class TestAttachWithResponseSchema:
    """Tests for attach()'s response-schema-specific supervisor tool wiring."""

    def test_response_schema_tool_added_to_supervisor_tools(
        self,
        all_node_mocks,
        mock_router,
        mock_state_graph,
        mock_toolset,
        mock_prompt_registry,
        mock_schema_registry,
        make_supervisor,
    ):
        mock_toolset.__contains__ = Mock(
            side_effect=lambda name: name != "custom_response_tool"
        )
        supervisor = make_supervisor(
            response_schema_id="general/structured_response",
            response_schema_version="1.0.0",
        )

        supervisor.attach(mock_state_graph, mock_router)

        prompt_call_kwargs = mock_prompt_registry.get_on_behalf.call_args.kwargs
        expected_response_schema = mock_schema_registry.get.return_value
        assert expected_response_schema in prompt_call_kwargs["tools"]


@pytest.fixture(name="mock_tool_approval_request_node_cls")
def mock_tool_approval_request_node_cls_fixture(supervisor_name):
    """Fixture for mocked ToolApprovalRequestNode class in the agent component module."""
    with patch(f"{_AGENT_COMPONENT_MODULE}.ToolApprovalRequestNode") as mock_cls:
        mock_cls.return_value.name = f"{supervisor_name}#tool_approval_request"
        yield mock_cls


@pytest.fixture(name="mock_tool_approval_fetch_node_cls")
def mock_tool_approval_fetch_node_cls_fixture(supervisor_name):
    """Fixture for mocked ToolApprovalFetchNode class in the agent component module."""
    with patch(f"{_AGENT_COMPONENT_MODULE}.ToolApprovalFetchNode") as mock_cls:
        mock_cls.return_value.name = f"{supervisor_name}#tool_approval_fetch"
        yield mock_cls


class TestSupervisorV2ToolApprovalAttribution:
    """Tests for how the v2 supervisor attributes its own tool-approval prompts."""

    def test_supervisors_own_approval_prompt_is_not_subsession_scoped(
        self,
        all_node_mocks,  # pylint: disable=unused-argument
        mock_tool_approval_fetch_node_cls,  # pylint: disable=unused-argument
        mock_tool_approval_request_node_cls,
        mock_router,
        supervisor_name,
        make_supervisor,
    ):
        """The supervisor's own prompt carries no subsession, and the event list is fixed.

        Mirrors the v1 supervisor's contract. Under v2 several subagents can be
        dispatched concurrently, so scoping the *supervisor's* own prompt to a
        subsession would mis-attribute it to whichever dispatch happened to be
        in flight. A dispatched subagent's own prompts are attributed to its
        subsession instead, by ``AgentComponent.compile_as_subagent`` pointing
        ``_session_id_key`` at ``SUBSESSION_ID_CONTEXT_KEY``. The fixed event
        list is what stops a config omitting ``on_tool_approval_request`` from
        hanging the flow; neither is visible from graph topology.
        """
        supervisor = make_supervisor(require_tool_approval=True, pre_approved_tools=[])
        _compile(supervisor, mock_router)

        call_kwargs = mock_tool_approval_request_node_cls.call_args.kwargs
        assert isinstance(call_kwargs["session_id_key"], NoneIOKey)

        ui_history = call_kwargs["ui_history"]
        assert ui_history.events == [UILogEventsAgent.ON_TOOL_APPROVAL_REQUEST]
        # pylint: disable=protected-access
        assert ui_history.log._component_name == supervisor_name
        assert ui_history.log._ui_roles_as == MessageTypeEnum.REQUEST
        # pylint: enable=protected-access


class TestSupervisorV2DelegationEventWiring:
    """Tests that attach() gives the delegation nodes what they need to emit internal events."""

    @pytest.mark.usefixtures("all_node_mocks")
    @pytest.mark.parametrize(
        "node_cls_fixture",
        ["mock_delegation_prepare_node_cls", "mock_delegation_collect_node_cls"],
    )
    def test_event_tracking_args_passed_to_delegation_nodes(
        self,
        request,
        node_cls_fixture,
        mock_router,
        make_supervisor,
        mock_state_graph,
        flow_id,
        flow_type,
        mock_internal_event_client,
        supervisor_name,
    ):
        """Without these, parallel delegations produce no analytics events."""
        node_cls = request.getfixturevalue(node_cls_fixture)
        supervisor = make_supervisor()

        supervisor.attach(mock_state_graph, mock_router)

        tracker = node_cls.call_args[1]["tracker"]
        assert isinstance(tracker, SubagentDelegationTracker)

        tracker.rejected(reason=DelegationRejectionReason.LIMIT_REACHED)
        call_kwargs = mock_internal_event_client.track_event.call_args.kwargs
        assert call_kwargs["category"] == flow_type.value
        assert call_kwargs["additional_properties"].label == supervisor_name
        assert call_kwargs["additional_properties"].value == flow_id
        assert call_kwargs["additional_properties"].extra["parallel"] is True
