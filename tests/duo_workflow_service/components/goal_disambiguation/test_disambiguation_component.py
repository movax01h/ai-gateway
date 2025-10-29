import os
from unittest.mock import MagicMock, patch

import pytest
from gitlab_cloud_connector import CloudConnectorUser
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from duo_workflow_service.agents.agent import Agent
from duo_workflow_service.components import ToolsRegistry
from duo_workflow_service.components.goal_disambiguation import (
    GoalDisambiguationComponent,
)
from duo_workflow_service.entities import (
    MessageTypeEnum,
    Plan,
    WorkflowEventType,
    WorkflowState,
    WorkflowStatusEnum,
)
from duo_workflow_service.gitlab.http_client import GitlabHttpClient
from duo_workflow_service.llm_factory import AnthropicConfig
from duo_workflow_service.tools.request_user_clarification import (
    RequestUserClarificationTool,
)
from lib.internal_events.event_enum import CategoryEnum


@pytest.fixture(name="llm_judge_response_unclear")
def llm_judge_response_unclear_fixture() -> AIMessage:
    return AIMessage(
        content="Please list which bugs you want to be fixed",
        tool_calls=[
            {
                "id": "1",
                "name": "request_user_clarification_tool",
                "args": {
                    "recommendations": ["List issue links for bugs to fix"],
                    "message": "I need to understand which bug do you need to fix",
                    "clarity_score": 2.1,
                    "clarity_verdict": "UNCLEAR",
                },
            }
        ],
    )


@pytest.fixture(name="agent_responses")
def agent_responses_fixture(llm_judge_response_unclear: AIMessage):
    return [
        {
            "conversation_history": {"clarity_judge": [llm_judge_response_unclear]},
        },
        {
            "conversation_history": {
                "clarity_judge": [AIMessage(content="All clear please proceed")]
            },
        },
    ]


@pytest.fixture(name="mock_interrupt")
def mock_interrupt_fixture():
    with patch(
        "duo_workflow_service.components.goal_disambiguation.component.interrupt"
    ) as mock:
        mock.return_value = {
            "event_type": WorkflowEventType.MESSAGE,
            "message": "Bugs are described in issues 1, 2, 3, and 4",
        }
        yield mock


@pytest.mark.usefixtures("mock_duo_workflow_service_container")
class TestGoalDisambiguationComponent:
    @pytest.fixture(name="graph_config")
    def graph_config_fixture(self) -> RunnableConfig:
        return RunnableConfig(
            recursion_limit=50,
            configurable={"thread_id": "test-workflow"},
        )

    @pytest.fixture(name="tools_registry_mock")
    def tools_registry_mock_fixture(self):
        mock = MagicMock(ToolsRegistry)
        mock.get_batch.return_value = [RequestUserClarificationTool]
        return mock

    @pytest.fixture(name="goal")
    def goal_fixture(str) -> str:
        return "Fix all the bugs"

    @pytest.fixture(name="graph")
    def graph_fixture(self) -> StateGraph:
        return StateGraph(WorkflowState)

    @pytest.fixture(name="component_env")
    def component_env_fixture(self) -> dict[str, str]:
        return {"FEATURE_GOAL_DISAMBIGUATION": "True"}

    @pytest.fixture(name="allow_agent_to_request_user")
    def allow_agent_to_request_user_fixture(self) -> bool:
        return True

    @pytest.fixture(name="entry_point")
    def entry_point_fixture(
        self,
        component_env: dict[str, str],
        user: CloudConnectorUser,
        goal: str,
        allow_agent_to_request_user: bool,
        graph_config: RunnableConfig,
        tools_registry_mock: ToolsRegistry,
        gl_http_client: GitlabHttpClient,
        graph: StateGraph,
    ) -> str:
        with patch.dict(os.environ, component_env):
            component = GoalDisambiguationComponent(
                user=user,
                goal=goal,
                workflow_id=graph_config["configurable"]["thread_id"],
                allow_agent_to_request_user=allow_agent_to_request_user,
                model_config=AnthropicConfig(model_name="claude-sonnet-4-20250514"),
                tools_registry=tools_registry_mock,
                http_client=gl_http_client,
                workflow_type=CategoryEnum.WORKFLOW_SOFTWARE_DEVELOPMENT,
            )

        return component.attach(
            graph=graph,
            component_exit_node=END,
            component_execution_state=WorkflowStatusEnum.PLANNING,
            graph_termination_node=END,
        )

    @pytest.fixture(name="compiled_graph")
    def compiled_graph_fixture(self, graph: StateGraph, entry_point: str):
        graph.set_entry_point(entry_point)
        return graph.compile()

    @pytest.fixture(name="graph_input")
    def graph_input_fixture(self, goal: str) -> WorkflowState:
        return WorkflowState(
            plan=Plan(steps=[]),
            status=WorkflowStatusEnum.NOT_STARTED,
            conversation_history={},
            last_human_input=None,
            handover=[],
            ui_chat_log=[],
            project=None,
            goal=goal,
            additional_context=None,
        )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "component_env",
        [
            {"FEATURE_GOAL_DISAMBIGUATION": "False"},
            {"USE_MEMSAVER": "True"},
        ],
    )
    async def test_attach_with_feature_disabled(self, entry_point: str):
        assert entry_point == END, "Then disambiguation component should be skipped"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("allow_agent_to_request_user", [False])
    async def test_attach_without_allow_agent_to_request_user(self, entry_point: str):
        assert entry_point == END, "Then disambiguation component should be skipped"

    @pytest.mark.asyncio
    async def test_component_run_with_clear_goal(
        self,
        graph_input: WorkflowState,
        graph_config: RunnableConfig,
        mock_agent: MagicMock,
        compiled_graph: CompiledStateGraph,
    ):
        mock_agent.run.side_effect = [
            {
                "conversation_history": {
                    "clarity_judge": [
                        AIMessage(
                            content="All clear please proceed",
                            tool_calls=[
                                {
                                    "id": "1",
                                    "name": "handover_tool",
                                    "args": {"summary": "This is a summary"},
                                }
                            ],
                        )
                    ]
                },
            }
        ]

        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        assert len(response["handover"]) == 1
        assert response["handover"][-1] == AIMessage(content="This is a summary")

    @pytest.mark.asyncio
    async def test_component_run_with_unclear_goal(
        self,
        mock_interrupt,
        graph_input: WorkflowState,
        graph_config: RunnableConfig,
        llm_judge_response_unclear: AIMessage,
        mock_agent: MagicMock,
        compiled_graph: CompiledStateGraph,
    ):
        mock_agent.run.side_effect = [
            {
                "conversation_history": {"clarity_judge": [llm_judge_response_unclear]},
            },
            {
                "conversation_history": {
                    "clarity_judge": [
                        AIMessage(
                            content="All clear please proceed",
                            tool_calls=[
                                {
                                    "id": "1",
                                    "name": "handover_tool",
                                    "args": {"summary": "This is a summary"},
                                }
                            ],
                        )
                    ]
                },
            },
        ]

        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        # assert that the component requested user input
        mock_interrupt.assert_called_once()
        # assert correct ui communication between the component and UI client
        assert len(response["ui_chat_log"]) == 3
        assert (
            response["ui_chat_log"][0]["content"]
            == "I need to understand which bug do you need to fix\n\nI'm ready to help with your project but I need a few key details:\n\n1. List issue links for bugs to fix"
        )
        assert response["ui_chat_log"][0]["message_type"] == MessageTypeEnum.REQUEST
        assert (
            response["ui_chat_log"][1]["content"]
            == "Bugs are described in issues 1, 2, 3, and 4"
        )
        assert response["ui_chat_log"][1]["message_type"] == MessageTypeEnum.USER
        assert response["ui_chat_log"][2]["content"] == "This is a summary"
        assert response["ui_chat_log"][2]["message_type"] == MessageTypeEnum.AGENT

        # assert clarity reevaluation cycle
        assert mock_agent.run.call_count == 2
        # only 1 handover message which is a summary
        assert len(response["handover"]) == 1
        assert response["handover"][-1].content == "This is a summary"

    @pytest.mark.asyncio
    async def test_component_run_with_string_recommendations(
        self,
        mock_interrupt,
        graph_input: WorkflowState,
        graph_config: RunnableConfig,
        llm_judge_response_unclear: AIMessage,
        mock_agent: MagicMock,
        compiled_graph: CompiledStateGraph,
    ):
        llm_judge_response_unclear.tool_calls[0]["args"][
            "recommendations"
        ] = "List issue links for bugs to fix"

        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        # assert that the component requested user input
        mock_interrupt.assert_called_once()
        # assert correct ui communication between the component and UI client

        assert len(response["ui_chat_log"]) == 2
        assert (
            response["ui_chat_log"][0]["content"]
            == "I need to understand which bug do you need to fix\n\nI'm ready to help with your project but I need a few key details:\n\n1. List issue links for bugs to fix"
        )

    @pytest.mark.asyncio
    async def test_component_run_with_resume(
        self,
        mock_interrupt,
        graph_input: WorkflowState,
        graph_config: RunnableConfig,
        mock_agent: MagicMock,
        compiled_graph: CompiledStateGraph,
    ):
        mock_interrupt.side_effect = [
            {
                "event_type": WorkflowEventType.RESUME,
                "message": "",
            },
            {
                "event_type": WorkflowEventType.MESSAGE,
                "message": "Bugs are described in issues 1, 2, 3, and 4",
            },
        ]

        await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        # Assert that interrupt was called twice - first for cancellation, then for message event
        assert mock_interrupt.call_count == 2

    @pytest.mark.asyncio
    async def test_component_run_with_stop_event(
        self,
        mock_interrupt,
        graph_input: WorkflowState,
        graph_config: RunnableConfig,
        mock_agent: MagicMock,
        compiled_graph: CompiledStateGraph,
    ):
        mock_interrupt.side_effect = [
            {
                "event_type": WorkflowEventType.STOP,
                "message": "",
            },
        ]

        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        assert mock_agent.run.call_count == 1
        mock_interrupt.assert_called_once()
        assert response["handover"] == graph_input["handover"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "agent_responses", [[{"status": WorkflowStatusEnum.CANCELLED}]]
    )
    async def test_component_run_with_agent_stop_response(
        self,
        mock_interrupt,
        graph_input: WorkflowState,
        graph_config: RunnableConfig,
        mock_agent: MagicMock,
        compiled_graph: CompiledStateGraph,
    ):
        response = await compiled_graph.ainvoke(input=graph_input, config=graph_config)

        assert mock_agent.run.call_count == 1
        assert response["status"] == WorkflowStatusEnum.CANCELLED
        assert response["handover"] == graph_input["handover"]
        mock_interrupt.assert_not_called()
