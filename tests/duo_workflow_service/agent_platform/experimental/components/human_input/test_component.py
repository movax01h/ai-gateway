from unittest.mock import Mock, call, patch

import pytest
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph

from ai_gateway.model_metadata import ModelMetadata, current_model_metadata_context
from ai_gateway.prompts import LocalPromptRegistry
from duo_workflow_service.agent_platform.experimental.components.human_input.component import (
    HumanInputComponent,
)
from duo_workflow_service.agent_platform.experimental.components.human_input.ui_log import (
    AgentLogWriter,
    UILogEventsHumanInput,
    UserLogWriter,
)
from duo_workflow_service.agent_platform.experimental.state import FlowState, IOKey
from duo_workflow_service.agent_platform.experimental.ui_log.base import UIHistory
from lib.internal_events.event_enum import CategoryEnum


class TestHumanInputComponent:
    """Test suite for HumanInputComponent."""

    @pytest.fixture
    def mock_prompt_registry(self):
        """Mock prompt registry."""
        registry = Mock(spec=LocalPromptRegistry)
        prompt = Mock(spec=PromptTemplate)
        prompt.format.return_value = "Test prompt content"
        registry.get.return_value = prompt
        return registry

    @pytest.fixture
    def human_input_component(self, mock_prompt_registry):
        """Create a HumanInputComponent instance for testing."""
        return HumanInputComponent(
            name="test_human_input",
            sends_response_to="awesome_agent",
            flow_id="test_flow",
            flow_type=CategoryEnum.WORKFLOW_CHAT,
            prompt_id="test_prompt",
            prompt_version="v1.0",
            prompt_registry=mock_prompt_registry,
            ui_log_events=[
                UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
                UILogEventsHumanInput.ON_USER_RESPONSE,
            ],
        )

    def test_iokey_template_replacement(self, human_input_component):
        """Test that IOKeyTemplate correctly replaces SENDS_RESPONSE_TO_COMPONENT_NAME_TEMPLATE."""
        outputs = human_input_component.outputs

        assert len(outputs) == 2

        # First output should be conversation_history
        conversation_output = outputs[0]
        assert isinstance(conversation_output, IOKey)
        assert conversation_output.target == "conversation_history"
        assert conversation_output.subkeys == ["awesome_agent"]

        # Second output should be approval context
        approval_output = outputs[1]
        assert isinstance(approval_output, IOKey)
        assert approval_output.target == "context"
        assert approval_output.subkeys == ["test_human_input", "approval"]

    def test_attach_method_creates_nodes(self, human_input_component):
        """Test that attach method creates proper nodes in the graph."""

        graph = StateGraph(FlowState)
        router = Mock()
        router.route = Mock(return_value="next_node")

        with (
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.RequestNode"
            ) as mock_request_node,
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.FetchNode"
            ) as mock_fetch_node,
        ):

            # Mock node instances
            request_instance = Mock()
            request_instance.name = "test_human_input#request"
            request_instance.run = Mock()
            mock_request_node.return_value = request_instance

            fetch_instance = Mock()
            fetch_instance.name = "test_human_input#fetch"
            fetch_instance.run = Mock()
            mock_fetch_node.return_value = fetch_instance

            # Mock graph methods to verify calls
            graph.add_node = Mock()
            graph.add_edge = Mock()
            graph.add_conditional_edges = Mock()

            human_input_component.attach(graph, router)

            # Verify nodes were created with correct arguments
            mock_request_node.assert_called_once_with(
                name="test_human_input#request",
                component_name="test_human_input",
                prompt=mock_request_node.call_args[1]["prompt"],
                inputs=human_input_component.inputs,
                ui_history=mock_request_node.call_args[1]["ui_history"],
            )

            mock_fetch_node.assert_called_once_with(
                name="test_human_input#fetch",
                component_name="test_human_input",
                sends_response_to="awesome_agent",
                output=human_input_component._approval_output,
                ui_history=mock_fetch_node.call_args[1]["ui_history"],
            )

            # Verify graph received calls to add_node, add_edge and add_conditional_edges with correct arguments
            graph.add_node.assert_any_call(
                "test_human_input#request", request_instance.run
            )
            graph.add_node.assert_any_call("test_human_input#fetch", fetch_instance.run)
            graph.add_edge.assert_called_once_with(
                "test_human_input#request", "test_human_input#fetch"
            )
            graph.add_conditional_edges.assert_called_once_with(
                "test_human_input#fetch", router.route
            )

    def test_prompt_registry_integration(
        self, human_input_component, mock_prompt_registry
    ):
        """Test integration with prompt registry like AgentComponent."""
        graph = StateGraph(FlowState)
        router = Mock()

        with (
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.RequestNode"
            ) as mock_request_node,
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.FetchNode"
            ) as mock_fetch_node,
        ):

            # Mock node instances
            request_instance = Mock()
            request_instance.name = "test_human_input#request"
            mock_request_node.return_value = request_instance

            fetch_instance = Mock()
            fetch_instance.name = "test_human_input#fetch"
            mock_fetch_node.return_value = fetch_instance

            human_input_component.attach(graph, router)

            # Verify prompt registry was called
            mock_prompt_registry.get.assert_called_once_with(
                "test_prompt", "v1.0", model_metadata=None
            )

            # Verify request node was created with prompt
            call_args = mock_request_node.call_args
            assert call_args[1]["prompt"] is not None

    def test_optional_prompt(self, mock_prompt_registry):
        """Test that prompt is optional when no prompt_id or prompt_version provided."""
        component = HumanInputComponent(
            name="test_human_input",
            sends_response_to="awesome_agent",
            flow_id="test_flow",
            flow_type=CategoryEnum.WORKFLOW_CHAT,
            prompt_registry=mock_prompt_registry,
        )

        graph = StateGraph(FlowState)
        router = Mock()

        with (
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.RequestNode"
            ) as mock_request_node,
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.FetchNode"
            ) as mock_fetch_node,
        ):

            # Mock node instances
            request_instance = Mock()
            request_instance.name = "test_human_input#request"
            mock_request_node.return_value = request_instance

            fetch_instance = Mock()
            fetch_instance.name = "test_human_input#fetch"
            mock_fetch_node.return_value = fetch_instance

            component.attach(graph, router)

            # Verify prompt registry was not called
            mock_prompt_registry.get.assert_not_called()

            # Verify request node was created with None prompt
            call_args = mock_request_node.call_args
            assert call_args[1]["prompt"] is None

    def test_ui_log_events_integration(self, human_input_component):
        """Test UI log events are properly passed to nodes."""
        graph = StateGraph(FlowState)
        router = Mock()

        with (
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.RequestNode"
            ) as mock_request_node,
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.FetchNode"
            ) as mock_fetch_node,
            patch(
                "duo_workflow_service.agent_platform.experimental.components.human_input.component.UIHistory"
            ) as mock_ui_history,
        ):
            mock_ui_history.return_value = Mock(spec=UIHistory)

            # Mock node instances
            request_instance = Mock()
            request_instance.name = "test_human_input#request"
            mock_request_node.return_value = request_instance

            fetch_instance = Mock()
            fetch_instance.name = "test_human_input#fetch"
            mock_fetch_node.return_value = fetch_instance

            human_input_component.attach(graph, router)

            mock_ui_history.assert_has_calls(
                [
                    call(
                        events=[
                            UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
                            UILogEventsHumanInput.ON_USER_RESPONSE,
                        ],
                        writer_class=AgentLogWriter,
                    ),
                    call(
                        events=[
                            UILogEventsHumanInput.ON_USER_INPUT_PROMPT,
                            UILogEventsHumanInput.ON_USER_RESPONSE,
                        ],
                        writer_class=UserLogWriter,
                    ),
                ]
            )

            request_node_ui_history = mock_request_node.call_args[1]["ui_history"]
            assert request_node_ui_history == mock_ui_history.return_value

            fetch_node_ui_history = mock_fetch_node.call_args[1]["ui_history"]
            assert fetch_node_ui_history == mock_ui_history.return_value


class TestHumanInputComponentModelMetadata:
    """Test suite for HumanInputComponent model metadata handling."""

    @pytest.fixture
    def mock_prompt_registry(self):
        """Mock prompt registry."""
        registry = Mock(spec=LocalPromptRegistry)
        prompt = Mock(spec=PromptTemplate)
        prompt.format.return_value = "Test prompt content"
        registry.get.return_value = prompt
        return registry

    @pytest.fixture
    def human_input_component(self, mock_prompt_registry):
        """Create a HumanInputComponent instance for testing."""
        return HumanInputComponent(
            name="test_human_input",
            sends_response_to="awesome_agent",
            flow_id="test_flow",
            flow_type=CategoryEnum.WORKFLOW_CHAT,
            prompt_id="test_prompt",
            prompt_version="v1.0",
            prompt_registry=mock_prompt_registry,
        )

    def test_attach_passes_model_metadata_from_context_to_prompt_registry(
        self,
        human_input_component,
        mock_prompt_registry,
    ):
        mock_model_metadata = ModelMetadata(
            name="gpt_5",
            provider="gitlab",
            friendly_name="OpenAI GPT-5",
        )

        metadata_token = current_model_metadata_context.set(mock_model_metadata)

        try:
            graph = StateGraph(FlowState)
            router = Mock()

            with (
                patch(
                    "duo_workflow_service.agent_platform.experimental.components.human_input.component.RequestNode"
                ) as mock_request_node,
                patch(
                    "duo_workflow_service.agent_platform.experimental.components.human_input.component.FetchNode"
                ) as mock_fetch_node,
            ):
                request_instance = Mock()
                request_instance.name = "test_human_input#request_metadata_test"
                request_instance.run = Mock()
                mock_request_node.return_value = request_instance

                fetch_instance = Mock()
                fetch_instance.name = "test_human_input#fetch_metadata_test"
                fetch_instance.run = Mock()
                mock_fetch_node.return_value = fetch_instance

                human_input_component.attach(graph, router)

            mock_prompt_registry.get.assert_called_once()
            call_kwargs = mock_prompt_registry.get.call_args[1]

            assert "model_metadata" in call_kwargs
            assert call_kwargs["model_metadata"] == mock_model_metadata
        finally:
            current_model_metadata_context.reset(metadata_token)

    def test_attach_passes_none_when_no_model_metadata_in_context(
        self,
        human_input_component,
        mock_prompt_registry,
    ):
        metadata_token = current_model_metadata_context.set(None)

        try:
            graph = StateGraph(FlowState)
            router = Mock()

            with (
                patch(
                    "duo_workflow_service.agent_platform.experimental.components.human_input.component.RequestNode"
                ) as mock_request_node,
                patch(
                    "duo_workflow_service.agent_platform.experimental.components.human_input.component.FetchNode"
                ) as mock_fetch_node,
            ):
                request_instance = Mock()
                request_instance.name = "test_human_input#request_none_metadata_test"
                request_instance.run = Mock()
                mock_request_node.return_value = request_instance

                fetch_instance = Mock()
                fetch_instance.name = "test_human_input#fetch_none_metadata_test"
                fetch_instance.run = Mock()
                mock_fetch_node.return_value = fetch_instance

                human_input_component.attach(graph, router)

            mock_prompt_registry.get.assert_called_once()
            call_kwargs = mock_prompt_registry.get.call_args[1]

            assert "model_metadata" in call_kwargs
            assert call_kwargs["model_metadata"] is None
        finally:
            current_model_metadata_context.reset(metadata_token)
