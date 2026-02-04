from unittest.mock import AsyncMock, MagicMock

import pytest

from ai_gateway.prompts import BasePromptRegistry, Prompt
from contract import contract_pb2
from duo_workflow_service.executor.outbox import Outbox
from duo_workflow_service.interceptors.route.usage_billing import (
    PromptRegistrySelfHostedBillingSupport,
    SelfHostedBillingPromptCallbackHandler,
    support_self_hosted_billing,
)
from duo_workflow_service.workflows.abstract_workflow import AbstractWorkflow
from lib.events import GLReportingEventContext
from lib.events.contextvar import self_hosted_dap_billing_enabled


@pytest.fixture(name="workflow_id")
def workflow_id_fixture():
    return "test-workflow-123"


@pytest.fixture(name="gl_reporting_event_context")
def gl_reporting_event_context_fixture():
    return GLReportingEventContext.from_workflow_definition("test_workflow")


@pytest.fixture(name="mock_outbox")
def mock_outbox_fixture():
    outbox = MagicMock(spec=Outbox)
    outbox.put_action_and_wait_for_response = AsyncMock()
    return outbox


@pytest.fixture(name="mock_prompt_registry")
def mock_prompt_registry_fixture():
    registry = MagicMock(spec=BasePromptRegistry)
    mock_prompt = MagicMock(spec=Prompt)
    mock_prompt.internal_callbacks = []
    registry.get.return_value = mock_prompt
    return registry


@pytest.fixture(name="mock_prompt")
def mock_prompt_fixture():
    prompt = MagicMock(spec=Prompt)
    prompt.internal_callbacks = []
    return prompt


class TestSelfHostedBillingPromptCallbackHandler:
    """Test SelfHostedBillingPromptCallbackHandler class."""

    @pytest.mark.asyncio
    async def test_on_before_llm_call_sends_track_action(
        self, workflow_id, gl_reporting_event_context, mock_outbox
    ):
        """Test that on_before_llm_call sends the correct tracking action."""
        handler = SelfHostedBillingPromptCallbackHandler(
            workflow_id, gl_reporting_event_context, mock_outbox
        )

        await handler.on_before_llm_call()

        # Verify the action was sent
        mock_outbox.put_action_and_wait_for_response.assert_called_once()

        # Verify the action content
        call_args = mock_outbox.put_action_and_wait_for_response.call_args
        action = call_args[0][0]

        assert isinstance(action, contract_pb2.Action)
        assert action.HasField("trackLlmCallForSelfHosted")
        assert action.trackLlmCallForSelfHosted.workflowID == workflow_id
        assert (
            action.trackLlmCallForSelfHosted.featureQualifiedName
            == gl_reporting_event_context.feature_qualified_name
        )
        expected_catalog_item = (
            gl_reporting_event_context.feature_ai_catalog_item or False
        )
        assert (
            action.trackLlmCallForSelfHosted.featureAiCatalogItem
            == expected_catalog_item
        )

    @pytest.mark.asyncio
    async def test_on_before_llm_call_with_ai_catalog_item(
        self, workflow_id, mock_outbox
    ):
        """Test on_before_llm_call with AI catalog item enabled."""
        gl_context = GLReportingEventContext.from_workflow_definition(
            "test_workflow", is_ai_catalog_item=True
        )
        handler = SelfHostedBillingPromptCallbackHandler(
            workflow_id, gl_context, mock_outbox
        )

        await handler.on_before_llm_call()

        call_args = mock_outbox.put_action_and_wait_for_response.call_args
        action = call_args[0][0]

        assert action.trackLlmCallForSelfHosted.featureAiCatalogItem is True


class TestPromptRegistrySelfHostedBillingSupport:
    """Test PromptRegistrySelfHostedBillingSupport wrapper class."""

    def test_getattr_redirects_to_instance(
        self,
        mock_prompt_registry,
        workflow_id,
        gl_reporting_event_context,
        mock_outbox,
    ):
        """Test that attribute access is redirected to the wrapped instance."""
        mock_prompt_registry.some_attribute = "test_value"

        wrapper = PromptRegistrySelfHostedBillingSupport(
            mock_prompt_registry, workflow_id, gl_reporting_event_context, mock_outbox
        )

        assert wrapper.some_attribute == "test_value"

    def test_get_registers_callback(
        self,
        mock_prompt_registry,
        workflow_id,
        gl_reporting_event_context,
        mock_outbox,
        mock_prompt,
    ):
        """Test that get() registers the billing callback on the prompt."""
        mock_prompt_registry.get.return_value = mock_prompt

        wrapper = PromptRegistrySelfHostedBillingSupport(
            mock_prompt_registry, workflow_id, gl_reporting_event_context, mock_outbox
        )

        result = wrapper.get("test_prompt", "1.0.0")

        # Verify the prompt registry get was called
        mock_prompt_registry.get.assert_called_once_with("test_prompt", "1.0.0")

        # Verify the callback was registered
        assert len(mock_prompt.internal_callbacks) == 1
        assert isinstance(
            mock_prompt.internal_callbacks[0], SelfHostedBillingPromptCallbackHandler
        )

        assert result == mock_prompt

    def test_get_with_kwargs(
        self,
        mock_prompt_registry,
        workflow_id,
        gl_reporting_event_context,
        mock_outbox,
        mock_prompt,
    ):
        """Test that get() passes through additional kwargs."""
        mock_prompt_registry.get.return_value = mock_prompt

        wrapper = PromptRegistrySelfHostedBillingSupport(
            mock_prompt_registry, workflow_id, gl_reporting_event_context, mock_outbox
        )

        result = wrapper.get(
            "test_prompt", "1.0.0", model_metadata={"test": "data"}, tools=[]
        )

        mock_prompt_registry.get.assert_called_once_with(
            "test_prompt", "1.0.0", model_metadata={"test": "data"}, tools=[]
        )
        assert result == mock_prompt


class TestSupportSelfHostedBillingDecorator:
    """Test support_self_hosted_billing decorator."""

    def test_decorator_with_billing_enabled_patches_init_legacy(
        self, mock_prompt_registry, workflow_id, gl_reporting_event_context, mock_outbox
    ):
        """Test that decorator patches __init__ for legacy schema when billing is enabled."""
        self_hosted_dap_billing_enabled.set(True)

        @support_self_hosted_billing(class_schema="legacy")
        class TestWorkflow(AbstractWorkflow):
            def __init__(self):
                self._workflow_id = workflow_id
                self._workflow_type = gl_reporting_event_context
                self._outbox = mock_outbox
                self._prompt_registry = mock_prompt_registry

            def _compile(self):
                pass

            def _handle_workflow_failure(self, error):
                pass

            def get_workflow_state(self):
                return {}

        workflow = TestWorkflow()

        # Verify the prompt registry was wrapped
        assert isinstance(
            workflow._prompt_registry, PromptRegistrySelfHostedBillingSupport
        )
        assert workflow._prompt_registry.instance == mock_prompt_registry

    def test_decorator_with_billing_enabled_patches_init_flow_v1(
        self, mock_prompt_registry, workflow_id, gl_reporting_event_context, mock_outbox
    ):
        """Test that decorator patches __init__ for flow/v1 schema when billing is enabled."""
        self_hosted_dap_billing_enabled.set(True)

        @support_self_hosted_billing(class_schema="flow/v1")
        class TestWorkflow(AbstractWorkflow):
            def __init__(self):
                self._workflow_id = workflow_id
                self._workflow_type = gl_reporting_event_context
                self._outbox = mock_outbox
                self._flow_prompt_registry = mock_prompt_registry

            def _compile(self):
                pass

            def _handle_workflow_failure(self, error):
                pass

            def get_workflow_state(self):
                return {}

        workflow = TestWorkflow()

        # Verify the flow prompt registry was wrapped
        assert isinstance(
            workflow._flow_prompt_registry, PromptRegistrySelfHostedBillingSupport
        )
        assert workflow._flow_prompt_registry.instance == mock_prompt_registry

    def test_decorator_with_billing_enabled_patches_init_flow_experimental(
        self, mock_prompt_registry, workflow_id, gl_reporting_event_context, mock_outbox
    ):
        """Test that decorator patches __init__ for flow/experimental schema when billing is enabled."""
        self_hosted_dap_billing_enabled.set(True)

        @support_self_hosted_billing(class_schema="flow/experimental")
        class TestWorkflow(AbstractWorkflow):
            def __init__(self):
                self._workflow_id = workflow_id
                self._workflow_type = gl_reporting_event_context
                self._outbox = mock_outbox
                self._flow_prompt_registry = mock_prompt_registry

            def _compile(self):
                pass

            def _handle_workflow_failure(self, error):
                pass

            def get_workflow_state(self):
                return {}

        workflow = TestWorkflow()

        # Verify the flow prompt registry was wrapped
        assert isinstance(
            workflow._flow_prompt_registry, PromptRegistrySelfHostedBillingSupport
        )
        assert workflow._flow_prompt_registry.instance == mock_prompt_registry

    def test_decorator_preserves_original_init_behavior(
        self, mock_prompt_registry, workflow_id, gl_reporting_event_context, mock_outbox
    ):
        """Test that decorator preserves original __init__ behavior."""
        self_hosted_dap_billing_enabled.set(True)

        @support_self_hosted_billing(class_schema="legacy")
        class TestWorkflow(AbstractWorkflow):
            def __init__(self, custom_arg: str):
                self.custom_arg = custom_arg
                self._workflow_id = workflow_id
                self._workflow_type = gl_reporting_event_context
                self._outbox = mock_outbox
                self._prompt_registry = mock_prompt_registry

            def _compile(self):
                pass

            def _handle_workflow_failure(self, error):
                pass

            def get_workflow_state(self):
                return {}

        workflow = TestWorkflow("test_value")

        # Verify custom initialization still works
        assert workflow.custom_arg == "test_value"
        # Verify billing support was added
        assert isinstance(
            workflow._prompt_registry, PromptRegistrySelfHostedBillingSupport
        )


class TestIntegrationScenarios:
    """Integration tests for complete billing flow."""

    @pytest.mark.asyncio
    async def test_end_to_end_billing_flow(
        self, mock_prompt_registry, workflow_id, gl_reporting_event_context, mock_outbox
    ):
        """Test complete flow from workflow creation to LLM call tracking."""
        self_hosted_dap_billing_enabled.set(True)

        # Create a mock prompt
        mock_prompt = MagicMock(spec=Prompt)
        mock_prompt.internal_callbacks = []
        mock_prompt_registry.get.return_value = mock_prompt

        @support_self_hosted_billing(class_schema="legacy")
        class TestWorkflow(AbstractWorkflow):
            def __init__(self):
                self._workflow_id = workflow_id
                self._workflow_type = gl_reporting_event_context
                self._outbox = mock_outbox
                self._prompt_registry = mock_prompt_registry

            def _compile(self):
                pass

            def _handle_workflow_failure(self, error):
                pass

            def get_workflow_state(self):
                return {}

        # Create workflow instance
        workflow = TestWorkflow()

        # Get a prompt (simulating workflow execution)
        workflow._prompt_registry.get("test_prompt", "1.0.0")

        # Verify callback was registered
        assert len(mock_prompt.internal_callbacks) == 1
        callback = mock_prompt.internal_callbacks[0]

        # Simulate LLM call
        await callback.on_before_llm_call()

        # Verify tracking action was sent
        mock_outbox.put_action_and_wait_for_response.assert_called_once()
        action = mock_outbox.put_action_and_wait_for_response.call_args[0][0]
        assert action.trackLlmCallForSelfHosted.workflowID == workflow_id

    @pytest.mark.asyncio
    async def test_billing_disabled_no_tracking(
        self, mock_prompt_registry, workflow_id, gl_reporting_event_context, mock_outbox
    ):
        """Test that no tracking occurs when billing is disabled."""
        self_hosted_dap_billing_enabled.set(False)

        mock_prompt = MagicMock(spec=Prompt)
        mock_prompt.internal_callbacks = []
        mock_prompt_registry.get.return_value = mock_prompt

        @support_self_hosted_billing(class_schema="legacy")
        class TestWorkflow(AbstractWorkflow):
            def __init__(self):
                self._workflow_id = workflow_id
                self._workflow_type = gl_reporting_event_context
                self._outbox = mock_outbox
                self._prompt_registry = mock_prompt_registry

            def _compile(self):
                pass

            def _handle_workflow_failure(self, error):
                pass

            def get_workflow_state(self):
                return {}

        workflow = TestWorkflow()

        # Prompt registry should not be wrapped
        assert workflow._prompt_registry == mock_prompt_registry
        assert not isinstance(
            workflow._prompt_registry, PromptRegistrySelfHostedBillingSupport
        )

        # Get a prompt
        workflow._prompt_registry.get("test_prompt", "1.0.0")

        # No callbacks should be registered
        assert len(mock_prompt.internal_callbacks) == 0
        mock_outbox.put_action_and_wait_for_response.assert_not_called()

    def test_billing_disabled_early_return(
        self, mock_prompt_registry, workflow_id, gl_reporting_event_context, mock_outbox
    ):
        """Test that decorator returns early when billing is disabled for legacy schema."""
        self_hosted_dap_billing_enabled.set(False)

        @support_self_hosted_billing(class_schema="legacy")
        class TestWorkflow(AbstractWorkflow):
            def __init__(self):
                self._workflow_id = workflow_id
                self._workflow_type = gl_reporting_event_context
                self._outbox = mock_outbox
                self._prompt_registry = mock_prompt_registry

            def _compile(self):
                pass

            def _handle_workflow_failure(self, error):
                pass

            def get_workflow_state(self):
                return {}

        workflow = TestWorkflow()

        # Verify that the prompt registry was NOT wrapped (early return occurred)
        assert workflow._prompt_registry is mock_prompt_registry
        assert (
            type(workflow._prompt_registry)
            is not PromptRegistrySelfHostedBillingSupport
        )
