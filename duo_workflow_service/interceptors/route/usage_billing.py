from typing import TYPE_CHECKING, Any, Callable, Literal, NamedTuple, Optional

import structlog

from ai_gateway.prompts import BasePromptCallbackHandler, BasePromptRegistry, Prompt
from contract import contract_pb2
from duo_workflow_service.executor.outbox import Outbox
from lib.events import GLReportingEventContext
from lib.events.contextvar import llm_call_is_retry, self_hosted_dap_billing_enabled

if TYPE_CHECKING:
    from duo_workflow_service.workflows.abstract_workflow import AbstractWorkflow

log = structlog.stdlib.get_logger("usage_billing")


class SelfHostedBillingPromptCallbackHandler(BasePromptCallbackHandler):
    def __init__(
        self,
        workflow_id: str,
        gl_feat_context: GLReportingEventContext,
        outbox: Outbox,
        source_type: Optional[Callable[[], Optional[str]]] = None,
    ):
        self.workflow_id = workflow_id
        self.gl_reporting_event_context = gl_feat_context
        self.outbox = outbox
        self.source_type = source_type

    async def on_before_llm_call(self):
        # A retry repeats a call that failed and was already billed, so it is the same logical call.
        if llm_call_is_retry.get():
            log.info(
                "Skipping self-hosted billing for a retried LLM call",
                workflow_id=self.workflow_id,
                feature_qualified_name=self.gl_reporting_event_context.feature_qualified_name,
            )
            return

        action = contract_pb2.Action(
            trackLlmCallForSelfHosted=contract_pb2.TrackLlmCallForSelfHosted(
                workflowID=self.workflow_id,
                featureQualifiedName=self.gl_reporting_event_context.feature_qualified_name,
                featureAiCatalogItem=self.gl_reporting_event_context.feature_ai_catalog_item
                or False,
                sourceType=(self.source_type and self.source_type()) or "",
            )
        )
        await self.outbox.put_action_and_wait_for_response(action)


class PromptRegistrySelfHostedBillingSupport(BasePromptRegistry):
    def __init__(
        self,
        instance: BasePromptRegistry,
        workflow_id: str,
        gl_feat_context: GLReportingEventContext,
        outbox: Outbox,
        source_type: Optional[Callable[[], Optional[str]]] = None,
    ):
        super().__init__(instance.internal_event_client, instance.model_limits)
        self.instance = instance
        self.callback = SelfHostedBillingPromptCallbackHandler(
            workflow_id, gl_feat_context, outbox, source_type
        )

    def __getattr__(self, name: str) -> Any:
        """Redirect all attribute access to the wrapped prompt_registry."""
        return getattr(self.instance, name)

    def get(self, *args: Any, **kwargs: Any) -> Prompt:
        """Capture the get method specifically to register self-hosted billing callbacks.

        Skips billing callback registration for compaction_auto operation types so that compaction LLM calls on self-
        hosted deployments do not trigger billing events.
        """
        prompt = self.instance.get(*args, **kwargs)
        if prompt.operation_type != "compaction_auto":
            prompt.internal_callbacks.append(self.callback)

        return prompt

    def get_required_variables(self, prompt_id: str, prompt_version: Any) -> Any:
        """Delegate to the wrapped registry — covered by __getattr__ at runtime."""
        return self.instance.get_required_variables(prompt_id, prompt_version)


class FieldsWorkflowLegacy(NamedTuple):
    workflow_id: str = "_workflow_id"
    workflow_type: str = "_workflow_type"
    outbox: str = "_outbox"
    prompt_registry: str = "_prompt_registry"
    workflow_config: str = "_workflow_config"


class FieldsWorkflowFlow(NamedTuple):
    workflow_id: str = "_workflow_id"
    workflow_type: str = "_workflow_type"
    outbox: str = "_outbox"
    prompt_registry: str = "_flow_prompt_registry"
    workflow_config: str = "_workflow_config"


def support_self_hosted_billing(
    *, class_schema: Literal["legacy", "flow/v1", "flow/experimental"]
):
    fields: FieldsWorkflowFlow | FieldsWorkflowLegacy

    if class_schema in ("flow/v1", "flow/experimental"):
        fields = FieldsWorkflowFlow()
    elif class_schema == "legacy":
        fields = FieldsWorkflowLegacy()
    else:
        raise ValueError(f"unsupported class_schema value '{class_schema}'")

    def decorator(cls: type["AbstractWorkflow"]):
        original_init = cls.__init__

        def new_init(self, *args, **kwargs):
            original_init(self, *args, **kwargs)

            # Don't patch the init method when self-hosted billing is disabled
            if not self_hosted_dap_billing_enabled.get():
                return

            prompt_registry = getattr(self, fields.prompt_registry)
            workflow_id = getattr(self, fields.workflow_id)
            workflow_type = getattr(self, fields.workflow_type)
            outbox = getattr(self, fields.outbox)

            # Patch the prompt registry
            setattr(
                self,
                fields.prompt_registry,
                PromptRegistrySelfHostedBillingSupport(
                    prompt_registry,
                    workflow_id,
                    workflow_type,
                    outbox,
                    # Read lazily: the workflow config is fetched in run(), after __init__.
                    lambda: getattr(self, fields.workflow_config).get("source_type"),
                ),
            )

        setattr(cls, "__init__", new_init)

        return cls

    return decorator
