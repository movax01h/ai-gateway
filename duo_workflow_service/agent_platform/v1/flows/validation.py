"""DryRunFlowValidator: dry-run compilation of a Flow config for validation purposes.

Key design decisions
--------------------
* **Not executable**: ``_compile_and_run_graph`` is overridden to raise
    ``RuntimeError`` unconditionally, making it impossible to accidentally use
    ``DryRunFlowValidator`` as a real workflow.  ``validate()`` calls ``_compile()``
    directly, bypassing that guard intentionally.

* **Tools**: The real ``ToolsRegistry`` is used (all agent privileges granted, stub
    ``ToolMetadata``).  ``DeterministicStepComponent`` can therefore verify that every
    tool name declared in a config actually exists in the registry.  Tool instances
    store the metadata dict but only call into it during async execution, so passing
    ``None`` for client/project is safe at compile time.

* **Prompt registry**: ``_flow_prompt_registry`` is replaced with a
    ``_StubPromptRegistry`` after ``super().__init__()`` runs.
    ``AgentComponent.attach()`` calls ``get_on_behalf()`` synchronously during
    ``_compile()`` and passes the result to ``AgentNode(prompt=...)``, where it is
    stored but never called.  The stub returns ``None`` — safe because
    ``_compile_and_run_graph`` blocks all execution paths that would invoke the
    stored prompt.  ``get_required_variables()`` is delegated to the real registry
    so that component ``model_validator`` hooks perform genuine Jinja2 variable-set
    checks.

* **Checkpointer**: ``MemorySaver`` — no external state store needed.
"""

from __future__ import annotations

from typing import Any, List, Optional

from gitlab_cloud_connector import CloudConnectorUser
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import MemorySaver

from ai_gateway.config import ConfigModelLimits
from ai_gateway.model_metadata import TypeModelMetadata
from ai_gateway.prompts import BasePromptRegistry
from ai_gateway.prompts.base import Prompt
from ai_gateway.response_schemas.registry import ResponseSchemaRegistry
from duo_workflow_service.agent_platform.v1.flows.base import Flow
from duo_workflow_service.agent_platform.v1.flows.flow_config import FlowConfig
from duo_workflow_service.components.tools_registry import (
    _AGENT_PRIVILEGES,
    ToolMetadata,
    ToolsRegistry,
)
from duo_workflow_service.executor.outbox import Outbox
from lib.events import GLReportingEventContext
from lib.internal_events.client import InternalEventsClient

__all__ = ["DryRunFlowValidator", "VALIDATION_FLOW_TYPE", "VALIDATION_USER"]

_VALIDATION_WORKFLOW_ID = "_validation_"
VALIDATION_FLOW_TYPE = GLReportingEventContext(
    legacy_workflow_type="_validation_",
    flow_definition="_validation_",
    is_ai_catalog_item=False,
)
VALIDATION_USER = CloudConnectorUser(authenticated=False)

_DISABLED_INTERNAL_EVENT_CLIENT = InternalEventsClient(
    enabled=False,
    endpoint="",
    app_id="",
    namespace="",
    batch_size=1,
    thread_count=1,
)


class _StubPromptRegistry(BasePromptRegistry):
    """Prompt registry stub used during dry-run compilation.

    * ``get_required_variables()`` — delegates to the real registry so that
        component ``model_validator`` hooks see genuine Jinja2 variable sets.

    * ``get_on_behalf()`` — returns a ``_StubPrompt`` sentinel so that
        component ``attach()`` calls do not attempt real LLM model loading.
        The sentinel is stored by component nodes but never invoked during a
        dry-run compile.
    """

    def __init__(
        self,
        real_registry: BasePromptRegistry,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._real_registry = real_registry

    def get(
        self,
        prompt_id: str,
        prompt_version: Optional[str],
        model_metadata: Optional[TypeModelMetadata] = None,
        tools: Optional[List[BaseTool]] = None,
        **kwargs: Any,
    ) -> Prompt:
        """Not reached during validation — ``get_on_behalf`` is intercepted instead."""
        raise NotImplementedError(
            "_StubPromptRegistry.get() should not be called during validation"
        )

    def get_required_variables(
        self,
        prompt_id: str,
        prompt_version: Optional[str],
    ) -> set[str]:
        """Delegate to the real registry so component validators see genuine variable sets."""
        return self._real_registry.get_required_variables(prompt_id, prompt_version)

    def get_on_behalf(self, *args: Any, **kwargs: Any) -> Any:
        """Return None — the result is stored by component nodes but never invoked.

        ``AgentComponent.attach()`` passes the return value directly to
        ``AgentNode(prompt=...)`` which stores it as ``self._prompt``.  That
        attribute is only accessed inside ``AgentNode.run()``, which is async
        graph execution.  Execution is unconditionally blocked by
        ``DryRunFlowValidator._compile_and_run_graph()``, so ``None`` is safe.
        """


def _make_validation_tools_registry() -> ToolsRegistry:
    """Build a real ``ToolsRegistry`` suitable for dry-run flow validation.

    All agent privileges are granted so every named tool resolves correctly.
    ``ToolMetadata`` fields that require live infrastructure (gitlab client,
    project) are ``None`` — tool instances store but never access them until
    ``_arun`` / ``_execute`` is called.

    Returns:
        A fully-populated ``ToolsRegistry`` containing real tool instances.
    """
    stub_metadata = ToolMetadata(
        outbox=Outbox(),
        gitlab_client=None,  # type: ignore[typeddict-item]
        gitlab_host="",
        project=None,
    )
    all_privileges = list(_AGENT_PRIVILEGES.keys())
    return ToolsRegistry(
        enabled_tools=all_privileges,
        preapproved_tools=all_privileges,
        tool_call_approvals={},
        tool_metadata=stub_metadata,
        mcp_tools=[],
    )


class DryRunFlowValidator(Flow):
    """Validates a v1 flow configuration via dry-run compilation.

    Inherits from ``Flow`` to reuse the production ``_compile()`` path exactly,
    ensuring validation can never drift from production behaviour.

    ``_compile_and_run_graph`` is overridden to raise ``RuntimeError``, making
    this class non-executable.  All real execution paths are therefore blocked
    while ``validate()`` can still call ``_compile()`` directly.

    Usage::

        DryRunFlowValidator(
            config=config,
            prompt_registry=prompt_registry,
            internal_event_client=internal_event_client,
        ).validate()
    """

    def __init__(
        self,
        config: FlowConfig,
        prompt_registry: BasePromptRegistry,
        internal_event_client: InternalEventsClient,
    ) -> None:
        """Initialise with the minimum required dependencies.

        Args:
            config: Parsed flow configuration to validate.
            prompt_registry: Used by the parent ``Flow`` to register inline
                prompts; replaced by a stub before compilation so no real
                model loading occurs.
            internal_event_client: Events client (no events are emitted
                during validation).
        """
        super().__init__(
            workflow_id=_VALIDATION_WORKFLOW_ID,
            workflow_metadata={},
            workflow_type=VALIDATION_FLOW_TYPE,
            user=VALIDATION_USER,
            config=config,
            prompt_registry=prompt_registry,
            internal_event_client=internal_event_client,
        )
        # Replace the flow-scoped prompt registry with a stub that delegates
        # get_required_variables() to the real registry while short-circuiting
        # get_on_behalf() to avoid real LLM model loading.
        self._flow_prompt_registry = _StubPromptRegistry(
            real_registry=self._flow_prompt_registry,
            internal_event_client=_DISABLED_INTERNAL_EVENT_CLIENT,
            model_limits=ConfigModelLimits(),
        )
        # Enable strict variable validation so ExtraInputVariablesError is raised
        # during dry-run compilation. Production flows leave this as False.
        # Inject schema_registry so response schema validation works without DI.
        schema_registry = ResponseSchemaRegistry()
        for comp_config in self._config.components:
            comp_config["strict_validation"] = True
            comp_config["schema_registry"] = schema_registry

    def validate(self) -> None:
        """Run dry-run compilation to validate the flow configuration.

        Exercises component construction, tool-name resolution, routing, and
        prompt-variable validation without touching any real external systems.

        Raises:
            ValueError: If the configuration is invalid (unknown component
                types, missing entry_point, unresolvable tool names, duplicate
                component names, routing errors, or prompt-variable mismatches).
        """
        try:
            self._compile(
                goal="",
                tools_registry=_make_validation_tools_registry(),
                checkpointer=MemorySaver(),
            )
        except Exception as exc:
            raise ValueError(
                f"Flow compilation failed during validation: {exc}"
            ) from exc

    async def _compile_and_run_graph(self, *args: Any, **kwargs: Any) -> None:
        """Unconditionally blocked — ``DryRunFlowValidator`` must not execute graphs."""
        raise RuntimeError(
            "DryRunFlowValidator is not executable. Call validate() instead."
        )

    async def _handle_workflow_failure(
        self, error: BaseException, compiled_graph: Any, graph_config: Any
    ) -> None:
        """No-op: validation never runs the graph."""
