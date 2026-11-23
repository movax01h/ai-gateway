from pathlib import Path
from typing import Any, Optional, cast
from unittest.mock import Mock

from gitlab_cloud_connector import CloudConnectorUser, UserClaims
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import Field

from ai_gateway.config import ConfigModelLimits
from ai_gateway.prompts.config import ModelClassProvider
from ai_gateway.prompts.feature_roots import discover_feature_prompts
from ai_gateway.prompts.in_memory_registry import InMemoryPromptRegistry
from ai_gateway.prompts.registry import LocalPromptRegistry
from duo_workflow_service.agent_platform.v1.chat_engine import ChatFlow
from duo_workflow_service.agent_platform.v1.flows.flow_config import PartialFlowConfig
from duo_workflow_service.agent_platform.v1.flows.graph_builder import FlowGraph
from duo_workflow_service.components.tools_registry import Toolset, ToolsRegistry
from lib.events import GLReportingEventContext
from lib.internal_events.client import InternalEventsClient

__all__ = ["AGENT", "RecordingModel", "chat_graph", "template_chat_flow"]

AGENT = "agent"

# The smallest chat-partial flow on the ``template`` format: one agent whose ``message`` block renders the text of
# what it received.
_CONFIG = Path(__file__).parents[4] / "prompts" / "config" / "chat_partial_template.yml"


def _model_config() -> dict[str, Any]:
    # A flow's prompts take their model from model selection at runtime; these resolve one directly. The registry
    # consumes the params it reads, so each prompt gets its own.
    return {
        "params": {"model_class_provider": "anthropic", "model": "claude-3-5-sonnet"}
    }


class RecordingModel(FakeMessagesListChatModel):
    """A fake model that keeps what each call sent it, so a test reads a turn as the model did."""

    calls: list[list[BaseMessage]] = Field(default_factory=list)

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model": "fake-model"}

    def _generate(self, messages: list[BaseMessage], *args: Any, **kwargs: Any) -> Any:
        self.calls.append(list(messages))
        return super()._generate(messages, *args, **kwargs)

    def conversation(self, call: int = -1) -> list[str]:
        """The text of every message one call read, the system prompt left out."""
        return [
            message.text
            for message in self.calls[call]
            if not isinstance(message, SystemMessage)
        ]


def template_chat_flow(
    model: RecordingModel, workflow_id: str, **kwargs: Any
) -> ChatFlow:
    """``ChatFlow`` as the registry builds it for a version on the ``template`` format, answering with ``model``."""
    discover_feature_prompts()
    config = PartialFlowConfig.from_registry_file(
        _CONFIG, "developer", "4.0.0"
    ).to_config()
    flow = ChatFlow(
        config=config,
        workflow_id=workflow_id,
        workflow_metadata={
            "git_url": "https://gitlab.com/test/project",
            "git_sha": "abc",
        },
        workflow_type=GLReportingEventContext.from_workflow_definition("chat"),
        user=CloudConnectorUser(
            authenticated=True, claims=UserClaims(scopes=["duo_agent_platform"])
        ),
        prompt_registry=LocalPromptRegistry.from_local_yaml(
            prompt_template_factories={},
            model_factories={
                ModelClassProvider.ANTHROPIC: cast(Any, lambda **_: model)
            },
            internal_event_client=Mock(spec=InternalEventsClient),
            model_limits=ConfigModelLimits(),
            custom_models_enabled=False,
        ),
        **kwargs,
    )
    registry = cast(InMemoryPromptRegistry, flow._flow_prompt_registry)
    for prompt in config.prompts or []:
        registry.register_prompt(
            prompt.prompt_id, prompt.to_prompt_data() | {"model": _model_config()}
        )
    # What a run fetches before its graph compiles.
    flow._project = None
    flow._namespace = None
    return flow


def chat_graph(flow: ChatFlow, tools_registry: Optional[Mock] = None) -> FlowGraph:
    """The graph ``ChatFlow`` builds for its config: the real agent and the seeded boundary."""
    if tools_registry is None:
        toolset = Mock(spec=Toolset)
        toolset.bindable = []
        toolset.__contains__ = Mock(return_value=False)
        tools_registry = Mock(spec=ToolsRegistry)
        tools_registry.mcp_tool_names.return_value = []
        tools_registry.ask_listed_tool_names.return_value = set()
        tools_registry.toolset.return_value = toolset
    return flow._graph_builder(tools_registry).build(flow._config)
