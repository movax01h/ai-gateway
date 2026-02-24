from typing import Any, Optional, cast, override

from langchain_core.tools import BaseTool

from ai_gateway.model_metadata import TypeModelMetadata
from ai_gateway.prompts.base import BasePromptRegistry, Prompt
from ai_gateway.prompts.config import ModelConfig, PromptConfig
from ai_gateway.prompts.registry import LocalPromptRegistry


class InMemoryPromptRegistry(BasePromptRegistry):
    """Registry for flow-local prompts defined inline within flow YAML configurations.

    This registry enables flows to define prompts locally within their YAML config
    rather than requiring separate prompt definition files. It provides automatic
    routing between local (in-memory) and remote (repo-based) prompts based on
    whether prompt_version is provided.

    Routing Logic:
        - prompt_version=None: Use local prompts from flow YAML
        - prompt_version="^1.0.0": Delegate to shared file-based registry

    Example Flow YAML:
        prompts:
            - prompt_id: "my_local_prompt"
            model: {...}
            prompt_template:
                system: "You are a helpful assistant"
                user: "Task: {{goal}}"
    """

    _DEFAULT_VERSION = None

    def __init__(self, shared_registry: BasePromptRegistry):
        if not isinstance(shared_registry, LocalPromptRegistry):
            raise TypeError("only LocalPromptRegistry is supported at this moment")

        # Shared singleton to avoid duplication
        self.shared_registry = cast(LocalPromptRegistry, shared_registry)
        self._raw_prompt_data: dict[str, dict] = {}
        # Abstract attributes from shared registry
        self.internal_event_client = shared_registry.internal_event_client
        self.model_limits = shared_registry.model_limits

    def register_prompt(self, prompt_id: str, prompt_data: dict) -> None:
        """Register a prompt from flow yaml data.

        Args:
            prompt_id: The prompt identifier from flow yaml
            prompt_data: Raw prompt dict from flow yaml (contains model, prompt_template, etc.)
        """
        self._raw_prompt_data[prompt_id] = prompt_data

    def _process_prompt_data(self, prompt_id: str) -> dict:
        """Process raw prompt data and resolve config_file references."""
        raw_data = self._raw_prompt_data.get(prompt_id)
        if not raw_data:
            raise ValueError(f"Local prompt not found: {prompt_id}")

        return raw_data

    def _get_in_memory_prompt(
        self,
        prompt_id: str,
        model_metadata: Optional[TypeModelMetadata] = None,
        tools: Optional[list[BaseTool]] = None,
        tool_choice: Optional[str] = None,
        **kwargs: Any,
    ) -> Prompt:
        """Retrieve and instantiate a local prompt from in-memory storage.

        Converts flow YAML prompt data to PromptConfig format and creates a Prompt instance with the appropriate model
        factory.
        """
        raw_data = self._process_prompt_data(prompt_id)

        model_data: dict[str, Any]

        if model_metadata:
            model_data = {"params": model_metadata.llm_definition.params}
        elif model_from_prompt := raw_data.get("model"):
            model_data = model_from_prompt
        else:
            raise ValueError(f"Model config not provided for prompt {prompt_id}")

        prompt_config = PromptConfig(
            name=prompt_id,
            model=ModelConfig(**model_data),
            unit_primitives=raw_data.get("unit_primitives", []),
            prompt_template=raw_data["prompt_template"],
            params=raw_data.get("params"),
        )

        model_class_provider = (
            model_metadata.llm_definition.params.get("model_class_provider")
            if model_metadata
            else None
        ) or prompt_config.model.params.model_class_provider
        model_factory = self.shared_registry.model_factories.get(
            model_class_provider, None
        )

        if not model_factory:
            raise ValueError(
                f"unrecognized model class provider `{model_class_provider}`."
            )

        tool_choice = self.shared_registry._adjust_tool_choice_for_model(
            tool_choice, model_metadata
        )

        return Prompt(
            model_factory=model_factory,
            config=prompt_config,
            model_metadata=model_metadata,
            disable_streaming=self.shared_registry.disable_streaming,
            tools=tools,
            tool_choice=tool_choice,
            **kwargs,
        )

    @override
    def get(
        self,
        prompt_id: str,
        prompt_version: Optional[str],
        model_metadata: Optional[TypeModelMetadata] = None,
        tools: Optional[list[BaseTool]] = None,
        tool_choice: Optional[str] = None,  # auto, any, <tool name>. By default, auto.
        **kwargs: Any,
    ) -> Prompt:
        if not prompt_version:
            return self._get_in_memory_prompt(
                prompt_id,
                model_metadata,
                tools,
                tool_choice,
                **kwargs,
            )
        return self.shared_registry.get(
            prompt_id,
            prompt_version,
            model_metadata,
            tools,
            tool_choice,
            **kwargs,
        )
