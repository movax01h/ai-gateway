from typing import Any, Optional, cast, override

from gitlab_cloud_connector import GitLabUnitPrimitive
from langchain_core.tools import BaseTool

from ai_gateway.model_metadata import TypeModelMetadata
from ai_gateway.prompts.base import BasePromptRegistry, Prompt, TemplateNotFoundError
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

        super().__init__(
            shared_registry.internal_event_client, shared_registry.model_limits
        )

        # Shared singleton to avoid duplication
        self.shared_registry = cast(LocalPromptRegistry, shared_registry)
        self._raw_prompt_data: dict[str, dict] = {}

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

        model_params: dict[str, Any]

        if model_metadata:
            model_params = {}
            model_class_provider = model_metadata.llm_definition.model_class_provider
        elif model_from_prompt := cast(dict, raw_data.get("model")):
            model_params = model_from_prompt["params"]
            model_class_provider = model_params.pop("model_class_provider")
        else:
            raise ValueError(f"Model config not provided for prompt {prompt_id}")

        unit_primitives = raw_data.get("unit_primitives")
        prompt_config = PromptConfig(
            name=prompt_id,
            model=ModelConfig(params=model_params),  # type: ignore[arg-type]
            unit_primitive=(
                GitLabUnitPrimitive(unit_primitives[0])
                if unit_primitives
                else GitLabUnitPrimitive.DUO_AGENT_PLATFORM
            ),
            prompt_template=raw_data["prompt_template"],
            params=raw_data.get("params"),
        )

        return self.shared_registry._build_prompt(
            model_class_provider=model_class_provider,
            config=prompt_config,
            model_metadata=model_metadata,
            tool_choice=tool_choice,
            tools=tools,
            **kwargs,
        )

    @override
    def get_required_variables(
        self,
        prompt_id: str,
        prompt_version: Optional[str],
    ) -> set[str]:
        """Return Jinja2 variables required by an inline or file-based prompt.

        Inline prompts (``prompt_version=None``) are resolved from in-memory
        storage.  Registry-based prompts (``prompt_version`` set) are delegated
        to the shared ``LocalPromptRegistry``.

        Args:
            prompt_id: Prompt identifier to inspect.
            prompt_version: Version constraint, or ``None`` for inline prompts.

        Returns:
            Flat set of required variable names.

        Raises:
            TemplateNotFoundError: When the prompt cannot be resolved (not
                registered inline and no version provided, or the file-based
                registry cannot find it).
        """
        if prompt_version:
            return self.shared_registry.get_required_variables(
                prompt_id, prompt_version
            )

        raw_data = self._raw_prompt_data.get(prompt_id)
        if not raw_data:
            raise TemplateNotFoundError(
                f"Cannot resolve required variables for '{prompt_id}': "
                "prompt is not registered inline and no prompt_version was provided"
            )

        prompt_template = raw_data.get("prompt_template", {})
        variables: set[str] = set()
        for template_str in prompt_template.values():
            if template_str:
                variables.update(self._collect_jinja2_variables(template_str))
        return variables

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
