import importlib
from functools import cache
from pathlib import Path
from typing import Any, List, NamedTuple, Optional, Type, cast

import structlog
import yaml
from langchain.tools import BaseTool
from poetry.core.constraints.version import Version, parse_constraint

from ai_gateway.config import ConfigModelLimits
from ai_gateway.model_metadata import (
    ModelMetadata,
    TypeModelMetadata,
    create_model_metadata,
)
from ai_gateway.prompts.base import BasePromptRegistry, Prompt
from ai_gateway.prompts.bind_tools_cache import BindToolsCacheProtocol
from ai_gateway.prompts.config import BaseModelConfig, ModelClassProvider, PromptConfig
from ai_gateway.prompts.typing import TypeModelFactory, TypePromptTemplateFactory
from lib.internal_events.client import InternalEventsClient
from lib.internal_events.context import current_event_context

__all__ = ["LocalPromptRegistry", "PromptRegistered"]

log = structlog.stdlib.get_logger("prompts")


LEGACY_MODEL_MAPPING = {
    "chat/agent": {
        "1.0.0": "claude_sonnet_4_20250514",
    },
    "chat/build_reader": {
        "1.0.0": "claude_sonnet_4_20250514",
        "1.0.1-dev": "claude_sonnet_4_20250514",
        "1.0.1": "claude_sonnet_4_20250514",
        "1.1.0": "claude_sonnet_4_20250514",
    },
    "chat/commit_reader": {
        "1.0.0": "claude_sonnet_4_20250514",
        "1.0.1-dev": "claude_sonnet_4_20250514",
        "1.0.1": "claude_sonnet_4_20250514",
        "1.1.0": "claude_sonnet_4_20250514",
    },
    "chat/documentation_search": {
        "1.0.0": "claude_sonnet_4_20250514",
        "1.0.1": "claude_sonnet_4_20250514",
        "1.1.0": "claude_sonnet_4_20250514",
    },
    "chat/epic_reader": {
        "1.0.0": "claude_sonnet_4_20250514",
        "1.0.1-dev": "claude_sonnet_4_20250514",
        "1.0.1": "claude_sonnet_4_20250514",
        "1.1.0": "claude_sonnet_4_20250514",
    },
    "chat/explain_code": {
        "0.0.1-dev": "claude_sonnet_4_20250514",
        "1.0.0": "claude_sonnet_4_20250514",
        "1.0.1": "claude_sonnet_4_20250514",
        "1.1.0-dev": "claude_sonnet_4_20250514",
        "1.1.0": "claude_sonnet_4_20250514",
    },
    "chat/explain_vulnerability": {
        "0.0.1-dev": "claude_sonnet_4_20250514",
        "1.0.0": "claude_sonnet_4_20250514",
        "1.0.1": "claude_sonnet_4_20250514",
        "1.1.0": "claude_sonnet_4_20250514",
    },
    "chat/fix_code": {
        "0.0.1-dev": "claude_sonnet_4_20250514",
        "1.0.0": "claude_sonnet_4_20250514",
        "1.0.1": "claude_sonnet_4_20250514",
        "1.1.0-dev": "claude_sonnet_4_20250514",
        "1.1.0": "claude_sonnet_4_20250514",
    },
    "chat/issue_reader": {
        "1.0.0": "claude_sonnet_4_20250514",
        "1.0.1-dev": "claude_sonnet_4_20250514",
        "1.0.1": "claude_sonnet_4_20250514",
        "1.1.0": "claude_sonnet_4_20250514",
    },
    "chat/merge_request_reader": {
        "1.0.0": "claude_sonnet_4_20250514",
        "1.0.1-dev": "claude_sonnet_4_20250514",
        "1.0.1": "claude_sonnet_4_20250514",
        "1.1.0": "claude_sonnet_4_20250514",
    },
    "chat/react": {
        "1.0.0": "claude_sonnet_4_20250514",
        "1.0.1": "claude_sonnet_4_20250514",
        "1.0.2-dev": "claude_sonnet_4_20250514",
        "1.1.0": "claude_sonnet_4_20250514",
    },
    "chat/refactor_code": {
        "0.0.1-dev": "claude_sonnet_4_20250514",
        "1.0.0": "claude_sonnet_4_20250514",
        "1.0.1": "claude_sonnet_4_20250514",
        "1.1.0-dev": "claude_sonnet_4_20250514",
        "1.1.0": "claude_sonnet_4_20250514",
    },
    "chat/summarize_comments": {
        "1.0.0": "claude_sonnet_4_20250514",
        "1.1.0": "claude_sonnet_4_20250514",
    },
    "chat/troubleshoot_job": {
        "0.0.1-dev": "claude_sonnet_4_20250514",
        "1.0.0": "claude_3_sonnet_20240229",
        "1.0.1-alpha": "claude_sonnet_4_20250514",
        "1.0.2": "claude_sonnet_4_20250514",
        "1.1.0-dev": "claude_sonnet_4_20250514",
        "1.1.0": "claude_sonnet_4_20250514",
    },
    "chat/write_tests": {
        "0.0.1-dev": "claude_sonnet_4_20250514",
        "1.0.0": "claude_sonnet_4_20250514",
        "1.0.1": "claude_sonnet_4_20250514",
        "1.1.0-dev": "claude_sonnet_4_20250514",
        "1.1.0": "claude_sonnet_4_20250514",
    },
    "chat/work_item_reader": {
        "1.0.0": "claude_sonnet_4_20250514",
    },
    "code_suggestions/generations": {
        "1.0.0": "claude_sonnet_4_20250514",
        "1.0.1": "claude_sonnet_4_20250514",
        "1.0.2": "claude_sonnet_4_20250514",
        "1.1.0-dev": "claude_sonnet_4_20250514",
        "1.1.0": "claude_sonnet_4_20250514",
        "1.2.0-dev": "gemini_2_5_flash_vertex",
        "2.0.0": "claude_sonnet_4_20250514_vertex",
        "2.0.1": "claude_sonnet_4_20250514_vertex",
        "2.0.2-dev": "claude_sonnet_4_20250514_vertex",
        "2.0.2": "claude_sonnet_4_20250514_vertex",
        "2.0.3": "claude_sonnet_4_20250514",
        "3.0.2-dev": "claude_sonnet_4_20250514",
    },
    "generate_commit_message": {
        "1.0.0": "claude_sonnet_4_20250514",
        "1.1.0": "claude_sonnet_4_20250514",
        "1.2.0": "claude_sonnet_4_20250514",
    },
    "glab_ask_git_command": {
        "1.0.0": "claude_haiku_4_5_20251001",
        "1.0.1": "claude_haiku_4_5_20251001",
    },
    "measure_comment_temperature": {
        "1.0.0": "gemini_1_5_flash_vertex",
        "1.0.1": "gemini_2_0_flash_lite_vertex",
        "1.0.2": "gemini_2_0_flash_lite_vertex",
    },
    "resolve_vulnerability": {
        "0.0.1-dev": "claude_sonnet_4_20250514",
        "1.0.0": "claude_sonnet_4_20250514",
        "1.0.1": "claude_sonnet_4_20250514",
        "1.0.2": "claude_sonnet_4_20250514",
    },
    "review_merge_request": {
        "0.9.0": "claude_sonnet_4_20250514",
        "1.0.0": "claude_sonnet_4_20250514",
        "1.1.0": "claude_sonnet_4_20250514",
        "1.2.0": "claude_sonnet_4_20250514",
        "1.3.0": "claude_sonnet_4_20250514",
    },
    "summarize_new_merge_request": {
        "1.0.0": "claude_sonnet_4_20250514",
        "2.0.0": "claude_sonnet_4_20250514",
        "2.0.1": "claude_sonnet_4_20250514",
        "2.0.2-dev": "claude_sonnet_4_20250514",
        "2.0.2": "claude_sonnet_4_20250514",
        "2.1.0-dev": "claude_sonnet_4_20250514",
        "2.1.0": "claude_sonnet_4_20250514",
    },
    "summarize_review": {
        "1.0.0": "claude_sonnet_4_20250514",
        "2.0.0": "claude_sonnet_4_20250514",
        "2.1.0": "claude_sonnet_4_20250514",
    },
}


def feature_setting_for_prompt_id(prompt_id: str) -> str:
    feature_setting = prompt_id.split("/", 1)[0]

    # the folder for chat in the definitions doesn't match the feature_setting name
    return "duo_chat" if feature_setting == "chat" else feature_setting


class PromptRegistered(NamedTuple):
    versions: dict[str, PromptConfig]


class LocalPromptRegistry(BasePromptRegistry):
    key_prompt_type_base: str = "base"

    def __init__(
        self,
        prompt_template_factories: dict[str, TypePromptTemplateFactory | str],
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        internal_event_client: InternalEventsClient,
        model_limits: ConfigModelLimits,
        custom_models_enabled: bool,
        disable_streaming: bool = False,
        bind_tools_cache: Optional[BindToolsCacheProtocol] = None,
    ):
        self.prompt_template_factories = prompt_template_factories
        self.model_factories = model_factories
        self.internal_event_client = internal_event_client
        self.model_limits = model_limits
        self.custom_models_enabled = custom_models_enabled
        self.disable_streaming = disable_streaming
        self.bind_tools_cache = bind_tools_cache

    def _resolve_id(
        self,
        prompt_id: str,
        family: list[str],
    ) -> Path:
        base_path = Path(__file__).parent
        prompts_definitions_dir = base_path / "definitions" / prompt_id

        # Look for the first existing prompt definition in the family, or `base` as a last option
        for prompt_folder in family + [self.key_prompt_type_base]:
            prompt_path = prompts_definitions_dir / prompt_folder
            if prompt_path.exists() and prompt_path.is_dir():
                return prompt_path

        raise FileNotFoundError(
            f"Prompt definition directory not found: {prompts_definitions_dir}"
        )

    @cache  # pylint: disable=method-cache-max-size-none
    def _load_prompt_definition(
        self,
        prompt_id: str,
        prompt_path: Path,
    ) -> PromptRegistered:
        versions = {
            version.stem: self._process_version_file(version)
            for version in prompt_path.glob("*.yml")
        }

        if not versions:
            raise ValueError(f"No version YAML files found for prompt id: {prompt_id}")

        return PromptRegistered(
            versions=versions,
        )

    def _get_prompt_config(
        self, versions: dict[str, PromptConfig], prompt_version: str
    ) -> PromptConfig:
        # Parse constraint according to poetry rules. See
        # https://python-poetry.org/docs/dependency-specification/#version-constraints
        constraint = parse_constraint(prompt_version)
        all_versions = [Version.parse(version) for version in versions.keys()]

        # If the query is not "simple" (in other words, it's not referencing specific versions but is a constraint or
        # set of constraints, for example a range) we only want to consider stable versions. This allows us to not
        # auto-serve dev/rc versions to clients using queries like `^1.0.0`
        if not constraint.is_simple():
            all_versions = [version for version in all_versions if version.is_stable()]

        compatible_versions = list(filter(constraint.allows, all_versions))
        if not compatible_versions:
            log.info(
                "No compatible versions found",
                versions=versions,
                prompt_version=prompt_version,
            )
            raise ValueError(
                f"No prompt version found matching the query: {prompt_version}"
            )
        compatible_versions.sort(reverse=True)

        return versions[str(compatible_versions[0])]

    def _default_model_metadata(
        self, prompt_id: str, prompt_version: str
    ) -> TypeModelMetadata | None:
        # For backwards compatibility with client code that doesn't send model_metadata and would've used the model from
        # the `base` prompt, create model metadata from know version mappings or the feature setting default
        if identifier := LEGACY_MODEL_MAPPING.get(prompt_id, {}).get(
            prompt_version, None
        ):
            model_metadata = create_model_metadata(
                {"provider": "gitlab", "identifier": identifier}
            )
            if model_metadata:
                # For these legacy prompt version tied to a model, we always used the `base` prompt, so we override the
                # `family` in case the current model from models.yml specifies a different value for this property
                model_metadata.family = [self.key_prompt_type_base]
            return model_metadata

        return create_model_metadata(
            {
                "provider": "gitlab",
                "feature_setting": feature_setting_for_prompt_id(prompt_id),
            }
        )

    def _adjust_tool_choice_for_model(
        self, tool_choice: Optional[str], model_metadata: Optional[TypeModelMetadata]
    ) -> Optional[str]:
        """Adjust tool_choice based on model-specific requirements.

        Different model providers have different tool_choice requirements:
        - Bedrock and Azure models don't support 'any' as a tool_choice value, so we convert it to 'required'.

        Args:
            tool_choice: The original tool_choice value
            model_metadata: The model metadata

        Returns:
            The adjusted tool_choice value
        """

        model_identifier = getattr(model_metadata, "identifier", None)
        if (
            tool_choice == "any"
            and model_identifier
            and ("bedrock/" in model_identifier or "azure/" in model_identifier)
        ):
            return "required"
        return tool_choice

    # prompt_version is never None when called on LocalPromptRegistry
    # but it must be set to str | None to match the abstract signature
    def get(
        self,
        prompt_id: str,
        prompt_version: str | None,
        model_metadata: Optional[TypeModelMetadata] = None,
        tools: Optional[List[BaseTool]] = None,
        tool_choice: Optional[str] = None,  # auto, any, <tool name>. By default, auto.
        **kwargs: Any,
    ) -> Prompt:
        try:
            if not model_metadata:
                model_metadata = self._default_model_metadata(prompt_id, prompt_version)  # type: ignore[arg-type]

            family = model_metadata.family if model_metadata else []
            prompt_path = self._resolve_id(prompt_id, family)

            log.info("Resolved prompt id", prompt_id=prompt_id, prompt_path=prompt_path)

            prompt_registered = self._load_prompt_definition(prompt_id, prompt_path)
        except (FileNotFoundError, ValueError) as e:
            raise ValueError(
                f"Failed to load prompt definition for '{prompt_id}': {e}"
            ) from e

        config = self._get_prompt_config(prompt_registered.versions, prompt_version)  # type: ignore[arg-type]
        model_class_provider = (
            # From model definition in models.yml
            model_metadata.llm_definition.params.get("model_class_provider")
            if model_metadata
            else None
        ) or config.model.params.model_class_provider  # From prompt file

        model_factory = self.model_factories.get(model_class_provider, None)

        if not model_factory:
            raise ValueError(
                f"unrecognized model class provider `{model_class_provider}`."
            )

        log.info(
            "Returning prompt from the registry",
            prompt_id=prompt_id,
            prompt_name=config.name,
            prompt_version=prompt_version,
            model_class_provider=model_class_provider,
            model_identifier=(
                # identifier works for custom models, name works for gitlab models
                getattr(model_metadata, "identifier", None)
                or getattr(model_metadata, "name", None)
                if isinstance(model_metadata, ModelMetadata)
                else None
            ),
            gitlab_feature_enabled_by_namespace_ids=getattr(
                current_event_context.get(), "feature_enabled_by_namespace_ids", None
            ),
        )

        prompt_template_override = self.prompt_template_factories.get(prompt_id, None)
        prompt_template_factory: TypePromptTemplateFactory | None
        if isinstance(prompt_template_override, str):
            prompt_template_factory = cast(
                TypePromptTemplateFactory,
                self._resolve_string_class_name(prompt_template_override),
            )
        else:
            prompt_template_factory = cast(
                TypePromptTemplateFactory | None, prompt_template_override
            )

        # Adjust tool_choice for model-specific requirements
        tool_choice = self._adjust_tool_choice_for_model(tool_choice, model_metadata)

        return Prompt(
            model_factory,
            config,
            model_metadata,
            prompt_template_factory,
            disable_streaming=self.disable_streaming,
            tools=tools,
            tool_choice=tool_choice,
            bind_tools_cache=self.bind_tools_cache,
            **kwargs,
        )

    @classmethod
    def from_local_yaml(
        cls,
        prompt_template_factories: dict[str, TypePromptTemplateFactory | str],
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        internal_event_client: InternalEventsClient,
        model_limits: ConfigModelLimits,
        custom_models_enabled: bool = False,
        disable_streaming: bool = False,
        bind_tools_cache: Optional[BindToolsCacheProtocol] = None,
    ) -> "LocalPromptRegistry":
        """Create a LocalPromptRegistry with lazy loading enabled.

        Prompt definition files matching [usecase]/[type]/[version].yml are loaded on-demand when requested.
        """

        log.info(
            "Initializing prompt registry with lazy loading",
            custom_models_enabled=custom_models_enabled,
        )

        return cls(
            prompt_template_factories,
            model_factories,
            internal_event_client,
            model_limits,
            custom_models_enabled,
            disable_streaming,
            bind_tools_cache,
        )

    @classmethod
    def _resolve_string_class_name(
        cls, path: str
    ) -> Type[Prompt] | TypePromptTemplateFactory:
        parts = path.split(".")
        module = importlib.import_module(".".join(parts[:-1]))
        return getattr(module, parts[-1])

    @classmethod
    def _parse_base_model(cls, file_name: Path) -> BaseModelConfig:
        """Parses a YAML file and converts its content to a BaseModelConfig object.

        This method reads the specified YAML file, extracts the configuration
        parameters, and constructs a BaseModelConfig object. It handles the
        conversion of YAML data types to appropriate Python types.

        Args:
            file_name (Path): A Path object pointing to the YAML file to be parsed.

        Returns:
            BaseModelConfig: An instance of BaseModelConfig containing the
            parsed configuration data.
        """

        with open(file_name, "r") as fp:
            return BaseModelConfig(**yaml.safe_load(fp))

    @classmethod
    def _process_version_file(cls, version_file: Path) -> PromptConfig:
        """Processes a single version YAML file and returns a PromptConfig.

        Args:
            version_file: Path to the version YAML file

        Returns:
            PromptConfig: Processed prompt configuration
        """

        with open(version_file, "r") as fp:
            return PromptConfig(**yaml.safe_load(fp))
