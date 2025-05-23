from pathlib import Path
from typing import NamedTuple, Optional, Type

import structlog
import yaml
from poetry.core.constraints.version import Version, parse_constraint

from ai_gateway.config import ConfigModelLimits
from ai_gateway.internal_events.client import InternalEventsClient
from ai_gateway.model_metadata import TypeModelMetadata
from ai_gateway.prompts.base import BasePromptRegistry, Prompt
from ai_gateway.prompts.config import BaseModelConfig, ModelClassProvider, PromptConfig
from ai_gateway.prompts.typing import TypeModelFactory

__all__ = ["LocalPromptRegistry", "PromptRegistered"]

log = structlog.stdlib.get_logger("prompts")


class PromptRegistered(NamedTuple):
    klass: Type[Prompt]
    versions: dict[str, PromptConfig]


class LocalPromptRegistry(BasePromptRegistry):
    key_prompt_type_base: str = "base"

    def __init__(
        self,
        prompts_registered: dict[str, PromptRegistered],
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        default_prompts: dict[str, str],
        internal_event_client: InternalEventsClient,
        model_limits: ConfigModelLimits,
        custom_models_enabled: bool,
        disable_streaming: bool = False,
    ):
        self.prompts_registered = prompts_registered
        self.model_factories = model_factories
        self.default_prompts = default_prompts
        self.internal_event_client = internal_event_client
        self.model_limits = model_limits
        self.custom_models_enabled = custom_models_enabled
        self.disable_streaming = disable_streaming

    def _resolve_id(
        self,
        prompt_id: str,
        model_metadata: Optional[TypeModelMetadata] = None,
    ) -> str:
        if model_metadata:
            return f"{prompt_id}/{model_metadata.name}"

        type = self.default_prompts.get(prompt_id, self.key_prompt_type_base)
        return f"{prompt_id}/{type}"

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

    def get(
        self,
        prompt_id: str,
        prompt_version: str,
        model_metadata: Optional[TypeModelMetadata] = None,
    ) -> Prompt:
        prompt_id = self._resolve_id(prompt_id, model_metadata)

        log.info("Resolved prompt id", prompt_id=prompt_id)

        prompt_registered = self.prompts_registered[prompt_id]
        config = self._get_prompt_config(prompt_registered.versions, prompt_version)
        model_class_provider = config.model.params.model_class_provider
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
        )

        return prompt_registered.klass(
            model_factory,
            config,
            model_metadata,
            disable_streaming=self.disable_streaming,
        )

    @classmethod
    def from_local_yaml(
        cls,
        class_overrides: dict[str, Type[Prompt]],
        model_factories: dict[ModelClassProvider, TypeModelFactory],
        default_prompts: dict[str, str],
        internal_event_client: InternalEventsClient,
        model_limits: ConfigModelLimits,
        custom_models_enabled: bool = False,
        disable_streaming: bool = False,
    ) -> "LocalPromptRegistry":
        """Iterate over all prompt definition files matching [usecase]/[type]/[version].yml, and create a corresponding
        prompt for each one.

        The base Prompt class is
        used if no matching override is provided in `class_overrides`.
        """

        base_path = Path(__file__).parent
        prompts_definitions_dir = base_path / "definitions"
        model_configs_dir = (
            base_path / "model_configs"
        )  # New directory for model configs
        prompts_registered = {}

        # Parse model config YAML files
        model_configs = {
            file.stem: cls._parse_base_model(file)
            for file in model_configs_dir.glob("*.yml")
        }

        # Iterate over each folder
        for path in prompts_definitions_dir.glob("**"):
            # Iterate over each version file
            versions = {
                version.stem: cls._process_version_file(version, model_configs)
                for version in path.glob("*.yml")
            }

            # If there were no yml files in this folder, skip it
            if not versions:
                continue

            # E.g., "chat/react/base", "generate_description/mistral", etc.
            prompt_id_with_model_name = path.relative_to(prompts_definitions_dir)

            klass = class_overrides.get(str(prompt_id_with_model_name.parent), Prompt)
            prompts_registered[str(prompt_id_with_model_name)] = PromptRegistered(
                klass=klass, versions=versions
            )

        log.info(
            "Initializing prompt registry from local yaml",
            default_prompts=default_prompts,
            custom_models_enabled=custom_models_enabled,
        )

        return cls(
            prompts_registered,
            model_factories,
            default_prompts,
            internal_event_client,
            model_limits,
            custom_models_enabled,
            disable_streaming,
        )

    @classmethod
    def _parse_base_model(cls, file_name: Path) -> BaseModelConfig:
        """Parses a YAML file and converts its content to a BaseModelConfig object.

        This method reads the specified YAML file, extracts the configuration
        parameters, and constructs a BaseModelConfig object. It handles the
        conversion of YAML data types to appropriate Python types.

        Args:
            file (Path): A Path object pointing to the YAML file to be parsed.

        Returns:
            BaseModelConfig: An instance of BaseModelConfig containing the
            parsed configuration data.
        """

        with open(file_name, "r") as fp:
            return BaseModelConfig(**yaml.safe_load(fp))

    @classmethod
    def _process_version_file(
        cls, version_file: Path, model_configs: dict[str, BaseModelConfig]
    ) -> PromptConfig:
        """Processes a single version YAML file and returns a PromptConfig.

        Args:
            version_file: Path to the version YAML file
            model_configs: Dictionary of model configurations

        Returns:
            PromptConfig: Processed prompt configuration
        """

        with open(version_file, "r") as fp:
            prompt_config_params = yaml.safe_load(fp)

            if "config_file" in prompt_config_params["model"]:
                model_config = prompt_config_params["model"]["config_file"]
                config_for_general_model = model_configs.get(model_config)
                if config_for_general_model:
                    prompt_config_params = cls._patch_model_configuration(
                        config_for_general_model, prompt_config_params
                    )

            return PromptConfig(**prompt_config_params)

    @classmethod
    def _patch_model_configuration(
        cls, config_for_general_model: BaseModelConfig, prompt_config_params: dict
    ) -> dict:
        params = {
            **config_for_general_model.params.model_dump(),
            **prompt_config_params["model"].get("params", {}),
        }

        return {
            **prompt_config_params,
            "model": {
                "name": config_for_general_model.name,
                "params": params,
            },
        }
