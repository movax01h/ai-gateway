from abc import abstractmethod
from typing import Annotated, Any, Dict, Literal, Optional, override

from pydantic import (
    AnyUrl,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    UrlConstraints,
)

from ai_gateway.model_selection import LLMDefinition, ModelSelectionConfig
from lib.context import ModelSizeBucket, StarletteUser

PROVIDERS_WITHOUT_API_BASE = frozenset({"bedrock", "vertex_ai"})


class BaseModelMetadata(BaseModel):
    llm_definition: LLMDefinition
    friendly_name: Optional[Annotated[str, StringConstraints(max_length=255)]] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._user = None

    @abstractmethod
    def to_params(self) -> Dict[str, Any]:
        pass

    def add_user(self, user: StarletteUser):
        self._user = user


class AmazonQModelMetadata(BaseModelMetadata):
    provider: Literal["amazon_q"]
    name: Literal["amazon_q"]
    role_arn: Annotated[str, StringConstraints(max_length=255)]

    @override
    def to_params(self) -> Dict[str, Any]:
        return {
            "role_arn": self.role_arn,
            "user": self._user,
        }


class FireworksModelMetadata(BaseModelMetadata):
    provider: Literal["fireworks_ai"]
    name: Annotated[str, StringConstraints(max_length=255)]
    endpoint: Optional[Annotated[AnyUrl, UrlConstraints(max_length=255)]] = None
    api_key: Optional[Annotated[str, StringConstraints(max_length=2000)]] = None
    model_identifier: Optional[str] = None
    using_cache: Optional[bool] = None
    session_id: Optional[str] = None

    @override
    def to_params(self) -> Dict[str, Any]:
        params = {}

        if self.model_identifier:
            params["model"] = self.model_identifier

        if self.api_key:
            params["api_key"] = self.api_key

        if self.endpoint:
            params["api_base"] = str(self.endpoint).removesuffix("/")

        if self.using_cache is not None:
            params["using_cache"] = str(self.using_cache)

        if self.session_id is not None:
            params["session_id"] = self.session_id

        return params


class ModelMetadata(BaseModelMetadata):
    name: Annotated[str, StringConstraints(max_length=255)]
    provider: Annotated[str, StringConstraints(max_length=255)]
    endpoint: Optional[Annotated[AnyUrl, UrlConstraints(max_length=255)]] = None
    api_key: Optional[Annotated[str, StringConstraints(max_length=2000)]] = None
    identifier: Optional[Annotated[str, StringConstraints(max_length=1000)]] = None

    @override
    def to_params(self) -> Dict[str, Any]:
        """Retrieve model parameters for a given identifier.

        This function also allows setting custom provider details based on the identifier, like fetching endpoints based
        on AIGW location.
        """
        params: Dict[str, Any] = self.llm_definition.prompt_params.model_dump(
            exclude_none=True
        )

        if self.endpoint:
            params["api_base"] = str(self.endpoint).removesuffix("/")

        if self.identifier:
            provider, _, model_name = self.identifier.partition("/")
            if model_name:
                params["custom_llm_provider"] = provider
                params["model"] = model_name

                if provider in PROVIDERS_WITHOUT_API_BASE:
                    params.pop("api_base", None)
            else:
                params["custom_llm_provider"] = "custom_openai"
                params["model"] = self.identifier

        if self.api_key:
            params["api_key"] = self.api_key
        # Set a default dummy key to avoid LiteLLM errors
        # See https://gitlab.com/gitlab-org/gitlab/-/issues/520512
        elif params.get("custom_llm_provider", "") == "custom_openai":
            params["api_key"] = "dummy_key"

        return params


TypeModelMetadata = AmazonQModelMetadata | ModelMetadata | FireworksModelMetadata


class ModelMetadataBySize(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    default: TypeModelMetadata
    by_size: Dict[Literal["small", "large"], TypeModelMetadata] = Field(
        default_factory=dict
    )

    def get(self, model_size: ModelSizeBucket | None = None) -> TypeModelMetadata:
        if model_size is None:
            return self.default
        return self.by_size.get(model_size, self.default)

    def add_user(self, user: StarletteUser) -> None:
        self.default.add_user(user)
        for metadata in self.by_size.values():
            metadata.add_user(user)


def create_model_metadata_by_size(
    data: dict[str, Any] | None, mock_model_responses: bool = False
) -> ModelMetadataBySize:
    """Create a ModelMetadataBySize from request data, enriching with size preferences from YAML config.

    If the data contains a `feature_setting`, looks up `models_for_size_preference` in the YAML
    config and creates ModelMetadata objects for each size bucket.
    """
    if not data or "provider" not in data:
        raise ValueError("Argument error: provider must be present.")

    # Read feature_setting before create_model_metadata pops it from data
    feature_setting = data.get("feature_setting")

    default_metadata = create_model_metadata(data, mock_model_responses)

    by_size: Dict[Literal["small", "large"], TypeModelMetadata] = {}
    if feature_setting:
        configs = ModelSelectionConfig.instance()
        unit_primitive_config = configs.get_unit_primitive_config_map().get(
            feature_setting
        )
        if unit_primitive_config:
            for (
                size,
                model_id,
            ) in unit_primitive_config.models_for_size_preference.items():
                size_data: Dict[str, Any] = {"provider": "gitlab", "name": model_id}
                by_size[size] = create_model_metadata(size_data, mock_model_responses)

    return ModelMetadataBySize(default=default_metadata, by_size=by_size)


def _create_fireworks_metadata(
    data: dict[str, Any], mock_model_responses: bool
) -> FireworksModelMetadata:
    configs = ModelSelectionConfig.instance()
    llm_definition = configs.get_model(data["name"]) if data.get("name") else None

    if not llm_definition:
        raise ValueError(f"No LLM definition found for Fireworks model {data['name']}.")

    # Fireworks models using a router have a separate `identifier` distinct from
    # `model` (e.g. Codestral: model="codestral-2501", identifier=the router path).
    # Models deployed directly (no router indirection) only have `model`, which IS
    # the Fireworks deployment path. Fall back to it so every Fireworks model works
    # without requiring a redundant `identifier` field in models.yml.
    model_identifier = getattr(llm_definition.params, "identifier", None) or getattr(
        llm_definition.params, "model", None
    )

    if not model_identifier and not mock_model_responses:
        raise ValueError(
            f"Fireworks model identifier is missing for model {data['name']}."
        )

    provider_keys = data.get("provider_keys", {})

    return FireworksModelMetadata(
        provider="fireworks_ai",
        name=data["name"],
        endpoint=data.get("fireworks_api_base_url"),
        api_key=provider_keys.get("fireworks_provider_api_key"),
        model_identifier=model_identifier or "",
        using_cache=data.get("using_cache"),
        session_id=data.get("session_id"),
        llm_definition=llm_definition,
        friendly_name=llm_definition.name,
    )


def _create_mistral_metadata(data: dict[str, Any]) -> ModelMetadata:
    configs = ModelSelectionConfig.instance()
    llm_definition = configs.get_model(data["name"]) if data.get("name") else None

    if not llm_definition:
        raise ValueError(f"No LLM definition found for Mistral model {data['name']}.")

    model = llm_definition.params.model
    if not model:
        raise ValueError(
            f"Mistral model `{data['name']}` is missing `params.model` in models.yml."
        )

    provider_keys = data.get("provider_keys", {})

    return ModelMetadata(
        llm_definition=llm_definition,
        friendly_name=llm_definition.name,
        provider="mistral",
        name=data["name"],
        api_key=provider_keys.get("mistral_api_key"),
        # identifier uses LiteLLM's "provider/model" convention expected by
        # ModelMetadata.to_params
        identifier=f"mistral/{model.removeprefix('mistral/')}",
    )


def _create_gitlab_metadata(data: dict[str, Any]) -> ModelMetadata:
    configs = ModelSelectionConfig.instance()

    if name := data.get("name"):
        llm_definition = configs.get_model(name)
    else:
        # When there's no name it means we're in presence of a GitLab-provider metadata, which may pass a
        # "feature_setting" or "identifier" to deduce the model from them. These values are not expected in the rest
        # of the code (and in the case of `identifier`, it has a different purpose that non-gitlab identifiers), so
        # we pop them before any comparisons.
        feature_setting = data.pop("feature_setting", None)
        identifier = data.pop("identifier", None)

        if identifier:
            llm_definition = configs.get_model(identifier)
        elif feature_setting:
            llm_definition = configs.get_model_for_feature(feature_setting)
        else:
            raise ValueError(
                "Argument error: either identifier or feature_setting must be present."
            )

        data["name"] = llm_definition.gitlab_identifier

    return ModelMetadata(
        llm_definition=llm_definition,
        friendly_name=llm_definition.name,
        **data,
    )


def create_model_metadata(
    data: dict[str, Any] | None, mock_model_responses: bool = False
) -> TypeModelMetadata:
    if not data or "provider" not in data:
        raise ValueError("Argument error: provider must be present.")

    configs = ModelSelectionConfig.instance()
    provider = data["provider"]

    if provider == "amazon_q":
        llm_definition = configs.get_model("amazon_q")
        return AmazonQModelMetadata(
            llm_definition=llm_definition,
            friendly_name=llm_definition.name,
            **data,
        )

    if provider == "fireworks_ai":
        return _create_fireworks_metadata(data, mock_model_responses)
    if provider == "mistral":
        return _create_mistral_metadata(data)

    return _create_gitlab_metadata(data)


def build_default_code_completions_metadata(
    fireworks_api_base_url: str,
    model_keys: Dict[str, Any],
    user: StarletteUser,
    using_cache: bool = True,
    mock_model_responses: bool = False,
) -> TypeModelMetadata:
    """Build the default model_metadata for ``code_completions``.

    Args:
        fireworks_api_base_url: The Fireworks API base URL from operator configuration.
        model_keys: Provider key mapping (e.g. ``{"fireworks_provider_api_key": "..."}``) used to
            extract the Fireworks API key.
        user: The authenticated Starlette user; ``global_user_id`` is used as the Fireworks session ID.
        using_cache: Whether prompt caching is enabled for this request. Defaults to ``True``.
        mock_model_responses: When ``True``, allows an empty model identifier for local testing.

    Returns:
        A ``FireworksModelMetadata`` when the configured default model uses the ``fireworks_ai``
        provider, otherwise a generic ``ModelMetadata`` for the ``gitlab`` provider.
    """
    llm_def = ModelSelectionConfig.instance().get_model_for_feature("code_completions")
    if getattr(llm_def.params, "custom_llm_provider", None) == "fireworks_ai":
        return create_model_metadata(
            {
                "provider": "fireworks_ai",
                "name": llm_def.gitlab_identifier,
                "fireworks_api_base_url": fireworks_api_base_url,
                "provider_keys": model_keys,
                "using_cache": using_cache,
                "session_id": user.global_user_id,
            },
            mock_model_responses=mock_model_responses,
        )
    return create_model_metadata(
        {"provider": "gitlab", "feature_setting": "code_completions"},
        mock_model_responses=mock_model_responses,
    )


def build_default_feature_setting_metadata(
    feature_setting: Optional[str],
    model_keys: Dict[str, Any],
    fireworks_api_base_url: str,
    identifier: Optional[str] = None,
    user: Optional[StarletteUser] = None,
    mock_model_responses: bool = False,
) -> TypeModelMetadata:
    """Build default metadata for a gitlab-provider request, injecting server-side provider configuration when the
    resolved model is Fireworks- or Mistral-backed.

    Resolution precedence:
    1. ``identifier``: the user's explicit model selection (dropdown).
    2. ``feature_setting``: YAML default for the feature.

    Dispatches directly to the typed metadata creators based on
    ``llm_def.params.custom_llm_provider`` so the provider decision is made exactly once.
    """
    configs = ModelSelectionConfig.instance()

    if identifier:
        llm_def = configs.get_model(identifier)
    elif feature_setting:
        llm_def = configs.get_model_for_feature(feature_setting)
    else:
        raise ValueError("Either identifier or feature_setting must be provided.")

    custom_llm_provider = getattr(llm_def.params, "custom_llm_provider", None)

    if custom_llm_provider == "fireworks_ai":
        return _create_fireworks_metadata(
            {
                "name": llm_def.gitlab_identifier,
                "fireworks_api_base_url": fireworks_api_base_url,
                "provider_keys": model_keys,
                "session_id": user.global_user_id if user else None,
            },
            mock_model_responses,
        )

    if custom_llm_provider == "mistral":
        return _create_mistral_metadata(
            {
                "name": llm_def.gitlab_identifier,
                "provider_keys": model_keys,
            }
        )

    # Non-Fireworks/Mistral: preserve original resolution semantics.
    gitlab_payload: dict[str, Any] = {"provider": "gitlab"}
    if identifier:
        gitlab_payload["identifier"] = identifier
    else:
        gitlab_payload["feature_setting"] = feature_setting
    return _create_gitlab_metadata(gitlab_payload)


_COMPLETION_CONTEXT_CAPPED_PROVIDERS = frozenset({"fireworks_ai", "vertex_ai"})
_COMPLETION_CONTEXT_MAX_PERCENT = 0.3


def completion_context_max_percent_for_model_metadata(
    model_metadata: TypeModelMetadata,
) -> Optional[float]:
    """Return the completion context cap (0.3) for Fireworks/Vertex models, otherwise None (use the engine default).

    Bounds the amount of context sent for these models so latency and cost stay in check.
    """
    custom_llm_provider = getattr(
        model_metadata.llm_definition.params, "custom_llm_provider", None
    )

    if custom_llm_provider in _COMPLETION_CONTEXT_CAPPED_PROVIDERS:
        return _COMPLETION_CONTEXT_MAX_PERCENT

    return None
