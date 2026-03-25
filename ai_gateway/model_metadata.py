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
    family: list[str] = []

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
    friendly_name: Optional[Annotated[str, StringConstraints(max_length=255)]] = None

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
    friendly_name: Optional[Annotated[str, StringConstraints(max_length=255)]] = None
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
    friendly_name: Optional[Annotated[str, StringConstraints(max_length=255)]] = None

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
        else:
            # Set a default dummy key to avoid LiteLLM errors
            # See https://gitlab.com/gitlab-org/gitlab/-/issues/520512
            if params.get("custom_llm_provider", "") == "custom_openai":
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


def create_model_metadata(
    data: dict[str, Any] | None, mock_model_responses: bool = False
) -> TypeModelMetadata:
    if not data or "provider" not in data:
        raise ValueError("Argument error: provider must be present.")

    configs = ModelSelectionConfig.instance()

    if data["provider"] == "amazon_q":
        llm_definition = configs.get_model("amazon_q")
        return AmazonQModelMetadata(
            llm_definition=llm_definition,
            family=llm_definition.family,
            friendly_name=llm_definition.name,
            **data,
        )

    if data["provider"] == "fireworks_ai":
        fireworks_llm_definition = (
            configs.get_model(data["name"]) if data.get("name") else None
        )

        if not fireworks_llm_definition:
            raise ValueError(
                f"No LLM definition found for Fireworks model {data['name']}."
            )

        provider_keys = data.get("provider_keys", {})

        model_identifier: str | None = getattr(
            fireworks_llm_definition.params, "identifier", None
        )

        if (
            not model_identifier or model_identifier == ""
        ) and not mock_model_responses:
            raise ValueError(
                f"Fireworks model identifier is missing for model {data['name']}."
            )

        if not model_identifier:
            # Allow empty identifier when mock_model_responses is True
            model_identifier = ""

        return FireworksModelMetadata(
            provider="fireworks_ai",
            name=data["name"],
            endpoint=data.get("fireworks_api_base_url"),
            api_key=provider_keys.get("fireworks_provider_api_key"),
            model_identifier=model_identifier,
            using_cache=data.get("using_cache"),
            session_id=data.get("session_id"),
            llm_definition=fireworks_llm_definition,
            family=fireworks_llm_definition.family,
        )

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
        family=llm_definition.family,
        friendly_name=llm_definition.name,
        **data,
    )
