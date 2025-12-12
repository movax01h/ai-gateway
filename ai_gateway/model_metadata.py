from abc import abstractmethod
from contextvars import ContextVar
from typing import Annotated, Any, Dict, Literal, Optional

from pydantic import AnyUrl, BaseModel, StringConstraints, UrlConstraints

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.model_selection import LLMDefinition, ModelSelectionConfig


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
    model_identifier: str
    using_cache: Optional[bool] = None
    session_id: Optional[str] = None

    def to_params(self) -> Dict[str, Any]:
        params = {
            "model": self.model_identifier,
        }

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

                if provider == "bedrock":
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


def create_model_metadata(data: dict[str, Any] | None) -> Optional[TypeModelMetadata]:
    if not data or "provider" not in data:
        return None

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
        model_endpoints = data.get("model_endpoints", {})

        region_config = model_endpoints.get("fireworks_current_region_endpoint", {})
        model_config = region_config.get(fireworks_llm_definition.params["model"], {})

        model_identifier = model_config.get("identifier")
        if not model_identifier or model_identifier == "":
            raise ValueError(
                f"Fireworks model identifier is missing for model {data['name']}."
            )

        return FireworksModelMetadata(
            provider="fireworks_ai",
            name=data["name"],
            endpoint=model_config.get("endpoint"),
            api_key=provider_keys.get("fireworks_api_key"),
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


current_model_metadata_context: ContextVar[Optional[TypeModelMetadata]] = ContextVar(
    "current_model_metadata_context", default=None
)
