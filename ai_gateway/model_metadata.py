from abc import abstractmethod
from contextvars import ContextVar
from typing import Annotated, Any, Dict, Literal, Optional

from pydantic import AnyUrl, BaseModel, StringConstraints, UrlConstraints

from ai_gateway.api.auth_utils import StarletteUser
from ai_gateway.model_selection import ModelSelectionConfig


class BaseModelMetadata(BaseModel):
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

    def to_params(self) -> Dict[str, Any]:
        return {"role_arn": self.role_arn, "user": self._user}


class ModelMetadata(BaseModelMetadata):
    name: Annotated[str, StringConstraints(max_length=255)]
    provider: Annotated[str, StringConstraints(max_length=255)]
    endpoint: Optional[Annotated[AnyUrl, UrlConstraints(max_length=255)]] = None
    api_key: Optional[Annotated[str, StringConstraints(max_length=2000)]] = None
    identifier: Optional[Annotated[str, StringConstraints(max_length=1000)]] = None

    def to_params(self) -> Dict[str, Any]:
        params: Dict[str, str] = {}

        if self.endpoint:
            params["api_base"] = str(self.endpoint).removesuffix("/")
        if self.api_key:
            params["api_key"] = str(self.api_key)

        params["model"] = self.name
        params["custom_llm_provider"] = self.provider

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

        return params


TypeModelMetadata = AmazonQModelMetadata | ModelMetadata


def parameters_for_gitlab_provider(identifier) -> dict[str, Any]:
    """
    Retrieve model parameters for a given GitLab identifier.

    This function also allows setting custom provider details based on the identifier, like fetching endpoints based on
    AIGW location.
    """
    gitlab_model = ModelSelectionConfig().get_gitlab_model(identifier)

    return {
        "provider": gitlab_model.provider,
        "identifier": gitlab_model.provider_identifier,
        "name": gitlab_model.family or "base",
    }


def create_model_metadata(data: Dict[str, Any]) -> Optional[TypeModelMetadata]:
    if not data or "provider" not in data:
        return None

    match data["provider"]:
        case "amazon_q":
            return AmazonQModelMetadata(**data)
        case "gitlab":
            return ModelMetadata(**parameters_for_gitlab_provider(data["identifier"]))

    return ModelMetadata(**data)


current_model_metadata_context: ContextVar[Optional[TypeModelMetadata]] = ContextVar(
    "current_model_metadata_context", default=None
)
