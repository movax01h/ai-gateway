from abc import abstractmethod
from contextvars import ContextVar
from typing import Annotated, Any, Dict, Literal, Optional

from pydantic import AnyUrl, BaseModel, StringConstraints, UrlConstraints

from ai_gateway.api.auth_utils import StarletteUser


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
    endpoint: Annotated[AnyUrl, UrlConstraints(max_length=255)]
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


TypeModelMetadata = ModelMetadata | AmazonQModelMetadata

current_model_metadata_context: ContextVar[Optional[TypeModelMetadata]] = ContextVar(
    "current_model_metadata_context", default=None
)
