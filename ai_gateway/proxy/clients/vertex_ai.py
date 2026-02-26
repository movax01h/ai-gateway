import re
from typing import Any, Literal, Self, override

import fastapi
from fastapi import status
from pydantic import BaseModel, ValidationError

from ai_gateway.auth.gcp import access_token
from ai_gateway.models.anthropic import KindAnthropicModel
from ai_gateway.models.base import KindModelProvider
from ai_gateway.models.vertex_text import KindVertexTextModel
from ai_gateway.proxy.clients.base import (
    BaseProxyModelFactory,
    ProxyModel,
    extract_json_body,
)

_ALLOWED_HEADERS_TO_UPSTREAM = ["content-type"]

_ALLOWED_HEADERS_TO_DOWNSTREAM = ["content-type"]

_UPSTREAM_SERVICE = KindModelProvider.VERTEX_AI.value


class PathParams(BaseModel):
    model_name: str
    upstream_provider: Literal["google", "anthropic"]
    action: Literal[
        "predict",
        "serverStreamingPredict",
        "streamRawPredict",
        "generateContent",
        "streamGenerateContent",
    ]
    sse: str = ""

    @classmethod
    def try_from_request(cls, request: fastapi.Request) -> Self:
        url = request.url.__str__()

        match = re.search(
            r"/v1/projects/[\w-]+/locations/[\w-]+/publishers/([\w-]+)/models/([\w@.-]+):([\w-]+)(\?alt=sse)?",
            url,
        )

        if match is None:
            raise fastapi.HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Not found"
            )

        try:
            instance = cls(
                upstream_provider=match.group(1),  # type: ignore[arg-type]
                model_name=match.group(2),
                action=match.group(3),  # type: ignore[arg-type]
                sse=match.group(4) or "",
            )
        except (IndexError, ValidationError):
            raise fastapi.HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Not found"
            )

        return instance


def _get_upstream_path(path_params: PathParams, project: str, location: str) -> str:
    return (
        f"/v1/projects/{project}/locations/{location}"
        f"/publishers/{path_params.upstream_provider}/models/{path_params.model_name}"
        f":{path_params.action}{path_params.sse}"
    )


def _get_stream_flag(path_params: PathParams, json_body: Any) -> bool:
    if path_params.upstream_provider == "anthropic":
        return json_body.get("stream", False)
    else:
        return path_params.action in (
            "serverStreamingPredict",
            "streamGenerateContent",  # introduced for the Gemini models
        )


def _load_allowed_upstream_models() -> list[str]:
    """Return all allowed models including Google and Anthropic models.

    Note: The check for textmodels will be removed in the future.
    See: https://gitlab.com/gitlab-org/modelops/applied-ml/code-suggestions/ai-assist/-/issues/906
    """
    google_models = [el.value for el in KindVertexTextModel]
    anthropic_models = [
        el.value
        for el in KindAnthropicModel
        if el.value.endswith("-vertex") or "@" in el.value
    ]
    return google_models + anthropic_models


def _build_headers_to_upstream() -> dict[str, str]:
    return {"Authorization": f"Bearer {access_token()}"}


class VertexAIProxyModelFactory(BaseProxyModelFactory):
    def __init__(self, endpoint: str, project: str, location: str):
        self.endpoint = endpoint
        self.project = project
        self.location = location

        self._base_url = f"https://{self.endpoint}"

    @override
    async def factory(self, request: fastapi.Request) -> ProxyModel:
        path_params = PathParams.try_from_request(request)

        upstream_path = _get_upstream_path(path_params, self.project, self.location)

        json_body = await extract_json_body(request)
        stream = _get_stream_flag(path_params, json_body)

        allowed_upstream_models = _load_allowed_upstream_models()

        if path_params.model_name not in allowed_upstream_models:
            raise fastapi.HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported model"
            )

        return ProxyModel(
            base_url=self._base_url,
            model_name=path_params.model_name,
            upstream_path=upstream_path,
            stream=stream,
            upstream_service=_UPSTREAM_SERVICE,
            headers_to_upstream=_build_headers_to_upstream(),
            allowed_upstream_models=allowed_upstream_models,
            allowed_headers_to_upstream=_ALLOWED_HEADERS_TO_UPSTREAM,
            allowed_headers_to_downstream=_ALLOWED_HEADERS_TO_DOWNSTREAM,
        )
