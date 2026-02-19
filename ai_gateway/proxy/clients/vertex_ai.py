import re

import fastapi
from fastapi import status

from ai_gateway.auth.gcp import access_token
from ai_gateway.models.anthropic import KindAnthropicModel
from ai_gateway.models.base import KindModelProvider
from ai_gateway.models.vertex_text import KindVertexTextModel
from ai_gateway.proxy.clients.base import BaseProxyModelFactory, ProxyModel

_ALLOWED_HEADERS_TO_UPSTREAM = ["content-type"]

_ALLOWED_HEADERS_TO_DOWNSTREAM = ["content-type"]

_UPSTREAM_SERVICE = KindModelProvider.VERTEX_AI.value


def _extract_params_from_path(path: str) -> tuple[str, str, str]:
    match = re.search(
        "/v1/projects/.*/locations/.*/publishers/google/models/(.*):(predict|serverStreamingPredict)(\\?alt=sse)?",
        path,
    )

    try:
        assert match is not None

        model = match.group(1)
        action = match.group(2)
        sse_flag = match.group(3) or ""
    except (IndexError, AssertionError):
        raise fastapi.HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Not found"
        )

    return model, action, sse_flag


def _extract_upstream_path(request_path: str, project: str, location: str) -> str:
    model, action, sse_flag = _extract_params_from_path(request_path)

    return (
        f"/v1/projects/{project}/locations/{location}"
        f"/publishers/google/models/{model}:{action}{sse_flag}"
    )


def _extract_stream_flag(upstream_path: str) -> bool:
    _, action, _ = _extract_params_from_path(upstream_path)

    return action == "serverStreamingPredict"


def _extract_model_name(upstream_path: str) -> str:
    model, _, _ = _extract_params_from_path(upstream_path)

    return model


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

    async def factory(self, request: fastapi.Request) -> ProxyModel:
        upstream_path = _extract_upstream_path(
            request.url.__str__(), self.project, self.location
        )
        stream = _extract_stream_flag(upstream_path)

        allowed_upstream_models = _load_allowed_upstream_models()
        model_name = _extract_model_name(upstream_path)

        if model_name not in allowed_upstream_models:
            raise fastapi.HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported model"
            )

        return ProxyModel(
            base_url=self._base_url,
            model_name=model_name,
            upstream_path=upstream_path,
            stream=stream,
            upstream_service=_UPSTREAM_SERVICE,
            headers_to_upstream=_build_headers_to_upstream(),
            allowed_upstream_models=allowed_upstream_models,
            allowed_headers_to_upstream=_ALLOWED_HEADERS_TO_UPSTREAM,
            allowed_headers_to_downstream=_ALLOWED_HEADERS_TO_DOWNSTREAM,
        )
