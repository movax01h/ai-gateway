from typing import Any, Self, override

import httpx
from google.genai import Client
from google.genai.types import HttpOptions
from langchain_core.language_models.chat_models import _ChatModelBinding
from langchain_google_genai import ChatGoogleGenerativeAI as _LCChatGoogleGenerativeAI
from langchain_google_genai._common import get_user_agent
from langchain_google_genai.chat_models import _ClientCleanup, _is_gemini_3_or_later
from pydantic import model_validator

from ai_gateway.models.base import validate_custom_endpoint

__all__ = ["ChatGoogleGenerativeAI", "connect_google_gen_vertex_ai"]


def connect_google_gen_vertex_ai(
    project: str,
    location: str,
    headers: dict[str, str] | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> Client:
    _, user_agent = get_user_agent("ChatGoogleGenerativeAI")
    all_headers = {"User-Agent": user_agent}
    if headers:
        all_headers.update(headers)

    # `http_client` is a pooled client shared across every resolution (see
    # `ContainerModels.google_gen_vertex_ai_http_client`): google-genai never closes a
    # caller-provided client itself (`_api_client.py`'s `close`/`aclose` explicitly skip it),
    # so sharing it here is safe even though `ChatGoogleGenerativeAI.__del__` (langchain_google_genai)
    # closes `client` on every resolution's garbage collection. Passing `http_client` also
    # forces google-genai to use the httpx transport instead of aiohttp (its default when aiohttp
    # happens to be installed); unlike aiohttp, httpx binds to the event loop lazily at request
    # time, so this client can be built once at import/container-wiring time with no running loop.
    http_options = HttpOptions(headers=all_headers, httpx_async_client=http_client)

    return Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=http_options,
    )


class ChatGoogleGenerativeAI(_LCChatGoogleGenerativeAI):
    custom_models_enabled: bool = False
    """Whether custom model endpoints are allowed."""

    @model_validator(mode="after")
    @override
    def validate_environment(self) -> Self:
        """Overwrite the LangChain model validator to set the client manually in DI.

        LangChain's validator builds its own `google.genai.Client`, which would replace the Vertex
        client DI injects, so we copy its other steps here instead. Missing one breaks every
        request: 4.4.2 sets `_client_cleanup` in its validator and reads it in `async_client`:
        https://github.com/langchain-ai/langchain-google/blob/69361c5365fc6c78d5948e3468bc7f415166219a/libs/genai/langchain_google_genai/chat_models.py#L3503-L3528
        """

        # Leaves temperature out of the request, so Google applies its recommended default.
        if "temperature" not in self.model_fields_set and _is_gemini_3_or_later(
            self.model
        ):
            self.temperature = None

        if self.temperature is not None and not 0 <= self.temperature <= 2.0:
            msg = "temperature must be in the range [0.0, 2.0]"
            raise ValueError(msg)

        if self.top_p is not None and not 0 <= self.top_p <= 1:
            msg = "top_p must be in the range [0.0, 1.0]"
            raise ValueError(msg)

        if self.top_k is not None and self.top_k <= 0:
            msg = "top_k must be positive"
            raise ValueError(msg)

        # `async_client` reads this since langchain-google-genai 4.4.2.
        if self.client is not None:
            self._client_cleanup = _ClientCleanup(self.client)

        return self

    @override
    def bind(self, **kwargs: Any) -> _ChatModelBinding:
        validate_custom_endpoint(
            self.custom_models_enabled,
            api_base=kwargs.get("api_base"),
            api_key=kwargs.get("api_key"),
        )
        return super().bind(**kwargs)
