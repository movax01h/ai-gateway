from typing import Self

from google.genai import Client
from google.genai.types import HttpOptions
from langchain_google_genai import ChatGoogleGenerativeAI as _LCChatGoogleGenerativeAI
from langchain_google_genai._common import get_user_agent
from langchain_google_genai.chat_models import _is_gemini_3_or_later
from pydantic import model_validator

__all__ = ["ChatGoogleGenerativeAI", "connect_google_gen_vertex_ai"]


def connect_google_gen_vertex_ai(
    project: str, location: str, headers: dict[str, str] | None = None
) -> Client:
    _, user_agent = get_user_agent("ChatGoogleGenerativeAI")
    all_headers = {"User-Agent": user_agent}
    if headers:
        all_headers.update(headers)

    http_options = HttpOptions(headers=all_headers)

    return Client(
        vertexai=True,
        project=project,
        location=location,
        http_options=http_options,
    )


class ChatGoogleGenerativeAI(_LCChatGoogleGenerativeAI):
    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Overwrite the LangChain model validator to set the client manually in DI."""

        if self.temperature is not None and not 0 <= self.temperature <= 2.0:
            msg = "temperature must be in the range [0.0, 2.0]"
            raise ValueError(msg)

        if "temperature" not in self.model_fields_set and _is_gemini_3_or_later(
            self.model
        ):
            self.temperature = 1.0

        if self.top_p is not None and not 0 <= self.top_p <= 1:
            msg = "top_p must be in the range [0.0, 1.0]"
            raise ValueError(msg)

        if self.top_k is not None and self.top_k <= 0:
            msg = "top_k must be positive"
            raise ValueError(msg)

        return self
