from unittest.mock import MagicMock, patch

import httpx

from ai_gateway.models.v2.chat_google_genai import connect_google_gen_vertex_ai


def test_connect_google_gen_vertex_ai_passes_httpx_async_client():
    """The httpx client is forwarded to `HttpOptions` so it's pooled across resolutions."""
    client = MagicMock(spec=httpx.AsyncClient)

    with patch("ai_gateway.models.v2.chat_google_genai.Client") as mock_client:
        connect_google_gen_vertex_ai(
            project="test-project",
            location="us-central1",
            http_client=client,
        )

    _, kwargs = mock_client.call_args
    assert kwargs["http_options"].httpx_async_client is client
