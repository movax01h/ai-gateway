from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from anthropic import AsyncAnthropic
from httpx import AsyncClient, Limits

from ai_gateway.models import ModelMetadata
from ai_gateway.models.base import connect_anthropic
from ai_gateway.models.base_text import TextGenModelBase


@pytest.mark.asyncio
async def test_connect_anthropic():
    with patch("ai_gateway.models.base._DefaultAsyncHttpxClient") as mock_client:
        mock_http_client = MagicMock(spec=AsyncClient)
        mock_client.return_value = mock_http_client

        client = connect_anthropic()

        assert isinstance(client, AsyncAnthropic)
        mock_client.assert_called_once()

        limits_arg = mock_client.call_args[1]["limits"]
        assert isinstance(limits_arg, Limits)
        assert limits_arg.max_connections == 1000
        assert limits_arg.max_keepalive_connections == 100
        assert limits_arg.keepalive_expiry == 30


class TestTextGenBaseModel:
    class TestClass(TextGenModelBase):
        @property
        def metadata(self):
            return ModelMetadata(engine="vertex", name="code-gecko@002")

        async def generate(self, **kwargs):
            pass

    @mock.patch("ai_gateway.models.base.config.model_engine_concurrency_limits")
    def test_instrumentator(self, mock_config):
        mock_config.for_model.return_value = 7

        model = TestTextGenBaseModel.TestClass()
        instrumentator = model.instrumentator

        mock_config.for_model.assert_called_with(engine="vertex", name="code-gecko@002")
        assert instrumentator.concurrency_limit == 7
