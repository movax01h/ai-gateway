from unittest.mock import patch

import pytest
from fastapi import FastAPI

from ai_gateway.app import get_app, get_config
from ai_gateway.config import Config, ConfigFastApi


def test_get_config():
    config = get_config()
    assert isinstance(config, Config)
    assert isinstance(config.fastapi, ConfigFastApi)


def test_get_app():
    app = get_app()
    assert isinstance(app, FastAPI)


class TestGetAppMockUsageQuotaServer:
    @pytest.fixture(name="mock_app_dependencies")
    def mock_app_dependencies_fixture(self):
        with (
            patch("ai_gateway.app.config") as mock_config,
            patch("lib.usage_quota.mock_server.start_mock_server") as mock_start,
            patch("ai_gateway.app.create_fast_api_server") as mock_create,
            patch("ai_gateway.app.setup_logging"),
            patch("ai_gateway.app.start_metrics_server"),
        ):
            mock_create.return_value = FastAPI()

            yield mock_config, mock_start

    def test_starts_when_enabled(self, mock_app_dependencies):
        mock_config, mock_start = mock_app_dependencies
        mock_config.mock_usage_credits = True
        mock_config.mock_usage_quota_server.port = 4567

        get_app()

        mock_start.assert_called_once_with(port=4567)

    @pytest.mark.parametrize("mock_usage_credits", [None, "false"])
    def test_does_not_start_when_disabled(
        self,
        mock_usage_credits,
        monkeypatch,
        mock_app_dependencies,
    ):
        if mock_usage_credits is None:
            monkeypatch.delenv("AIGW_MOCK_USAGE_CREDITS", raising=False)
        else:
            monkeypatch.setenv("AIGW_MOCK_USAGE_CREDITS", mock_usage_credits)

        mock_config, mock_start = mock_app_dependencies
        mock_config.mock_usage_credits = Config(_env_file=None).mock_usage_credits

        get_app()

        mock_start.assert_not_called()
