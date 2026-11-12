"""Tests for ai.apps.aigw.main.run_app."""

import importlib
from unittest.mock import MagicMock, patch

import pytest

from ai.apps.aigw.app import get_app
from ai.apps.aigw.main import run_app
from ai_gateway.config import ConfigFastApi, ConfigTLS


def _make_config(tls_enabled: bool, cert_file=None, key_file=None, ssl_ciphers=""):
    """Build a minimal mock Config object for run_app tests."""
    config = MagicMock()
    config.fastapi = ConfigFastApi(
        api_host="0.0.0.0",
        api_port=5000,
        reload=False,
        tls=ConfigTLS(
            enabled=tls_enabled,
            cert_file=cert_file,
            key_file=key_file,
            ssl_ciphers=ssl_ciphers,
        ),
    )
    return config


class TestRunAppTLS:
    @patch("ai.apps.aigw.main.uvicorn.run")
    @patch("ai.apps.aigw.main.get_config")
    def test_tls_disabled_does_not_pass_ssl_kwargs(
        self, mock_get_config, mock_uvicorn_run
    ):
        mock_get_config.return_value = _make_config(tls_enabled=False)

        run_app()

        _, kwargs = mock_uvicorn_run.call_args
        assert kwargs["ssl_certfile"] is None
        assert kwargs["ssl_keyfile"] is None
        assert kwargs["ssl_ciphers"] == "TLSv1"

    @patch("ai.apps.aigw.main.uvicorn.run")
    @patch("ai.apps.aigw.main.get_config")
    def test_tls_enabled_passes_ssl_kwargs(
        self, mock_get_config, mock_uvicorn_run, tmp_path
    ):
        cert = tmp_path / "server.crt"
        key = tmp_path / "server.key"
        ciphers = "TLSv1.2"
        cert.touch()
        key.touch()
        mock_get_config.return_value = _make_config(
            tls_enabled=True,
            cert_file=str(cert),
            key_file=str(key),
            ssl_ciphers=str(ciphers),
        )

        run_app()

        _, kwargs = mock_uvicorn_run.call_args
        assert kwargs["ssl_certfile"] == str(cert)
        assert kwargs["ssl_keyfile"] == str(key)
        assert kwargs["ssl_ciphers"] == str(ciphers)


class TestRunAppValidation:
    @patch("ai.apps.aigw.main.uvicorn.run")
    @patch("ai.apps.aigw.main.get_config")
    def test_validates_config_before_starting_server(
        self, mock_get_config, mock_uvicorn_run
    ):
        config = _make_config(tls_enabled=False)
        mock_get_config.return_value = config

        run_app()

        config.validate_required.assert_called_once_with()
        mock_uvicorn_run.assert_called_once()

    @patch("ai.apps.aigw.main.uvicorn.run")
    @patch("ai.apps.aigw.main.get_config")
    def test_does_not_start_server_when_validation_fails(
        self, mock_get_config, mock_uvicorn_run
    ):
        config = _make_config(tls_enabled=False)
        config.validate_required.side_effect = ValueError("missing setting")
        mock_get_config.return_value = config

        with pytest.raises(ValueError, match="missing setting"):
            run_app()

        mock_uvicorn_run.assert_not_called()


@patch("ai.apps.aigw.main.uvicorn.run")
@patch("ai.apps.aigw.main.get_config")
def test_app_factory_path_resolves(mock_get_config, mock_uvicorn_run):
    mock_get_config.return_value = _make_config(tls_enabled=False)

    run_app()

    module_name, attribute = mock_uvicorn_run.call_args.args[0].split(":")
    assert getattr(importlib.import_module(module_name), attribute) is get_app
