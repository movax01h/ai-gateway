"""Tests for ai_gateway.main.run_app TLS integration."""

from unittest.mock import MagicMock, patch

from ai_gateway.config import ConfigFastApi, ConfigTLS
from ai_gateway.main import run_app


def _make_config(tls_enabled: bool, cert_file=None, key_file=None):
    """Build a minimal mock Config object for run_app tests."""
    config = MagicMock()
    config.fastapi = ConfigFastApi(
        api_host="0.0.0.0",
        api_port=5000,
        reload=False,
        tls=ConfigTLS(enabled=tls_enabled, cert_file=cert_file, key_file=key_file),
    )
    return config


class TestRunAppTLS:
    @patch("ai_gateway.main.uvicorn.run")
    @patch("ai_gateway.main.get_config")
    def test_tls_disabled_does_not_pass_ssl_kwargs(
        self, mock_get_config, mock_uvicorn_run
    ):
        mock_get_config.return_value = _make_config(tls_enabled=False)

        run_app()

        _, kwargs = mock_uvicorn_run.call_args
        assert kwargs["ssl_certfile"] is None
        assert kwargs["ssl_keyfile"] is None

    @patch("ai_gateway.main.uvicorn.run")
    @patch("ai_gateway.main.get_config")
    def test_tls_enabled_passes_ssl_kwargs(
        self, mock_get_config, mock_uvicorn_run, tmp_path
    ):
        cert = tmp_path / "server.crt"
        key = tmp_path / "server.key"
        cert.touch()
        key.touch()
        mock_get_config.return_value = _make_config(
            tls_enabled=True,
            cert_file=str(cert),
            key_file=str(key),
        )

        run_app()

        _, kwargs = mock_uvicorn_run.call_args
        assert kwargs["ssl_certfile"] == str(cert)
        assert kwargs["ssl_keyfile"] == str(key)
