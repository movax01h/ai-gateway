import unittest
from unittest.mock import MagicMock, PropertyMock, patch

import google.auth.impersonated_credentials
import google.oauth2.credentials
import pytest
from google.auth.credentials import Credentials, TokenState

from ai_gateway.auth.gcp import (
    _fetch_application_default_credentials,
    _fetch_id_token_credentials,
    _id_token_credentials_for,
    access_token,
    id_token,
)


class TestAccessToken(unittest.TestCase):
    def test_access_token_new_token(self):
        with patch(
            "ai_gateway.auth.gcp._fetch_application_default_credentials"
        ) as mock_fetch:
            mock_creds = MagicMock(spec=Credentials)
            type(mock_creds).token = PropertyMock(return_value="new_token")
            type(mock_creds).token_state = PropertyMock(return_value=TokenState.FRESH)
            mock_fetch.return_value = mock_creds

            token = access_token()

            mock_fetch.assert_called_once()
            assert token == "new_token"

    def test_access_token_expired_token(self):
        with patch(
            "ai_gateway.auth.gcp._fetch_application_default_credentials"
        ) as mock_fetch:
            mock_creds = MagicMock(spec=Credentials)
            type(mock_creds).token = PropertyMock(return_value="expired_token")
            type(mock_creds).token_state = PropertyMock(return_value=TokenState.STALE)
            mock_fetch.return_value = mock_creds

            access_token()

            self.assertEqual(mock_fetch.call_count, 2)

    def test_access_token_empty(self):
        with patch(
            "ai_gateway.auth.gcp._fetch_application_default_credentials"
        ) as mock_fetch:
            mock_creds = MagicMock(spec=Credentials)
            type(mock_creds).token = PropertyMock(return_value=None)
            mock_fetch.return_value = mock_creds

            with pytest.raises(
                ValueError, match="Failed to retrieve Google Cloud access token"
            ):
                access_token()


class TestFetchApplicationDefaultCredentials(unittest.TestCase):
    def test_fetch_application_default_credentials(self):
        with patch("google.auth.default") as mock_default:
            mock_creds = MagicMock(spec=Credentials)
            mock_default.return_value = (mock_creds, None)

            _fetch_application_default_credentials.cache_clear()
            _fetch_application_default_credentials()

            mock_default.assert_called_once_with(
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
            mock_creds.refresh.assert_called_once()


class TestIdToken(unittest.TestCase):
    def test_returns_fresh_token(self):
        with patch("ai_gateway.auth.gcp._fetch_id_token_credentials") as mock_fetch:
            mock_creds = MagicMock(spec=Credentials)
            type(mock_creds).token = PropertyMock(return_value="jwt")
            type(mock_creds).token_state = PropertyMock(return_value=TokenState.FRESH)
            mock_fetch.return_value = mock_creds

            assert id_token("https://api.anthropic.com") == "jwt"

            mock_fetch.assert_called_once_with("https://api.anthropic.com")
            mock_creds.refresh.assert_not_called()

    def test_refreshes_stale_token_in_place(self):
        with patch("ai_gateway.auth.gcp._fetch_id_token_credentials") as mock_fetch:
            mock_creds = MagicMock(spec=Credentials)
            type(mock_creds).token = PropertyMock(return_value="jwt")
            type(mock_creds).token_state = PropertyMock(return_value=TokenState.STALE)
            mock_fetch.return_value = mock_creds

            id_token("https://api.anthropic.com")

            mock_fetch.assert_called_once_with("https://api.anthropic.com")
            mock_creds.refresh.assert_called_once()

    def test_empty_token_raises(self):
        with patch("ai_gateway.auth.gcp._fetch_id_token_credentials") as mock_fetch:
            mock_creds = MagicMock(spec=Credentials)
            type(mock_creds).token = PropertyMock(return_value=None)
            type(mock_creds).token_state = PropertyMock(return_value=TokenState.FRESH)
            mock_fetch.return_value = mock_creds

            with pytest.raises(
                ValueError, match="Failed to retrieve workload identity token"
            ):
                id_token("https://api.anthropic.com")


class TestIdTokenCredentialsFor(unittest.TestCase):
    AUDIENCE = "https://api.anthropic.com"

    def test_user_credentials_raise(self):
        source = MagicMock(spec=google.oauth2.credentials.Credentials)
        with patch("google.auth.default", return_value=(source, None)):
            with pytest.raises(ValueError, match="Cannot mint identity tokens"):
                _id_token_credentials_for(self.AUDIENCE, MagicMock())

    def test_impersonated_credentials_mint_id_token_with_email(self):
        source = MagicMock(spec=google.auth.impersonated_credentials.Credentials)
        with (
            patch("google.auth.default", return_value=(source, None)),
            patch(
                "google.auth.impersonated_credentials.IDTokenCredentials"
            ) as mock_id_creds,
            patch("google.oauth2.id_token.fetch_id_token_credentials") as mock_fetch,
        ):
            creds = _id_token_credentials_for(self.AUDIENCE, MagicMock())

        mock_id_creds.assert_called_once_with(
            source, target_audience=self.AUDIENCE, include_email=True
        )
        mock_fetch.assert_not_called()
        assert creds is mock_id_creds.return_value

    def test_other_credentials_delegate_to_google_auth(self):
        source = MagicMock(spec=Credentials)
        request = MagicMock()
        with (
            patch("google.auth.default", return_value=(source, None)),
            patch("google.oauth2.id_token.fetch_id_token_credentials") as mock_fetch,
        ):
            creds = _id_token_credentials_for(self.AUDIENCE, request)

        mock_fetch.assert_called_once_with(self.AUDIENCE, request=request)
        assert creds is mock_fetch.return_value


class TestFetchIdTokenCredentials(unittest.TestCase):
    def setUp(self):
        _fetch_id_token_credentials.cache_clear()
        self.addCleanup(_fetch_id_token_credentials.cache_clear)

    def test_refreshes_once_and_caches_per_audience(self):
        with patch("ai_gateway.auth.gcp._id_token_credentials_for") as mock_for:
            first = _fetch_id_token_credentials("aud-a")
            again = _fetch_id_token_credentials("aud-a")
            other = _fetch_id_token_credentials("aud-b")

        assert first is again is mock_for.return_value
        assert other is mock_for.return_value
        assert mock_for.call_count == 2
        first.refresh.assert_called()
