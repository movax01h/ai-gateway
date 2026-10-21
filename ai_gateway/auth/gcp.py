from functools import lru_cache

import google.auth
import google.auth.impersonated_credentials
import google.auth.transport.requests
import google.oauth2.credentials
import google.oauth2.id_token
import structlog
from google.auth.credentials import Credentials, TokenState

log = structlog.stdlib.get_logger("auth")

# Workload identity federation with service account impersonation rejects a
# token request that carries no scope, so ADC must be scoped explicitly.
_SCOPES = ["https://www.googleapis.com/auth/cloud-platform"]


def access_token() -> str:
    """Get access token from Google Application Default Credentials (ADC).

    See https://google-auth.readthedocs.io/en/latest/user-guide.html#application-default-credentials
    """

    creds = _fetch_application_default_credentials()

    if creds.token_state is not TokenState.FRESH:
        _fetch_application_default_credentials.cache_clear()
        creds = _fetch_application_default_credentials()

    # Ensure we always return a string
    token = creds.token
    if token is None:
        raise ValueError("Failed to retrieve Google Cloud access token")
    return token


@lru_cache(maxsize=1)
def _fetch_application_default_credentials() -> Credentials:
    creds, _ = google.auth.default(scopes=_SCOPES)
    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)
    return creds


def id_token(audience: str) -> str:
    """Get a Google-signed OIDC identity token for `audience` from ADC.

    Unlike `access_token()`, an identity token is a JWT that proves *who* the workload is
    rather than *what* it may call, so it can be presented to a third party that trusts
    `https://accounts.google.com` as an issuer, e.g. Anthropic Workload Identity Federation.

    The token comes from the credential type ADC resolved to: the metadata server, a service
    account key, or an `impersonated_service_account` ADC file. For local development, create
    the latter with `gcloud auth application-default login --impersonate-service-account=<email>`
    (needs `roles/iam.serviceAccountTokenCreator` on the target account). Plain user
    credentials cannot mint identity tokens for a custom audience.
    """

    creds = _fetch_id_token_credentials(audience)

    if creds.token_state is not TokenState.FRESH:
        creds.refresh(google.auth.transport.requests.Request())

    token = creds.token
    if token is None:
        raise ValueError("Failed to retrieve workload identity token")
    return token


@lru_cache(maxsize=8)
def _fetch_id_token_credentials(audience: str) -> Credentials:
    request = google.auth.transport.requests.Request()
    creds = _id_token_credentials_for(audience, request)
    creds.refresh(request)
    return creds


def _id_token_credentials_for(
    audience: str, request: google.auth.transport.requests.Request
) -> Credentials:
    source, _ = google.auth.default(scopes=_SCOPES)

    if isinstance(source, google.oauth2.credentials.Credentials):
        # The remediation is an operator task, so it goes to the logs and not into the exception, which can
        # surface to consumers invoking the model.
        log.error(
            "Google user credentials cannot mint identity tokens for a custom audience. "
            "Run `gcloud auth application-default login --impersonate-service-account=<email>`."
        )
        raise ValueError("Cannot mint identity tokens due to missing workload identity")

    if isinstance(source, google.auth.impersonated_credentials.Credentials):
        # `fetch_id_token_credentials` only reads `GOOGLE_APPLICATION_CREDENTIALS`, not the gcloud ADC file that
        # `gcloud auth application-default login --impersonate-service-account` writes, so wrap the impersonated
        # credentials directly. `include_email` is required for the federation rule's `email` claim.
        return google.auth.impersonated_credentials.IDTokenCredentials(
            source, target_audience=audience, include_email=True
        )

    # Service account keys and the metadata server are handled by google-auth itself.
    return google.oauth2.id_token.fetch_id_token_credentials(audience, request=request)
