from typing import Optional

import requests
from pydantic import SecretStr

__all__ = ["set_bearer_token"]


def set_bearer_token(session: requests.Session, api_key: Optional[SecretStr]) -> None:
    """Send the API key as a bearer token on every request made through the session.

    A GitLab instance acting as the Snowplow collector authenticates the gateway with it. Nothing is set when no API key
    is configured, so public collectors keep working unchanged.

    Args:
        session: The requests session the Snowplow emitter sends its batches through.
        api_key: The API key the collector expects, or None when the collector needs no authentication.
    """
    token = api_key.get_secret_value() if api_key else None
    if not token:
        return

    session.headers["Authorization"] = f"Bearer {token}"
